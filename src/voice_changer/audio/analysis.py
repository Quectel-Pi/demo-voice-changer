"""Bounded PCM tap and background frequency/pitch analysis."""

import math
import threading

import numpy as np

from .analysis_status import AnalysisReading, AnalysisSnapshot


class SignalAnalyzer:
    """Analyze independent channels with a common, calibrated dBFS reference."""

    def __init__(self, sample_rate: int, frames: int, channels: int) -> None:
        if sample_rate <= 0 or frames < 8 or channels not in (1, 2):
            raise ValueError("Analysis needs a positive sample rate, eight frames and one or two channels")
        self.sample_rate = sample_rate
        self.frames = frames
        self.channels = channels
        self.window = np.hanning(frames)
        self.normalization = frames * float(np.dot(self.window, self.window))
        self.frequencies = np.fft.rfftfreq(frames, 1 / sample_rate)
        self.weights = np.full(len(self.frequencies), 2.0)
        self.weights[0] = 1.0
        if frames % 2 == 0:
            self.weights[-1] = 1.0
        self.masks = tuple(
            (self.frequencies >= low)
            & (self.frequencies <= high if inclusive else self.frequencies < high)
            for low, high, inclusive in (
                (20, 250, False), (250, 2000, False), (2000, 8000, True),
            )
        )

    def _pitch(self, samples: np.ndarray) -> tuple[float, float] | None:
        samples = samples - np.mean(samples)
        if math.sqrt(float(np.mean(samples * samples))) < 0.0001:
            return None
        max_lag = min(self.sample_rate // 50, self.frames - 2)
        if max_lag < 3:
            return None
        fft_size = 1 << (2 * self.frames - 1).bit_length()
        spectrum = np.fft.rfft(samples, fft_size)
        correlation = np.fft.irfft(spectrum * spectrum.conjugate(), fft_size)
        energy = np.concatenate(([0.0], np.cumsum(samples * samples)))
        lags = np.arange(1, max_lag + 2)
        difference = energy[self.frames - lags] + energy[self.frames] - energy[lags] - 2 * correlation[lags]
        difference = np.maximum(difference, 0)
        cumulative = np.cumsum(difference)
        normalized = np.ones(max_lag + 2)
        np.divide(difference * lags, cumulative, out=normalized[1:], where=cumulative > 0)
        first_lag = max(2, self.sample_rate // 2000)
        minimum_lag = math.ceil(self.sample_rate / 800)
        for lag in range(first_lag, max_lag + 1):
            if (
                normalized[lag] < 0.15
                and normalized[lag] <= normalized[lag - 1]
                and normalized[lag] < normalized[lag + 1]
            ):
                if lag < minimum_lag:
                    return None
                previous, current, following = normalized[lag - 1:lag + 2]
                denominator = previous - 2 * current + following
                adjustment = 0.5 * (previous - following) / denominator if denominator else 0.0
                frequency = self.sample_rate / (lag + max(-1.0, min(1.0, adjustment)))
                if not 49.75 <= frequency <= 804:
                    return None
                return max(50.0, min(800.0, frequency)), float(current)
        return None

    def _spectrally_consistent(self, pitch: float, magnitude: np.ndarray) -> bool:
        maximum = float(np.max(magnitude))
        if maximum == 0:
            return False
        bin_width = self.sample_rate / self.frames
        near_fundamental = np.abs(self.frequencies - pitch) <= 1.5 * bin_width
        if not near_fundamental.any() or float(np.max(magnitude[near_fundamental])) < maximum * 0.1:
            return False
        peaks = np.flatnonzero(
            (magnitude[1:-1] > magnitude[:-2])
            & (magnitude[1:-1] >= magnitude[2:])
            & (magnitude[1:-1] >= maximum * 0.25)
        ) + 1
        for frequency in self.frequencies[peaks]:
            if not 50 <= frequency <= 8000:
                continue
            harmonic = round(frequency / pitch)
            if harmonic < 1 or abs(frequency - harmonic * pitch) > max(1.5 * bin_width, 0.04 * pitch):
                return False
        return True

    def analyze(self, samples: np.ndarray) -> AnalysisReading:
        if samples.shape != (self.frames, self.channels) or not np.isfinite(samples).all():
            raise ValueError("Analysis received an invalid or non-finite PCM window")
        data = np.asarray(samples, dtype=np.float64)
        spectrum = np.fft.rfft(data * self.window[:, None], axis=0)
        magnitude = np.abs(spectrum)
        power = magnitude ** 2 * self.weights[:, None] / self.normalization
        bands = tuple(
            math.sqrt(float(np.mean(np.sum(power[mask], axis=0)))) if mask.any() else None
            for mask in self.masks
        )
        pitches = []
        for channel in range(self.channels):
            result = self._pitch(data[:, channel])
            if result is not None and self._spectrally_consistent(result[0], magnitude[:, channel]):
                pitches.append(result)
        pitch = None
        if pitches:
            if len(pitches) == 1 or max(value[0] for value in pitches) / min(value[0] for value in pitches) <= 2 ** (1 / 12):
                pitch = min(pitches, key=lambda value: value[1])[0]
        return AnalysisReading(bands=bands, pitch_hz=pitch, silent=not np.any(data))


class AudioAnalysis:
    """One PCM producer and one worker; overflow never waits for the audio callback."""

    def __init__(self, sample_rate: int, blocksize: int, channels: int, *, capture: bool) -> None:
        if sample_rate <= 0 or blocksize <= 0 or channels not in (1, 2):
            raise ValueError("Invalid live analysis format")
        self.sample_rate = sample_rate
        self.channels = channels
        self.capture = capture
        self.window_frames = max(8, round(sample_rate / 10))
        self.capacity = max(self.window_frames * 3, blocksize * 8)
        self.sources = 2 if capture else 1
        self._data = np.empty((self.capacity, self.sources, channels), dtype=np.float32)
        self._window = np.empty((self.window_frames, self.sources, channels), dtype=np.float32)
        self._written = 0
        self._read = 0
        self._dropped = 0
        self._accepting = False
        self._producer_error: Exception | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.snapshot: AnalysisSnapshot | None = None

    def start(self) -> None:
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("Create a new analysis worker for each audio session")
        thread = threading.Thread(target=self._run, name="voice-changer-analysis", daemon=False)
        self._accepting = True
        try:
            thread.start()
        except Exception:
            self._accepting = False
            raise
        self._thread = thread

    def push(self, first: np.ndarray, second: np.ndarray | None = None) -> None:
        if not self._accepting:
            return
        try:
            count = len(first)
            if count > self.capacity - (self._written - self._read):
                self._dropped += 1
                return
            start = self._written % self.capacity
            part = min(count, self.capacity - start)
            self._data[start:start + part, 0] = first[:part]
            self._data[:count - part, 0] = first[part:]
            if self.capture:
                if second is None:
                    raise ValueError("Capture analysis needs both input and processed output")
                self._data[start:start + part, 1] = second[:part]
                self._data[:count - part, 1] = second[part:]
            self._written += count
        except Exception as error:
            self._producer_error = error
            self._accepting = False

    def _pop_window(self) -> None:
        start = self._read % self.capacity
        part = min(self.window_frames, self.capacity - start)
        self._window[:part] = self._data[start:start + part]
        self._window[part:] = self._data[:self.window_frames - part]
        self._read += self.window_frames

    def _run(self) -> None:
        sequence = 0
        observed_drops = 0
        failed: list[str | None] = [None] * self.sources
        try:
            analyzer = SignalAnalyzer(self.sample_rate, self.window_frames, self.channels)
            while not self._stop.is_set():
                if self._producer_error is not None:
                    error, self._producer_error = self._producer_error, None
                    raise error
                dropped = self._dropped
                if dropped != observed_drops:
                    self._read = self._written
                    observed_drops = dropped
                    sequence += 1
                    failures = [
                        AnalysisReading(None, None, error=error) if error is not None else None
                        for error in failed
                    ]
                    self.snapshot = AnalysisSnapshot(
                        sequence, self.sample_rate,
                        failures[0] if self.capture else None,
                        failures[1] if self.capture else failures[0], dropped,
                    )
                elif self._written - self._read >= self.window_frames:
                    self._pop_window()
                    if self._dropped != observed_drops:
                        continue
                    readings = []
                    for index in range(self.sources):
                        if failed[index] is None:
                            try:
                                reading = analyzer.analyze(self._window[:, index])
                            except Exception as error:
                                failed[index] = f"{type(error).__name__}: {error}"
                        if failed[index] is not None:
                            reading = AnalysisReading(None, None, error=failed[index])
                        readings.append(reading)
                    sequence += 1
                    self.snapshot = AnalysisSnapshot(
                        sequence, self.sample_rate,
                        readings[0] if self.capture else None,
                        readings[1] if self.capture else readings[0],
                        observed_drops,
                    )
                else:
                    self._stop.wait(0.01)
        except Exception as error:
            self._accepting = False
            failure = AnalysisReading(None, None, error=f"{type(error).__name__}: {error}")
            self.snapshot = AnalysisSnapshot(
                sequence + 1, self.sample_rate, failure if self.capture else None,
                failure, self._dropped,
            )

    def close(self) -> None:
        self._accepting = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("Analysis worker did not stop")
            self._thread = None
