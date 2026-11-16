"""A bounded SPSC bridge; the device callback never waits for the DSP worker."""

import math
import threading
from typing import TYPE_CHECKING

import numpy as np

from ..dsp.denoise import StreamingDenoiser
from ..dsp.parameters import LiveControl
from ..dsp.rubberband import RubberBand
from .device_types import CaptureConfig, StreamConfig

if TYPE_CHECKING:
    from .recording import RecordingWriter


class RecordingTap:
    def __init__(self, writer: "RecordingWriter") -> None:
        self.writer = writer
        self.started = threading.Event()
        self.finished = threading.Event()
        self.start_frame = 0
        self.stop_frame: int | None = None


class FrameRing:
    """One producer/consumer each; publish cursors only after copying samples."""

    def __init__(self, capacity: int, channels: int) -> None:
        self.data = np.empty((capacity, channels), dtype=np.float32)
        self.capacity = capacity
        self.written = 0
        self.read = 0

    @property
    def available(self) -> int:
        return self.written - self.read

    @property
    def free(self) -> int:
        return self.capacity - self.available

    def push(self, samples: np.ndarray) -> bool:
        count = len(samples)
        if count > self.free:
            return False
        start = self.written % self.capacity
        first = min(count, self.capacity - start)
        self.data[start:start + first] = samples[:first]
        self.data[:count - first] = samples[first:]
        self.written += count
        return True

    def pop(self, output: np.ndarray) -> bool:
        count = len(output)
        if count > self.available:
            return False
        start = self.read % self.capacity
        first = min(count, self.capacity - start)
        output[:first] = self.data[start:start + first]
        output[first:] = self.data[:count - first]
        self.read += count
        return True


class LiveProcessor:
    def __init__(self, config: StreamConfig | CaptureConfig, control: LiveControl) -> None:
        if config.samplerate != 48000 or config.blocksize not in (256, 512, 1024) or config.channels not in (1, 2):
            raise ValueError("D1 demo DSP supports 48000 Hz, 256/512/1024 frames and 1 or 2 matching channels")
        self.config = config
        self.control = control
        self.capacity = config.blocksize * 64
        self.input = FrameRing(self.capacity, config.channels)
        self.output = FrameRing(self.capacity, config.channels)
        self._dry = FrameRing(self.capacity, config.channels)
        self._gained_dry = FrameRing(self.capacity, config.channels)
        self._wet = FrameRing(self.capacity, config.channels)
        self._input_block = np.zeros((config.blocksize, config.channels), dtype=np.float32)
        self._gained_input = np.empty_like(self._input_block)
        self._retrieve_block = np.empty_like(self._input_block)
        self._dry_block = np.empty_like(self._input_block)
        self._gained_dry_block = np.empty_like(self._input_block)
        self._wet_block = np.empty_like(self._input_block)
        self._mixed = np.empty_like(self._input_block)
        self._post_block = np.empty_like(self._input_block)
        self._record_block = np.empty_like(self._input_block)
        from .sounddevice_backend import output_limit
        self._record_ceiling = output_limit(config.output_ceiling)
        self._ramp = np.linspace(0, 1, config.blocksize, dtype=np.float32)[:, None]
        self._weights = np.empty((config.blocksize, 1), dtype=np.float32)
        self._finite = np.empty_like(self._input_block, dtype=np.bool_)
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._playing = False
        self._submitted = 0
        self._rendered_frames = 0
        self._recording: RecordingTap | None = None
        self._record_command: tuple[str, RecordingTap] | None = None
        self._pitch = 0.0
        self._formant_log = 0.0
        self._mix = 0.0
        self._input_gain = 1.0
        self._output_gain = 1.0
        self._smoothing = 1 - math.exp(-config.blocksize / (config.samplerate * 0.02))
        from ..dsp.streaming_effects import StreamingEffects
        self._effects = StreamingEffects(config.samplerate, config.channels, config.blocksize)
        self._denoiser = StreamingDenoiser(
            config.samplerate, config.channels, config.blocksize,
        )
        self._denoised_input = np.empty_like(self._input_block)
        self._dsp = RubberBand(config.samplerate, config.channels, config.blocksize)
        self._discard = self._dsp.start_delay
        try:
            # R3 output arrives in bursts; retain two native windows for worker jitter.
            self._startup_frames = max(4 * config.blocksize, 2 * self._dsp.start_pad)
            if self._startup_frames > self.capacity:
                raise RuntimeError("DSP startup reserve exceeds the bounded output buffer")
            remaining = self._dsp.start_pad
            while remaining:
                count = min(remaining, config.blocksize)
                self._dsp.process(self._input_block[:count])
                self._drain()
                remaining -= count
        except Exception:
            self._dsp.close()
            raise

    def start(self) -> None:
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("Create a new DSP processor for each listening session")
        thread = threading.Thread(target=self._run, name="voice-changer-dsp", daemon=False)
        thread.start()
        self._thread = thread

    def exchange(self, samples: np.ndarray, output: np.ndarray) -> bool:
        output.fill(0)
        if self.error is not None:
            return False
        if not self.input.push(samples):
            self.error = RuntimeError("DSP input buffer overflow; listening stopped")
            return False
        self._submitted += len(samples)
        if not self._playing:
            if self.output.available < self._startup_frames:
                if self._submitted > self.capacity:
                    self.error = RuntimeError("DSP produced no timely output during startup")
                    return False
                return True
            self._playing = True
        if not self.output.pop(output):
            self.error = RuntimeError("DSP output buffer underrun; listening stopped")
            return False
        return True

    def _drain(self) -> None:
        for _ in range(64):
            available = self._dsp.available
            if not available:
                return
            count = self._dsp.retrieve(self._retrieve_block, min(available, self.config.blocksize))
            discard = min(count, self._discard)
            self._discard -= discard
            if discard < count:
                samples = self._retrieve_block[discard:count]
                np.isfinite(samples, out=self._finite[:len(samples)])
                if not self._finite[:len(samples)].all():
                    raise RuntimeError("DSP produced non-finite samples")
                if not self._wet.push(samples):
                    raise RuntimeError("DSP wet buffer overflow")
        raise RuntimeError("DSP retrieve exceeded its bounded drain budget")

    def _gain_weights(self, target: float, current: float) -> float:
        gain = current + (target - current) * self._smoothing
        if abs(target - gain) < 0.000001:
            gain = target
        np.multiply(self._ramp, gain - current, out=self._weights)
        np.add(self._weights, current, out=self._weights)
        return gain

    def _process_block(self) -> None:
        command = self._record_command
        if command is not None:
            action, tap = command
            self._record_command = None
            if action == "start":
                tap.start_frame = self.input.written
                self._recording = tap
                tap.started.set()
            else:
                tap.stop_frame = self.input.written
        revision, target = self.control.requested
        self._pitch += (target.pitch_semitones - self._pitch) * self._smoothing
        target_log = math.log(target.formant_scale)
        self._formant_log += (target_log - self._formant_log) * self._smoothing
        settled = (
            abs(target.pitch_semitones - self._pitch) < 0.001
            and abs(target_log - self._formant_log) < 0.0001
        )
        if settled:
            self._pitch, self._formant_log = target.pitch_semitones, target_log
        self._dsp.set_scales(2 ** (self._pitch / 12), math.exp(self._formant_log))
        input_gain = 10 ** (target.input_gain_db / 20)
        self._input_gain = self._gain_weights(input_gain, self._input_gain)
        np.multiply(self._input_block, self._weights, out=self._gained_input)
        np.isfinite(self._gained_input, out=self._finite)
        if not self._finite.all():
            raise RuntimeError("Input gain produced non-finite samples")
        if not self._dry.push(self._input_block):
            raise RuntimeError("DSP original-audio alignment buffer overflow")
        self._denoiser.process(
            self._gained_input, self._denoised_input, target.denoise_strength,
        )
        np.isfinite(self._denoised_input, out=self._finite)
        if not self._finite.all():
            raise RuntimeError("Denoise produced non-finite samples")
        if not self._gained_dry.push(self._denoised_input):
            raise RuntimeError("DSP gained-audio alignment buffer overflow")
        self._dsp.process(self._denoised_input)
        self._drain()
        for _ in range(64):
            if min(self._wet.available, self._dry.available, self._gained_dry.available) < self.config.blocksize:
                return
            self._wet.pop(self._wet_block)
            self._dry.pop(self._dry_block)
            self._gained_dry.pop(self._gained_dry_block)
            mix = 0.0 if target.bypass and target.denoise_strength == 0.0 else 1.0
            np.multiply(self._ramp, mix - self._mix, out=self._weights)
            np.add(self._weights, self._mix, out=self._weights)
            np.subtract(self._wet_block, self._gained_dry_block, out=self._mixed)
            np.multiply(self._mixed, self._weights, out=self._mixed)
            np.add(self._mixed, self._gained_dry_block, out=self._mixed)
            self._mix = mix
            self._effects.process(self._mixed, self._post_block, target)
            output_gain = 10 ** (target.output_gain_db / 20)
            self._output_gain = self._gain_weights(output_gain, self._output_gain)
            np.multiply(self._post_block, self._weights, out=self._mixed)
            np.isfinite(self._mixed, out=self._finite)
            if not self._finite.all():
                raise RuntimeError("Effect/output gain produced non-finite samples")
            if not self.output.push(self._mixed):
                raise RuntimeError("DSP output buffer overflow")
            tap = self._recording
            if tap is not None:
                begin = max(self._rendered_frames, tap.start_frame)
                end = self._rendered_frames + self.config.blocksize
                if tap.stop_frame is not None:
                    end = min(end, tap.stop_frame)
                if begin < end:
                    start = begin - self._rendered_frames
                    stop = end - self._rendered_frames
                    np.clip(
                        self._mixed, -self._record_ceiling,
                        self._record_ceiling, out=self._record_block,
                    )
                    if not tap.writer.submit(
                        self._dry_block[start:stop], self._record_block[start:stop],
                        source_frame=begin, parameters=target, revision=revision,
                    ):
                        raise RuntimeError(
                            f"Recording rejected DSP samples: {tap.writer.error}"
                        ) from tap.writer.error
                if tap.stop_frame is not None and self._rendered_frames + self.config.blocksize >= tap.stop_frame:
                    self._recording = None
                    tap.finished.set()
            self._rendered_frames += self.config.blocksize
            if (
                settled and self._input_gain == input_gain
                and self._output_gain == output_gain and self._effects.settled
                and self._denoiser.settled
            ):
                self.control.applied_revision = revision
        raise RuntimeError("DSP render exceeded its bounded block budget")

    def _run(self) -> None:
        try:
            while not self._stop.is_set() and self.error is None:
                if self.input.pop(self._input_block):
                    self._process_block()
                else:
                    self._stop.wait(0.001)
        except Exception as error:
            self.error = error
        finally:
            self._dsp.close()

    def check_health(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"DSP worker failed: {self.error}") from self.error
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("DSP worker is not running")

    @property
    def recording_alignment(self) -> dict[str, int | str]:
        return {
            "native_start_pad_frames": self._dsp.start_pad,
            "native_trimmed_start_delay_frames": self._dsp.start_delay,
            "denoise_delay_frames": self._denoiser.latency,
            "parameter_events": "DSP render target snapshots; input/native pipeline and smoothing may lag source markers; not measured audible response times",
            "effect_algorithms": "spectral-gate denoise before native DSP; 80Hz sine ring modulation; 200ms echo with 0.25 feedback; 31/37/41/47ms parallel comb reverb with nominal 0.8s decay",
            "gain_stages": "input before denoise and native DSP; output after effects; original track pre-gain",
            "tail_policy": "drain paired source interval using continued input lookahead; no extended effect tail",
            "acoustic_alignment": "not measured; native delay trimming is not an end-to-end latency result",
        }

    def begin_recording(self, writer: "RecordingWriter") -> RecordingTap:
        self.check_health()
        if self._recording is not None or self._record_command is not None:
            raise RuntimeError("A recording boundary is already active or pending")
        tap = RecordingTap(writer)
        self._record_command = ("start", tap)
        if not tap.started.wait(timeout=2):
            raise RuntimeError("DSP did not acknowledge the recording start boundary")
        return tap

    def end_recording(self, tap: RecordingTap) -> None:
        self.check_health()
        if self._recording is not tap or self._record_command is not None:
            raise RuntimeError("Recording is not active or a boundary is pending")
        self._record_command = ("stop", tap)
        if not tap.finished.wait(timeout=2):
            self.check_health()
            raise RuntimeError("DSP did not finish the recorded source interval")

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("DSP worker did not stop; device restart is blocked")
            self._thread = None
        self._dsp.close()
