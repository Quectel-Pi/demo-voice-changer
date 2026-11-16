"""Input-owned recording and output-only playback; never imported for idle UI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import wave

from .device_types import CaptureConfig, OutputConfig
from .engine import EngineError, OperationFailure
from .sounddevice_backend import CallbackProgress, check_device_identity, output_limit

if TYPE_CHECKING:
    import numpy as np
    import sounddevice as sd
    from ..dsp.parameters import LiveControl
    from .analysis import AudioAnalysis
    from .live_processing import FrameRing, LiveProcessor, RecordingTap
    from .recording import RecordingGroup, RecordingWriter


class CaptureBackend:
    def __init__(
        self, config: CaptureConfig, control: LiveControl, *,
        monitoring: OutputConfig | None = None, output_confirmed: bool = False,
    ) -> None:
        self.config = config
        self.control = control
        self.monitoring = monitoring
        self.output_confirmed = output_confirmed
        self.stream: CaptureStream | None = None

    def open_stream(self) -> CaptureStream:
        if self.monitoring is not None and self.output_confirmed is not True:
            raise RuntimeError("Confirm safe output before monitoring")
        self.stream = CaptureStream(self.config, self.control, self.monitoring)
        return self.stream


class CaptureStream:
    def __init__(
        self, config: CaptureConfig, control: LiveControl, monitoring: OutputConfig | None,
    ) -> None:
        import numpy as np
        import sounddevice as sd

        self.config = config
        self.control = control
        self._np, self._sd = np, sd
        self._monitor_config = monitoring
        self._input: sd.InputStream | None = None
        self._output: sd.OutputStream | None = None
        self.processor: LiveProcessor | None = None
        self.analysis: AudioAnalysis | None = None
        self.writer: RecordingWriter | None = None
        self._tap: RecordingTap | None = None
        self._monitor_ring: FrameRing | None = None
        self._monitor_playing = False
        self._monitor_stopping = False
        self._shape = config.blocksize, config.channels
        self._render = np.empty(self._shape, dtype=np.float32)
        self._finite = np.empty(self._shape, dtype=np.bool_)
        self._magnitude = np.empty(self._shape, dtype=np.float32)
        self._ceiling = output_limit(config.output_ceiling)
        self._input_progress = CallbackProgress()
        self._output_progress = CallbackProgress()
        self._fault: Exception | None = None
        self._closed = False

    @property
    def monitoring(self) -> bool:
        return self._output is not None and not self._monitor_stopping

    def start(self) -> None:
        if self._closed or self._input is not None:
            raise RuntimeError("Create a new capture session after stopping")
        from .analysis import AudioAnalysis
        from .live_processing import LiveProcessor

        config = self.config
        check_device_identity(self._sd.query_devices, config.input_device, "input", config.channels)
        self._sd.check_input_settings(
            device=config.input_device.index, channels=config.channels,
            samplerate=config.samplerate, dtype="float32",
        )
        self.processor = LiveProcessor(config, self.control)
        self.processor.start()
        self.analysis = AudioAnalysis(
            config.samplerate, config.blocksize, config.channels, capture=True,
        )
        self.control.analysis = self.analysis
        self.analysis.start()
        self._input = self._sd.InputStream(
            device=config.input_device.index, channels=config.channels,
            samplerate=config.samplerate, blocksize=config.blocksize,
            dtype="float32", latency="low", callback=self._capture,
        )
        if (
            self._input.device != config.input_device.index
            or self._input.channels != config.channels or self._input.dtype != "float32"
            or self._input.samplerate != config.samplerate or self._input.blocksize != config.blocksize
        ):
            raise RuntimeError("Input backend changed the requested device or format")
        check_device_identity(self._sd.query_devices, config.input_device, "input", config.channels)
        self._input_progress.reset()
        self._input.start()
        if self._monitor_config is not None:
            self.enable_monitor(self._monitor_config, output_confirmed=True)
        self.check_health()

    def enable_monitor(self, config: OutputConfig, *, output_confirmed: bool) -> None:
        if output_confirmed is not True:
            raise RuntimeError("Confirm safe output before monitoring")
        if self._output is not None:
            raise RuntimeError("Monitoring already owns an output; stop it first")
        if (
            (config.samplerate, config.blocksize, config.channels, config.output_ceiling)
            != (self.config.samplerate, self.config.blocksize,
                self.config.channels, self.config.output_ceiling)
        ):
            raise RuntimeError("Monitor format must match the running capture session")
        from .live_processing import FrameRing

        check_device_identity(self._sd.query_devices, config.output_device, "output", config.channels)
        self._sd.check_output_settings(
            device=config.output_device.index, channels=config.channels,
            samplerate=config.samplerate, dtype="float32",
        )
        self._monitor_ring = FrameRing(config.blocksize * 64, config.channels)
        self._monitor_playing = False
        self._monitor_stopping = False
        try:
            self._output = self._sd.OutputStream(
                device=config.output_device.index, channels=config.channels,
                samplerate=config.samplerate, blocksize=config.blocksize,
                dtype="float32", latency="low", callback=self._monitor,
            )
            if (
                self._output.device != config.output_device.index
                or self._output.channels != config.channels or self._output.dtype != "float32"
                or self._output.samplerate != config.samplerate
                or self._output.blocksize != config.blocksize
            ):
                raise RuntimeError("Monitor backend changed the requested device or format")
            check_device_identity(self._sd.query_devices, config.output_device, "output", config.channels)
            self._output_progress.reset()
            self._output.start()
        except Exception as error:
            try:
                self.disable_monitor()
            except Exception as cleanup:
                raise EngineError("Monitor startup and cleanup failed", (
                    OperationFailure("start", error), OperationFailure("close", cleanup),
                )) from error
            raise

    def disable_monitor(self) -> None:
        self._monitor_stopping = True
        self._monitor_ring = None
        output = self._output
        if output is None:
            return
        failures = []
        try:
            output.stop()
            if output.active:
                raise RuntimeError("Monitor did not stop")
        except Exception as error:
            failures.append(OperationFailure("stop monitor", error))
        try:
            output.close()
            if not output.closed:
                raise RuntimeError("Monitor did not close")
        except Exception as error:
            failures.append(OperationFailure("close monitor", error))
        else:
            self._output = None
        if failures:
            raise EngineError("Monitor shutdown failed", tuple(failures))

    def start_recording(self, writer: RecordingWriter) -> None:
        if self.writer is not None or self.processor is None:
            raise RuntimeError("Recording requires one running capture session")
        self.writer = writer
        writer.alignment.update(self.processor.recording_alignment)
        writer.start()
        self._tap = self.processor.begin_recording(writer)

    def stop_recording(self) -> RecordingGroup:
        if self.writer is None or self._tap is None or self.processor is None:
            raise RuntimeError("There is no active recording to finalize")
        self.processor.end_recording(self._tap)
        group = self.writer.finish(complete=True)
        self.writer.check_health()
        if not group.complete:
            raise RuntimeError("Recording finalization is incomplete; temporary data retained")
        self.writer = None
        self._tap = None
        return group

    def _capture(self, samples: np.ndarray, frames: int, _time: object, status: object) -> None:
        if self._fault is None:
            try:
                if status:
                    raise RuntimeError("PortAudio reported an input callback status")
                if frames != self.config.blocksize or samples.shape != self._shape or samples.dtype != self._np.float32:
                    raise RuntimeError("Unexpected input callback format")
                self._np.isfinite(samples, out=self._finite)
                if not self._finite.all():
                    raise RuntimeError("Non-finite audio input")
                if self.processor is None:
                    raise RuntimeError("Capture has no DSP processor")
                if not self.processor.exchange(samples, self._render):
                    raise RuntimeError("Capture DSP exchange failed") from self.processor.error
                self._np.clip(self._render, -self._ceiling, self._ceiling, out=self._render)
                self._np.abs(samples, out=self._magnitude)
                input_peak = float(self._magnitude.max())
                self._np.abs(self._render, out=self._magnitude)
                self.control.levels = input_peak, float(self._magnitude.max())
                if self.analysis is not None:
                    self.analysis.push(samples, self._render)
                ring = self._monitor_ring
                if ring is not None and not ring.push(self._render):
                    raise RuntimeError("Monitor output buffer overflow")
                self._input_progress.advance()
            except Exception as error:
                self._fault = error
        if self._fault is not None:
            raise self._sd.CallbackAbort

    def _monitor(self, output: np.ndarray, frames: int, _time: object, status: object) -> None:
        output.fill(0)
        if self._monitor_stopping:
            raise self._sd.CallbackStop
        if self._fault is None:
            try:
                if status:
                    raise RuntimeError("PortAudio reported a monitor callback status")
                if frames != self.config.blocksize or output.shape != self._shape or output.dtype != self._np.float32:
                    raise RuntimeError("Unexpected monitor callback format")
                ring = self._monitor_ring
                if ring is None:
                    raise RuntimeError("Monitor has no output buffer")
                if not self._monitor_playing and ring.available >= 4 * self.config.blocksize:
                    self._monitor_playing = True
                if self._monitor_playing and not ring.pop(output):
                    raise RuntimeError("Monitor output buffer underrun")
                self._output_progress.advance()
            except Exception as error:
                self._fault = error
        if self._fault is not None:
            output.fill(0)
            raise self._sd.CallbackAbort

    def check_health(self) -> None:
        if self.writer is not None:
            self.writer.check_health()
        if self._fault is not None:
            raise RuntimeError(f"Capture callback failed: {self._fault}") from self._fault
        if self._input is None or not self._input.active or self._input.closed:
            raise RuntimeError("Capture input stream is not active")
        self._input_progress.check()
        if self.processor is None:
            raise RuntimeError("Capture DSP processor is missing")
        self.processor.check_health()
        if self._output is not None:
            if not self._output.active or self._output.closed:
                raise RuntimeError("Monitor output stream is not active")
            self._output_progress.check()

    def stop(self) -> None:
        failures = []
        self._monitor_stopping = True
        self._monitor_ring = None
        for name, stream in (("input", self._input), ("monitor", self._output)):
            if stream is None:
                continue
            try:
                stream.stop()
                if stream.active:
                    raise RuntimeError(f"{name} stream did not stop")
            except Exception as error:
                failures.append(OperationFailure(f"stop {name}", error))
        if failures:
            raise EngineError("Capture shutdown failed", tuple(failures))

    def close(self) -> None:
        failures = []
        self._monitor_stopping = True
        self._monitor_ring = None
        for name in ("_input", "_output"):
            stream = getattr(self, name)
            if stream is None:
                continue
            try:
                stream.close()
                if not stream.closed:
                    raise RuntimeError(f"{name} stream did not close")
            except Exception as error:
                failures.append(OperationFailure(f"close {name}", error))
            else:
                setattr(self, name, None)
        if self._input is None and self._output is None:
            if self.analysis is not None:
                try:
                    self.analysis.close()
                except Exception as error:
                    failures.append(OperationFailure("close analysis", error))
                else:
                    if self.control.analysis is self.analysis:
                        self.control.analysis = None
                    self.analysis = None
            if self.processor is not None:
                try:
                    self.processor.close()
                except Exception as error:
                    failures.append(OperationFailure("close DSP", error))
                else:
                    self.processor = None
            if self.processor is None and self.writer is not None:
                try:
                    self.writer.finish(complete=False, reason="Audio stopped before recording finalization")
                except Exception as error:
                    failures.append(OperationFailure("finalize partial recording", error))
                else:
                    self.writer = None
        if failures:
            raise EngineError("Capture resource release failed", tuple(failures))
        self._closed = True


class PlaybackBackend:
    def __init__(
        self, config: OutputConfig, path: Path, frames: int, *, output_confirmed: bool = False,
    ) -> None:
        self.config, self.path, self.frames = config, path, frames
        self.output_confirmed = output_confirmed
        self.stream: PlaybackStream | None = None

    def open_stream(self) -> PlaybackStream:
        if self.output_confirmed is not True:
            raise RuntimeError("Confirm safe output before playback")
        self.stream = PlaybackStream(self.config, self.path, self.frames)
        return self.stream


class PlaybackStream:
    def __init__(self, config: OutputConfig, path: Path, frames: int) -> None:
        import numpy as np
        import sounddevice as sd
        from .live_processing import FrameRing

        if type(frames) is not int or frames <= 0:
            raise ValueError("Playback requires a nonempty recorded track")
        self._np, self._sd = np, sd
        self.config, self.path = config, path
        self._remaining = frames
        self._ring = FrameRing(config.blocksize * 64, config.channels)
        self._reader: wave.Wave_read | None = None
        self._raw: sd.OutputStream | None = None
        self._shape = config.blocksize, config.channels
        self._magnitude = np.empty(self._shape, dtype=np.float32)
        self._ceiling = output_limit(config.output_ceiling)
        self._fault: Exception | None = None
        self._progress = CallbackProgress()
        self._eof = False
        self._done = False
        # Publish the peak and its callback sequence as one immutable snapshot.
        self.output_peak_sample: tuple[int, float] | None = None
        self.analysis: AudioAnalysis | None = None
        self._closed = False

    @property
    def output_peak(self) -> float | None:
        sample = self.output_peak_sample
        return sample[1] if sample is not None else None

    @property
    def finished(self) -> bool:
        return self._done and self._raw is not None and not self._raw.active

    def fill(self) -> None:
        if self._reader is None:
            raise RuntimeError("Playback reader is not open")
        for _ in range(64):
            count = min(self.config.blocksize, self._ring.free, self._remaining)
            if count == 0:
                if self._remaining == 0:
                    self._eof = True
                return
            data = self._reader.readframes(count)
            if len(data) != count * self.config.channels * 2:
                raise RuntimeError(f"Recorded WAV is truncated: {self.path}")
            samples = self._np.frombuffer(data, dtype="<i2").reshape(count, self.config.channels)
            if not self._ring.push(samples.astype(self._np.float32) / 32768):
                raise RuntimeError("Playback prefetch exceeded its bounded buffer")
            self._remaining -= count
        if self._remaining == 0:
            self._eof = True

    def start(self) -> None:
        if self._closed or self._reader is not None or self._raw is not None:
            raise RuntimeError("Create a new playback session for each track")
        config = self.config
        check_device_identity(self._sd.query_devices, config.output_device, "output", config.channels)
        self._sd.check_output_settings(
            device=config.output_device.index, channels=config.channels,
            samplerate=config.samplerate, dtype="float32",
        )
        self._reader = wave.open(str(self.path), "rb")
        if (
            self._reader.getframerate() != config.samplerate
            or self._reader.getnchannels() != config.channels
            or self._reader.getsampwidth() != 2 or self._reader.getcomptype() != "NONE"
            or self._reader.getnframes() != self._remaining
        ):
            raise RuntimeError("Recorded WAV no longer matches the recording format/interval")
        self.fill()
        from .analysis import AudioAnalysis
        self.analysis = AudioAnalysis(
            config.samplerate, config.blocksize, config.channels, capture=False,
        )
        self.analysis.start()
        self._raw = self._sd.OutputStream(
            device=config.output_device.index, channels=config.channels,
            samplerate=config.samplerate, blocksize=config.blocksize,
            dtype="float32", latency="low", callback=self._play,
        )
        if (
            self._raw.device != config.output_device.index or self._raw.channels != config.channels
            or self._raw.dtype != "float32" or self._raw.samplerate != config.samplerate
            or self._raw.blocksize != config.blocksize
        ):
            raise RuntimeError("Playback backend changed the requested device or format")
        check_device_identity(self._sd.query_devices, config.output_device, "output", config.channels)
        self._progress.reset()
        self._raw.start()
        self.check_health()

    def _play(self, output: np.ndarray, frames: int, _time: object, status: object) -> None:
        output.fill(0)
        if self._fault is None:
            try:
                if status:
                    raise RuntimeError("PortAudio reported a playback callback status")
                if frames != self.config.blocksize or output.shape != self._shape or output.dtype != self._np.float32:
                    raise RuntimeError("Unexpected playback callback format")
                count = min(frames, self._ring.available)
                if count < frames and not self._eof:
                    raise RuntimeError("Playback output buffer underrun")
                if count and not self._ring.pop(output[:count]):
                    raise RuntimeError("Playback buffer changed during consumption")
                self._np.clip(output, -self._ceiling, self._ceiling, out=output)
                self._np.abs(output, out=self._magnitude)
                peak = float(self._magnitude.max())
                self._progress.advance()
                self.output_peak_sample = (self._progress.count, peak)
                if self.analysis is not None:
                    self.analysis.push(output)
                self._done = self._eof and self._ring.available == 0
            except Exception as error:
                self._fault = error
        if self._fault is not None:
            output.fill(0)
            raise self._sd.CallbackAbort
        if self._done:
            raise self._sd.CallbackStop

    def check_health(self) -> None:
        if self._fault is not None:
            raise RuntimeError(f"Playback callback failed: {self._fault}") from self._fault
        if not self._done:
            if self._raw is None or not self._raw.active or self._raw.closed:
                raise RuntimeError("Playback output stream is not active")
            self._progress.check()

    def stop(self) -> None:
        if self._raw is not None:
            self._raw.stop()
            if self._raw.active:
                raise RuntimeError("Playback output did not stop")

    def close(self) -> None:
        if self._raw is not None:
            self._raw.close()
            if not self._raw.closed:
                raise RuntimeError("Playback output did not close")
            self._raw = None
        if self.analysis is not None:
            self.analysis.close()
            self.analysis = None
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self._closed = True
