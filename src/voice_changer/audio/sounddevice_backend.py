"""Opt-in original-audio adapter; importing this module does not load devices."""

from __future__ import annotations

from enum import Enum
from collections.abc import Callable, Mapping
from time import monotonic
from typing import TYPE_CHECKING, Protocol

from .device_types import DeviceRef as DeviceRef, StreamConfig as StreamConfig

if TYPE_CHECKING:
    import numpy as np
    import sounddevice as sd
    from ..dsp.parameters import LiveControl
    from .analysis import AudioAnalysis
    from .live_processing import LiveProcessor


class CallbackProcessor(Protocol):
    error: Exception | None

    def start(self) -> None: ...

    def exchange(self, samples: np.ndarray, output: np.ndarray) -> bool: ...

    def check_health(self) -> None: ...

    def close(self) -> None: ...


ProcessorFactory = Callable[["StreamConfig", "LiveControl | None"], CallbackProcessor]


class CallbackFault(Enum):
    STATUS = "PortAudio reported a callback status"
    FORMAT = "Unexpected callback frame count, shape or dtype"
    NONFINITE = "Non-finite audio input"
    PROCESSING = "Audio callback processing failed"


def check_device_identity(
    query: Callable[[int, str], Mapping[str, object]], selection: DeviceRef,
    kind: str, channels: int,
) -> None:
    info = query(selection.index, kind)
    if info["name"] != selection.name or info["hostapi"] != selection.hostapi:
        raise RuntimeError(f"Selected {kind} device changed; reselect it")
    capacity = info[f"max_{kind}_channels"]
    if type(capacity) is not int or capacity < channels:
        raise RuntimeError(f"Selected {kind} device has too few channels")


def output_limit(value: float) -> np.float32:
    import numpy as np

    ceiling = np.float32(value)
    if float(ceiling) > value:
        ceiling = np.nextafter(ceiling, np.float32(0))
    if ceiling <= 0:
        raise ValueError("output_ceiling is too small for float32 audio")
    return ceiling


class CallbackProgress:
    def __init__(self) -> None:
        self.count = 0
        self.observed = 0
        self.last_progress = 0.0

    def reset(self) -> None:
        self.last_progress = monotonic()
        self.observed = self.count

    def advance(self) -> None:
        self.count += 1

    def check(self) -> None:
        now = monotonic()
        if self.count != self.observed:
            self.observed = self.count
            self.last_progress = now
        elif now - self.last_progress >= 2.0:
            raise RuntimeError(
                "Audio callback stopped advancing for 2 seconds; "
                "stop and reselect the input/output devices"
            )


class SoundDeviceBackend:
    def __init__(
        self, config: StreamConfig, *, output_confirmed: bool = False,
        control: LiveControl | None = None,
        processor_factory: ProcessorFactory | None = None,
    ) -> None:
        self.config = config
        self.output_confirmed = output_confirmed
        self.control = control
        self.processor_factory = processor_factory

    def open_stream(self) -> _SoundDeviceStream:
        if self.output_confirmed is not True:
            raise RuntimeError("Output requires explicit device and volume confirmation")
        return _SoundDeviceStream(self.config, self.control, self.processor_factory)


class _SoundDeviceStream:
    def __init__(
        self,
        config: StreamConfig,
        control: LiveControl | None = None,
        processor_factory: ProcessorFactory | None = None,
    ) -> None:
        import numpy as np
        import sounddevice as sd

        self._np = np
        self._sd = sd
        self._config = config
        self._shape = (config.blocksize, config.channels)
        self._finite = np.empty(self._shape, dtype=np.bool_)
        self._magnitude = np.empty(self._shape, dtype=np.float32)
        self._control = control
        self._processor_factory = processor_factory
        self._processor: CallbackProcessor | None = None
        self._analysis: AudioAnalysis | None = None
        self._ceiling = output_limit(config.output_ceiling)
        self._lower = -self._ceiling
        self._fault: CallbackFault | None = None
        self._callback_error: Exception | None = None
        self._callback_status = ""
        self._raw: sd.Stream | None = None
        self._closed = False
        self._progress = CallbackProgress()

    def _check_devices(self) -> None:
        config = self._config
        for selection, kind in (
            (config.input_device, "input"), (config.output_device, "output"),
        ):
            check_device_identity(self._sd.query_devices, selection, kind, config.channels)

    def start(self) -> None:
        if self._closed or self._raw is not None:
            raise RuntimeError("Stream cannot be started again; create a new stream")
        config = self._config
        self._check_devices()
        self._sd.check_input_settings(
            device=config.input_device.index, channels=config.channels,
            samplerate=config.samplerate, dtype="float32",
        )
        self._sd.check_output_settings(
            device=config.output_device.index, channels=config.channels,
            samplerate=config.samplerate, dtype="float32",
        )
        if self._control is not None or self._processor_factory is not None:
            if self._processor_factory is not None:
                self._processor = self._processor_factory(config, self._control)
            else:
                from .live_processing import LiveProcessor
                self._processor = LiveProcessor(config, self._control)
            self._processor.start()
        if self._control is not None:
            from .analysis import AudioAnalysis
            self._analysis = AudioAnalysis(
                config.samplerate, config.blocksize, config.channels, capture=True,
            )
            self._control.analysis = self._analysis
            self._analysis.start()
        self._raw = self._sd.Stream(
            device=(config.input_device.index, config.output_device.index),
            samplerate=config.samplerate, blocksize=config.blocksize,
            channels=(config.channels, config.channels), dtype="float32",
            latency="high" if self._processor_factory is not None else "low",
            callback=self._callback,
        )
        if (
            self._raw.device != (config.input_device.index, config.output_device.index)
            or self._raw.samplerate != config.samplerate
            or self._raw.blocksize != config.blocksize
            or self._raw.channels != (config.channels, config.channels)
            or self._raw.dtype != ("float32", "float32")
        ):
            raise RuntimeError("Audio backend changed the requested device or format")
        self._check_devices()
        self._progress.reset()
        self._raw.start()
        self.check_health()

    def stop(self) -> None:
        if self._raw is not None:
            self._raw.stop()
            if self._raw.active:
                raise RuntimeError("Audio backend did not stop the stream")

    def close(self) -> None:
        if self._raw is not None:
            self._raw.close()
            if not self._raw.closed:
                raise RuntimeError("Audio backend did not close the stream")
            self._raw = None
        if self._analysis is not None:
            self._analysis.close()
            if self._control is not None and self._control.analysis is self._analysis:
                self._control.analysis = None
            self._analysis = None
        if self._processor is not None:
            self._processor.close()
            self._processor = None
        self._closed = True

    def check_health(self) -> None:
        if self._fault is not None:
            error = RuntimeError(self._fault.value)
            if self._fault is CallbackFault.STATUS:
                error = RuntimeError(
                    f"{self._fault.value}: {self._callback_status}; "
                    f"rate={self._config.samplerate}, block={self._config.blocksize}, "
                    f"devices={self._config.input_device.name} -> {self._config.output_device.name}. "
                    "Stop and retry; if repeated, check device routing and system load."
                )
            if self._callback_error is not None:
                error = RuntimeError(f"{self._fault.value}: {self._callback_error}")
                raise error from self._callback_error
            raise error
        if self._raw is None or self._raw.closed or not self._raw.active:
            raise RuntimeError("Audio stream is not active")
        if self._processor is not None:
            self._processor.check_health()
        self._progress.check()

    def _callback(
        self, indata: np.ndarray, outdata: np.ndarray,
        frames: int, _time: object, status: object,
    ) -> None:
        outdata.fill(0)
        if self._fault is None:
            if status:
                self._callback_status = str(status)
                self._fault = CallbackFault.STATUS
            elif (
                frames != self._config.blocksize
                or indata.shape != self._shape or outdata.shape != self._shape
                or indata.dtype != self._np.float32 or outdata.dtype != self._np.float32
            ):
                self._fault = CallbackFault.FORMAT
            else:
                try:
                    self._np.isfinite(indata, out=self._finite)
                    if not self._finite.all():
                        self._fault = CallbackFault.NONFINITE
                    else:
                        if self._processor is not None:
                            if not self._processor.exchange(indata, outdata):
                                self._callback_error = self._processor.error
                                self._fault = CallbackFault.PROCESSING
                        else:
                            outdata[:] = indata
                        self._np.clip(
                            outdata, self._lower, self._ceiling, out=outdata,
                        )
                        if self._control is not None and self._fault is None:
                            self._np.abs(indata, out=self._magnitude)
                            input_peak = float(self._magnitude.max())
                            self._np.abs(outdata, out=self._magnitude)
                            self._control.levels = (input_peak, float(self._magnitude.max()))
                            if self._analysis is not None:
                                self._analysis.push(indata, outdata)
                except Exception as error:
                    self._callback_error = error
                    self._fault = CallbackFault.PROCESSING
        if self._fault is not None:
            outdata.fill(0)
            raise self._sd.CallbackAbort
        self._progress.advance()
