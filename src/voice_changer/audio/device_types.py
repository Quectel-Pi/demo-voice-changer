"""Pure device references and explicit stream configuration; no backend imports."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DeviceRef:
    """A current-session PortAudio selection, never a persistent device ID."""

    index: int
    name: str
    hostapi: int

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("Device index must be an explicit nonnegative integer")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Device name must be explicit")
        if type(self.hostapi) is not int or self.hostapi < 0:
            raise ValueError("Host API index must be an explicit nonnegative integer")


@dataclass(frozen=True)
class StreamConfig:
    input_device: DeviceRef
    output_device: DeviceRef
    samplerate: int
    blocksize: int
    channels: int
    output_ceiling: float

    def __post_init__(self) -> None:
        for selection in (self.input_device, self.output_device):
            if not isinstance(selection, DeviceRef):
                raise ValueError("Both input and output device selections are required")
        _validate_format(self.samplerate, self.blocksize, self.channels, self.output_ceiling)


def _validate_format(samplerate: int, blocksize: int, channels: int, ceiling: float) -> None:
    for name, value in (("samplerate", samplerate), ("blocksize", blocksize), ("channels", channels)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(ceiling, bool) or not isinstance(ceiling, (float, int))
        or not math.isfinite(ceiling) or not 0 < ceiling <= 1
    ):
        raise ValueError("output_ceiling must be finite and in (0, 1]")


@dataclass(frozen=True)
class CaptureConfig:
    input_device: DeviceRef
    samplerate: int
    blocksize: int
    channels: int
    output_ceiling: float

    def __post_init__(self) -> None:
        if not isinstance(self.input_device, DeviceRef):
            raise ValueError("An explicit input device is required")
        _validate_format(self.samplerate, self.blocksize, self.channels, self.output_ceiling)


@dataclass(frozen=True)
class OutputConfig:
    output_device: DeviceRef
    samplerate: int
    blocksize: int
    channels: int
    output_ceiling: float

    def __post_init__(self) -> None:
        if not isinstance(self.output_device, DeviceRef):
            raise ValueError("An explicit output device is required")
        _validate_format(self.samplerate, self.blocksize, self.channels, self.output_ceiling)
