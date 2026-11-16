"""Explicit read-only PortAudio discovery; call only outside the GUI/audio callback."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from .device_types import DeviceRef


@dataclass(frozen=True)
class DeviceInfo:
    """Reported capabilities, not proof of availability or supported stream formats."""

    ref: DeviceRef
    hostapi_name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    def __post_init__(self) -> None:
        if not isinstance(self.ref, DeviceRef):
            raise ValueError("DeviceInfo requires a validated DeviceRef")
        _text(self.hostapi_name, "Host API name")
        _count(self.max_input_channels, "Input channels")
        _count(self.max_output_channels, "Output channels")
        _samplerate(self.default_samplerate, "Default sample rate")


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a sequence")
    return value


def _record(value: object, field: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty name")
    return value


def _count(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _samplerate(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite sample rate")
    try:
        rate = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be a representable sample rate") from error
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"{field} must be a positive finite sample rate")
    return rate


def discover_devices() -> tuple[DeviceInfo, ...]:
    """Read and validate a fresh snapshot, without choosing devices or opening streams.

    Import/query failures propagate; inconsistent results require an explicit refresh.
    Separate PortAudio queries are not atomic. Revalidate selections before opening.
    """
    import sounddevice as sd

    devices = _sequence(sd.query_devices(), "Device list")
    hostapis = _sequence(sd.query_hostapis(), "Host API list")
    names: list[str] = []
    members: list[set[int]] = []
    for index, value in enumerate(hostapis):
        field = f"Host API {index}"
        info = _record(value, field)
        names.append(_text(info.get("name"), f"{field} name"))
        listed = _sequence(info.get("devices"), f"{field} devices")
        indices = tuple(_count(item, f"{field} device index") for item in listed)
        if len(set(indices)) != len(indices) or any(item >= len(devices) for item in indices):
            raise ValueError(f"{field} contains duplicate or missing device indices")
        members.append(set(indices))

    result: list[DeviceInfo] = []
    observed: list[set[int]] = [set() for _ in hostapis]
    for index, value in enumerate(devices):
        field = f"Device {index}"
        info = _record(value, field)
        hostapi = _count(info.get("hostapi"), f"{field} host API")
        if hostapi >= len(hostapis):
            raise ValueError(f"{field} refers to a missing host API")
        ref = DeviceRef(index, _text(info.get("name"), f"{field} name"), hostapi)
        result.append(DeviceInfo(
            ref=ref,
            hostapi_name=names[hostapi],
            max_input_channels=_count(info.get("max_input_channels"), f"{field} input channels"),
            max_output_channels=_count(info.get("max_output_channels"), f"{field} output channels"),
            default_samplerate=_samplerate(info.get("default_samplerate"), f"{field} sample rate"),
        ))
        observed[hostapi].add(index)
    if observed != members:
        raise ValueError("Device/host API snapshot is inconsistent; refresh devices")
    return tuple(result)
