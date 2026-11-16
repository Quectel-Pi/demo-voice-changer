"""Small versioned preferences; never persist audio activity or device indices."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile

from ..audio.device_discovery import DeviceInfo
from ..audio.device_types import DeviceRef, _validate_format


@dataclass(frozen=True)
class DeviceIdentity:
    name: str
    hostapi_name: str

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v.strip() for v in (self.name, self.hostapi_name)):
            raise ValueError("Saved device identity requires device and backend names")

    def match(self, devices: tuple[DeviceInfo, ...], kind: str, channels: int) -> DeviceRef:
        if kind not in ("input", "output"):
            raise ValueError("Device kind must be input or output")
        if type(channels) is not int or channels <= 0:
            raise ValueError("Restore channels must be a positive integer")
        matches = [
            info.ref for info in devices
            if info.ref.name == self.name and info.hostapi_name == self.hostapi_name
            and (info.max_input_channels if kind == "input" else info.max_output_channels) >= channels
        ]
        if len(matches) != 1:
            raise ValueError(f"Saved {kind} device is unavailable or ambiguous; reselect explicitly")
        return matches[0]


@dataclass(frozen=True)
class AudioSettings:
    samplerate: int
    blocksize: int
    channels: int
    output_ceiling: float

    def __post_init__(self) -> None:
        _validate_format(self.samplerate, self.blocksize, self.channels, self.output_ceiling)


@dataclass(frozen=True)
class Preferences:
    input_device: DeviceIdentity | None = None
    output_device: DeviceIdentity | None = None
    audio: AudioSettings | None = None
    save_directory: str | None = None

    def __post_init__(self) -> None:
        for value in (self.input_device, self.output_device):
            if value is not None and not isinstance(value, DeviceIdentity):
                raise ValueError("Invalid saved device identity")
        if self.audio is not None and not isinstance(self.audio, AudioSettings):
            raise ValueError("Invalid saved audio configuration")
        if self.save_directory is not None and (
            not isinstance(self.save_directory, str) or not self.save_directory
            or not Path(self.save_directory).is_absolute()
        ):
            raise ValueError("Saved directory must be an absolute path")


def preferences_path() -> Path:
    root = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    if not root.is_absolute():
        raise ValueError("XDG_CONFIG_HOME must be an absolute path")
    return root / "voice-changer" / "preferences.json"


def check_directory(directory: str) -> None:
    path = Path(directory)
    if not path.is_dir() or not os.access(path, os.W_OK | os.X_OK):
        raise OSError(f"Saved directory is missing or not writable; choose again: {directory}")


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError(f"Duplicate preferences field: {key}")
        fields[key] = value
    return fields


class PreferencesStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> Preferences:
        try:
            with self.path.open("rb") as stream:
                data = stream.read(16385)
        except FileNotFoundError:
            return Preferences()
        if len(data) > 16384:
            raise ValueError("Preferences exceed the 16KiB limit")
        document = json.loads(data, object_pairs_hook=_unique_fields)
        if (
            not isinstance(document, dict)
            or set(document) != {"version", "input_device", "output_device", "audio", "save_directory"}
            or type(document["version"]) is not int or document["version"] != 1
        ):
            raise ValueError("Unknown or invalid preferences schema")
        values = {}
        for key, cls in (
            ("input_device", DeviceIdentity), ("output_device", DeviceIdentity), ("audio", AudioSettings),
        ):
            value = document[key]
            if value is not None:
                if not isinstance(value, dict):
                    raise ValueError(f"Invalid preferences field: {key}")
                try:
                    value = cls(**value)
                except TypeError as error:
                    raise ValueError(f"Invalid preferences field: {key}") from error
            values[key] = value
        return Preferences(**values, save_directory=document["save_directory"])

    def save(self, preferences: Preferences) -> None:
        payload = (json.dumps(
            {"version": 1, **asdict(preferences)}, ensure_ascii=False, allow_nan=False, indent=2,
        ) + "\n").encode("utf-8")
        if len(payload) > 16384:
            raise ValueError("Preferences exceed the 16KiB limit")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=".preferences-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink()
