"""In-memory interaction model; never opens devices or creates recordings.

Errors are raised as ``ValueError(<message key>)``; the UI translates the key.
"""

from dataclasses import dataclass, field


BUILTIN_PRESETS = ("original", "m2f", "f2m", "robot", "echo", "girl", "uncle", "monster")
INPUT_DEVICES = ("usb_mic", "onboard_in")
OUTPUT_DEVICES = ("headphones", "hdmi")
TRACKS = ("original", "effect")
FAULTS = ("input_lost", "output_lost", "busy", "write_failed")
MAX_CUSTOM_PRESETS = 8
MAX_PRESET_NAME = 24


@dataclass
class CustomPreset:
    name: str
    pitch: int
    formant: int
    reverb: int
    robot: int = 0
    echo: int = 0


@dataclass
class PrototypeState:
    input_device: str = ""
    output_device: str = ""
    monitoring: bool = False
    recording: bool = False
    playing: str = ""
    take: bool = False
    reviewed_save: bool = False
    incomplete: bool = False
    duration: int = 0
    position: int = 0
    preset: str = "original"
    custom: bool = False
    pitch: int = 0
    formant: int = 0
    reverb: int = 0
    robot: int = 0
    echo: int = 0
    input_gain: int = 0
    output_gain: int = 0
    fault: str = ""
    custom_presets: dict[str, CustomPreset] = field(default_factory=dict)
    next_custom_id: int = 1

    @property
    def active(self) -> bool:
        return self.monitoring or self.recording or bool(self.playing)

    @property
    def capturing(self) -> bool:
        return self.monitoring or self.recording

    @property
    def unsaved(self) -> bool:
        return self.take and not self.reviewed_save

    def select_device(self, kind: str, device: str) -> None:
        if self.active:
            raise ValueError("err_stop_before_device")
        choices = {"input": INPUT_DEVICES, "output": OUTPUT_DEVICES}.get(kind)
        if choices is None or (device and device not in choices):
            raise ValueError("err_unknown_device")
        setattr(self, f"{kind}_device", device)

    def toggle_monitor(self) -> None:
        if self.playing:
            raise ValueError("err_stop_playback")
        if not self.monitoring and not (self.input_device and self.output_device):
            raise ValueError("err_monitor_needs_devices")
        if self.fault:
            raise ValueError("err_reset_fault")
        self.monitoring = not self.monitoring

    def start_recording(self) -> None:
        if self.playing or self.recording:
            raise ValueError("err_stop_play_or_record")
        if not self.input_device:
            raise ValueError("err_record_needs_input")
        if self.fault:
            raise ValueError("err_reset_fault")
        if self.unsaved:
            raise ValueError("err_unsaved_take")
        self.recording = True
        self.take = False
        self.reviewed_save = False
        self.incomplete = False
        self.duration = 0
        self.position = 0

    def stop_recording(self) -> None:
        if self.recording:
            self.recording = False
            self.take = True

    def play(self, track: str) -> None:
        if track not in TRACKS:
            raise ValueError("err_unknown_track")
        if self.recording:
            raise ValueError("err_stop_recording")
        if not self.take or not self.output_device:
            raise ValueError("err_play_needs_take")
        if self.fault:
            raise ValueError("err_reset_fault")
        self.monitoring = False
        self.playing = track
        self.position = 0

    def stop_all(self) -> None:
        self.stop_recording()
        self.monitoring = False
        self.playing = ""
        self.position = 0

    def fail(self, reason: str) -> None:
        was_recording = self.recording
        self.stop_all()
        if was_recording:
            self.incomplete = True
        self.fault = reason

    def tick(self) -> None:
        if self.recording:
            self.duration += 1
        if self.playing:
            self.position += 1
            if self.position >= max(1, self.duration):
                self.playing = ""
                self.position = 0

    def select_preset(self, preset: str) -> None:
        if preset not in BUILTIN_PRESETS and preset not in self.custom_presets:
            raise ValueError("err_unknown_preset")
        self.preset = preset
        self.reset_effect()

    def reset_effect(self) -> None:
        # Built-in presets carry no unverified DSP values; custom presets restore what the user saved.
        saved = self.custom_presets.get(self.preset)
        if saved:
            self.pitch, self.formant, self.reverb = saved.pitch, saved.formant, saved.reverb
            self.robot, self.echo = saved.robot, saved.echo
        else:
            self.pitch = self.formant = self.reverb = self.robot = self.echo = 0
        self.custom = False

    def _checked_name(self, name: str, reserved: set[str], exclude: str = "") -> str:
        name = name.strip()
        if not name:
            raise ValueError("err_name_empty")
        if len(name) > MAX_PRESET_NAME:
            raise ValueError("err_name_too_long")
        taken = {n.casefold() for n in reserved}
        taken |= {p.name.casefold() for key, p in self.custom_presets.items() if key != exclude}
        if name.casefold() in taken:
            raise ValueError("err_name_duplicate")
        return name

    def save_custom_preset(self, name: str, reserved: set[str] = frozenset()) -> str:
        """Store current effect parameters (not gains) under a new name and select it."""
        if len(self.custom_presets) >= MAX_CUSTOM_PRESETS:
            raise ValueError("err_custom_limit")
        name = self._checked_name(name, reserved)
        key = f"custom-{self.next_custom_id}"
        self.next_custom_id += 1
        self.custom_presets[key] = CustomPreset(
            name, self.pitch, self.formant, self.reverb, self.robot, self.echo,
        )
        self.preset = key
        self.custom = False
        return key

    def rename_custom_preset(self, key: str, name: str, reserved: set[str] = frozenset()) -> None:
        if key not in self.custom_presets:
            raise ValueError("err_not_custom")
        self.custom_presets[key].name = self._checked_name(name, reserved, exclude=key)

    def delete_custom_preset(self, key: str) -> None:
        if key not in self.custom_presets:
            raise ValueError("err_not_custom")
        del self.custom_presets[key]
        if self.preset == key:
            # Keep the current parameters so deletion never changes the (simulated) sound.
            self.preset = "original"
            self.custom = bool(self.pitch or self.formant or self.reverb or self.robot or self.echo)
