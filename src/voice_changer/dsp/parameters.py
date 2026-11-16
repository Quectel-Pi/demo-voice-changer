"""Validated demo candidates and a coalescing CPython control mailbox."""

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..audio.analysis import AudioAnalysis


@dataclass(frozen=True)
class EffectParameters:
    pitch_semitones: float = 0.0
    formant_scale: float = 1.0
    bypass: bool = True
    input_gain_db: float = 0.0
    output_gain_db: float = 0.0
    robot_mix: float = 0.0
    echo_mix: float = 0.0
    reverb_mix: float = 0.0
    denoise_strength: float = 0.0

    def __post_init__(self) -> None:
        for name, low, high in (
            ("pitch_semitones", -12.0, 12.0), ("formant_scale", 0.8, 1.25),
            ("input_gain_db", -24.0, 12.0), ("output_gain_db", -24.0, 12.0),
            ("robot_mix", 0.0, 1.0), ("echo_mix", 0.0, 0.4),
            ("reverb_mix", 0.0, 0.4),
            ("denoise_strength", 0.0, 1.0),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not low <= value <= high
            ):
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")
        if type(self.bypass) is not bool:
            raise ValueError("bypass must be a boolean")
        if self.bypass and (
            self.pitch_semitones != 0 or self.formant_scale != 1
            or self.robot_mix != 0 or self.echo_mix != 0 or self.reverb_mix != 0
        ):
            raise ValueError("Original audio requires unity pitch/formants and no effects")

    @property
    def pitch_scale(self) -> float:
        return 2.0 ** (self.pitch_semitones / 12.0)


# formant_scale is the net vocal-tract shift relative to the original voice
# (1.0 keeps it).  Adult female F0 is roughly 1.6-1.8x male and formants about
# 15-20 % higher, so pitch carries most of the change and formants stay modest.
PRESETS = {
    "original": EffectParameters(),
    "male_to_female": EffectParameters(7.0, 1.18, False),
    "female_to_male": EffectParameters(-6.0, 0.86, False),
    "robot": EffectParameters(bypass=False, robot_mix=0.8),
    "echo": EffectParameters(bypass=False, echo_mix=0.25),
    "reverb": EffectParameters(bypass=False, reverb_mix=0.2),
    "loli": EffectParameters(6.0, 1.25, False),
    "uncle": EffectParameters(-4.0, 0.9, False),
    "monster": EffectParameters(-8.0, 0.8, False, echo_mix=0.15),
}


class LiveControl:
    """Single GUI writer, DSP reader; CPython/GIL, not a hard-realtime guarantee."""

    def __init__(self, parameters: EffectParameters) -> None:
        if not isinstance(parameters, EffectParameters):
            raise ValueError("An explicit EffectParameters snapshot is required")
        self.requested = (0, parameters)
        self.applied_revision = -1
        self.levels: tuple[float, float] | None = None
        self.analysis: "AudioAnalysis | None" = None

    def submit(self, parameters: EffectParameters) -> None:
        if not isinstance(parameters, EffectParameters):
            raise ValueError("An explicit EffectParameters snapshot is required")
        self.requested = (self.requested[0] + 1, parameters)
