"""Truthful AI feature availability; models are never bundled or downloaded.

LLVC is enabled only with an explicit model path and optional runtime.
Nothing here imports, downloads or loads a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Feature = Literal["tts", "vc"]


@dataclass(frozen=True)
class PresetVoice:
    voice_id: str
    name: str
    language: str


@dataclass(frozen=True)
class Availability:
    feature: Feature
    available: bool
    reason: str


_UNAVAILABLE = {
    "tts": (
        "AI 朗读不可用：本版本未安装或接入朗读模型，不会生成音频。\n"
        "AI read-aloud unavailable: no text-to-speech model is installed or integrated; "
        "no audio will be generated."
    ),
    "vc": (
        "AI 变声不可用：未配置可用的变声模型或运行环境；不会用原声直通、DSP 或离线转换代替。\n"
        "AI voice conversion unavailable: no usable conversion model/runtime is configured; "
        "it is not replaced by passthrough, DSP or offline conversion."
    ),
}

_backends: dict[str, object] = {}


def availability(feature: Feature) -> Availability:
    if feature not in _UNAVAILABLE:
        raise ValueError(f"Unknown AI feature: {feature!r}")
    if feature in _backends:
        return Availability(feature, True, "")
    if feature == "vc":
        from .llvc import unavailable_reason
        reason = unavailable_reason()
        return Availability(feature, reason is None, f"{_UNAVAILABLE[feature]}\n{reason}" if reason else "")
    from .cosyvoice_tts import unavailable_reason
    reason = unavailable_reason()
    return Availability(
        feature, reason is None,
        f"AI 朗读不可用 / AI read-aloud unavailable:\n{reason}" if reason else "",
    )


def preset_voices() -> tuple[PresetVoice, ...]:
    """Installed preset TTS voices; empty until a TTS backend is registered."""
    return ()
