"""Offline Beatrice v2 preview: record a short clip, convert, play it once and save it.

The live path runs the engine in an audio worker, where the board measures about
0.92x real time and the GUI competes for the CPU. This module deliberately does
the opposite: record first, then convert the whole clip offline, so one of the
JVS speakers (including the low-pitch male candidates) can be auditioned even
when realtime conversion would underrun.

The model package and API are non-commercial only (JVS corpus terms); see
:mod:`voice_changer.ai.beatrice`. No model is loaded while importing this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import beatrice
from .hevc_preview import (
    FADE_SECONDS,
    HOST_RATE,
    MAX_PEAK,
    PREVIEW_SECONDS,
    PreviewError,
    play,
    preview_directory,
    record,
    resample,
    save_wav,
)

#: Device block the offline conversion feeds through the converter, in frames.
BLOCK_FRAMES = 1024
#: The engine converts one channel at a time for the preview.
CHANNELS = 1
#: Trailing silence appended before trimming, so the converter delay does not
#: cut the last spoken syllable.
TAIL_SECONDS = 0.5

__all__ = [
    "BLOCK_FRAMES",
    "CHANNELS",
    "PREVIEW_SECONDS",
    "PreviewError",
    "convert",
    "preview_file_name",
    "record",
    "play",
    "save_wav",
]


@dataclass(frozen=True)
class _ConverterConfig:
    """The slice of StreamConfig the Beatrice converter actually reads."""

    samplerate: int = HOST_RATE
    blocksize: int = BLOCK_FRAMES
    channels: int = CHANNELS


def convert(
    samples: np.ndarray, source_rate: int, engine_config, *, runtime=None,
) -> np.ndarray:
    """Convert a mono clip offline with the selected JVS speaker.

    Returns a mono float32 clip at ``source_rate`` with the same duration as the
    input (plus the engine's small algorithmic delay). Raises :class:`PreviewError`
    for anything the UI can show to the user as-is.
    """
    if engine_config is None:
        raise PreviewError("未配置 Beatrice 引擎 / the Beatrice engine is not configured")
    if (not isinstance(samples, np.ndarray) or samples.ndim != 1
            or samples.dtype != np.float32):
        raise PreviewError("录音必须是单声道 float32 / preview audio must be mono float32")
    if samples.size == 0:
        raise PreviewError("录音为空，请重试 / the recording is empty; try again")
    if not np.isfinite(samples).all():
        raise PreviewError("录音包含无效采样 / the recording contains invalid samples")
    if source_rate != HOST_RATE:
        samples = resample(samples, source_rate, HOST_RATE)
    if samples.size < HOST_RATE * 0.1:
        raise PreviewError("录音太短，至少 0.1 秒 / the recording is shorter than 0.1 s")

    converter = beatrice.BeatriceConverter(
        _ConverterConfig(), engine_config, runtime=runtime,
    )
    tail = np.zeros(int(round(TAIL_SECONDS * HOST_RATE)), dtype=np.float32)
    padded = np.concatenate((np.clip(samples, -1.0, 1.0), tail))
    blocks = -(-padded.size // BLOCK_FRAMES)
    remainder = blocks * BLOCK_FRAMES - padded.size
    if remainder:
        padded = np.concatenate((padded, np.zeros(remainder, dtype=np.float32)))

    parts = []
    for start in range(0, padded.size, BLOCK_FRAMES):
        block = np.ascontiguousarray(padded[start:start + BLOCK_FRAMES, None])
        parts.append(converter(block)[:, 0].copy())
    # The female voices append a Rubber Band formant stage, which holds the last
    # few blocks back; flush it so the tail padding is not silently cut.
    tail_audio = converter.flush()
    if tail_audio.size:
        parts.append(tail_audio[:, 0].copy())
    converter.close()
    converted = np.concatenate(parts)[:samples.size]
    if not np.isfinite(converted).all():
        raise PreviewError(
            "Beatrice 输出包含无效采样 / the Beatrice output contains invalid samples",
        )
    return _limit(converted)


def preview_file_name(engine_config) -> Path:
    """A unique WAV path for the selected speaker; the suffix never overwrites."""
    speaker = beatrice.clamp_speaker(engine_config.speaker) + 1
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return preview_directory() / f"beatrice-jvs{speaker:03d}-{stamp}.wav"


def _limit(samples: np.ndarray) -> np.ndarray:
    """Scale peaks below ``MAX_PEAK`` and apply short fades at both ends."""
    out = np.array(samples, dtype=np.float32, copy=True)
    np.clip(out, -1.0, 1.0, out=out)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > MAX_PEAK:
        out *= MAX_PEAK / peak
    fade = min(int(round(FADE_SECONDS * HOST_RATE)), out.size // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]
    return out
