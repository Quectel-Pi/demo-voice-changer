"""Offline HEEVoicechanger preview: record a short clip, convert in one batch, play.

This is intentionally *not* the live conversion path. The clip is recorded to
memory and converted after recording stops, so the realtime budget measured on
the Pi M2 CPU (about 415 ms per 128 ms streaming block) does not apply. The
result is saved as a WAV file and played once through the selected output.

No model is loaded while importing this module.
"""

from __future__ import annotations

import os
import time
import wave
from pathlib import Path

import numpy as np

from .hevc import INPUT_NAME, OUTPUT_NAME, SAMPLE_RATE

PREVIEW_SECONDS = 5
HOST_RATE = 48000
MAX_PEAK = 0.95
FADE_SECONDS = 0.01


class PreviewError(RuntimeError):
    """A preview problem that can be shown to the user as-is."""


def preview_directory() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "voice-changer" / "previews"


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample a mono float32 clip with the streaming resampler plus a drain tail."""
    if samples.dtype != np.float32 or samples.ndim != 1:
        raise ValueError("Preview audio must be mono float32")
    if source_rate == target_rate:
        return samples.copy()
    from .resampling import StreamingResampler

    resampler = StreamingResampler(source_rate, target_rate)
    tail = np.zeros(max(256, resampler.taps * 2), dtype=np.float32)
    parts = [resampler.process(samples), resampler.process(tail)]
    output = np.concatenate([part for part in parts if part.size])
    expected = int(round(samples.size * target_rate / source_rate))
    return output[:expected].copy() if output.size >= expected else output


def convert(samples: np.ndarray, source_rate: int, model: Path) -> np.ndarray:
    """Convert one recorded clip and return audio at ``source_rate``."""
    if samples.size == 0:
        raise PreviewError("录音为空，请重试 / The recording is empty; try again")
    from .hevc import validate_hevc_session

    session = _session(model)
    validate_hevc_session(session)
    audio_16k = resample(np.ascontiguousarray(samples, dtype=np.float32), source_rate, SAMPLE_RATE)
    if audio_16k.size < SAMPLE_RATE // 10:
        raise PreviewError("录音太短，请录制约 5 秒 / The recording is too short; record about 5 s")
    output = session.run(None, {INPUT_NAME: audio_16k.reshape(1, 1, -1)})[0]
    converted = np.asarray(output, dtype=np.float32).reshape(-1)
    if converted.size == 0 or not np.isfinite(converted).all():
        raise PreviewError("模型输出无效 / The model returned invalid audio")
    return _limit(resample(converted, SAMPLE_RATE, source_rate))


def record(seconds: float = PREVIEW_SECONDS, *, device: int | None = None) -> np.ndarray:
    """Record one clip from the selected input device."""
    if not 0.5 <= seconds <= 15:
        raise ValueError("Preview recording must be between 0.5 and 15 seconds")
    import sounddevice as sd

    frames = int(round(seconds * HOST_RATE))
    try:
        captured = sd.rec(
            frames, samplerate=HOST_RATE, channels=1, dtype="float32", device=device,
        )
        sd.wait()
    except Exception as error:  # PortAudio errors are not PreviewError subclasses
        raise PreviewError(f"录音失败 / Recording failed: {error}") from error
    samples = np.asarray(captured, dtype=np.float32).reshape(-1)
    if samples.size != frames or not np.isfinite(samples).all():
        raise PreviewError("录音数据无效 / The captured audio is invalid")
    return samples


def play(samples: np.ndarray, *, device: int | None = None) -> None:
    """Play one clip through the selected output and wait for it to finish."""
    import sounddevice as sd

    try:
        sd.play(np.ascontiguousarray(samples, dtype=np.float32), samplerate=HOST_RATE,
                device=device, blocking=True)
    except Exception as error:
        raise PreviewError(f"播放失败 / Playback failed: {error}") from error


def save_wav(path: Path, samples: np.ndarray, rate: int = HOST_RATE) -> Path:
    """Write mono float32 audio as PCM16 without overwriting an existing file."""
    path = Path(path)
    if path.exists():
        counter = 2
        while path.with_name(f"{path.stem}-{counter}{path.suffix}").exists():
            counter += 1
        path = path.with_name(f"{path.stem}-{counter}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((pcm * 32767.0).astype("<i2").tobytes())
    return path


def preview_file_name(model: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return preview_directory() / f"hevc-{model.stem}-{stamp}.wav"


def _session(model: Path):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise PreviewError(
            "缺少 onnxruntime，无法离线试听 / onnxruntime is missing; offline preview is unavailable"
        ) from error
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model), providers=["CPUExecutionProvider"], sess_options=options,
    )


def _limit(samples: np.ndarray) -> np.ndarray:
    output = np.asarray(samples, dtype=np.float32).copy()
    peak = float(np.abs(output).max()) if output.size else 0.0
    if peak > MAX_PEAK:
        output *= MAX_PEAK / peak
    fade = min(int(FADE_SECONDS * HOST_RATE), output.size // 4)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        output[:fade] *= ramp
        output[-fade:] *= ramp[::-1]
    return output
