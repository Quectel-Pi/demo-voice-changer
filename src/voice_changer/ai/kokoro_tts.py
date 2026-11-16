"""Local Kokoro preset synthesis; model loading occurs only in the worker."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from .cosyvoice_tts import GeneratedAudio, TtsError, write_wave

VOICES = {"female": 45, "xiaoxiao": 47, "male": 49}
TEXT_LIMIT = 200
_FILES = (
    "model.int8.onnx", "voices.bin", "tokens.txt", "lexicon-us-en.txt",
    "lexicon-zh.txt", "date-zh.fst", "phone-zh.fst", "number-zh.fst",
)


def unavailable_reason(model_dir: Path | None, python: str) -> str | None:
    if model_dir is None:
        return "未配置 Kokoro 模型 / Kokoro model is not configured"
    missing = [name for name in _FILES if not (model_dir / name).is_file()]
    if not (model_dir / "espeak-ng-data").is_dir():
        missing.append("espeak-ng-data")
    if missing:
        return "Kokoro 模型文件缺失 / Missing Kokoro files: " + ", ".join(missing)
    if not Path(python).is_file() or not os.access(python, os.X_OK):
        return "Kokoro Python 不可执行 / Kokoro Python is not executable"
    return None


def synthesize(model_dir: Path, text: str, voice: str) -> GeneratedAudio:
    reason = unavailable_reason(model_dir, sys.executable)
    if reason:
        raise TtsError(reason)
    text = text.strip()
    if not text or len(text) > TEXT_LIMIT:
        raise TtsError(f"Text must contain 1–{TEXT_LIMIT} characters")
    if voice not in VOICES:
        raise TtsError("Unknown Kokoro voice")
    try:
        import numpy as np
        import sherpa_onnx
    except ImportError as error:
        raise TtsError(f"Kokoro runtime is unavailable: {error}") from error
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(model_dir / "model.int8.onnx"),
                voices=str(model_dir / "voices.bin"),
                tokens=str(model_dir / "tokens.txt"),
                data_dir=str(model_dir / "espeak-ng-data"),
                lexicon=",".join(str(model_dir / name) for name in (
                    "lexicon-us-en.txt", "lexicon-zh.txt",
                )),
            ),
            num_threads=2, provider="cpu",
        ),
        rule_fsts=",".join(str(model_dir / name) for name in (
            "date-zh.fst", "phone-zh.fst", "number-zh.fst",
        )),
        max_num_sentences=1,
    )
    if not config.validate():
        raise TtsError("Invalid Kokoro configuration")
    try:
        tts = sherpa_onnx.OfflineTts(config)
        audio = tts.generate(text, sid=VOICES[voice], speed=1.0)
        samples = np.asarray(audio.samples, dtype=np.float32)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise TtsError(f"Kokoro synthesis failed: {error}") from error
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise TtsError("Kokoro returned invalid audio")
    return GeneratedAudio(samples, audio.sample_rate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate speech using local Kokoro preset voices.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--voice", choices=tuple(VOICES), required=True)
    parser.add_argument("--text-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        audio = synthesize(args.model_dir, args.text_file.read_text(encoding="utf-8"), args.voice)
        write_wave(args.output, audio)
    except (TtsError, OSError, UnicodeError) as error:
        print(f"AI TTS failed: {error}", file=sys.stderr)
        return 1
    print(f"Generated {audio.samples.size} samples at {audio.sample_rate} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
