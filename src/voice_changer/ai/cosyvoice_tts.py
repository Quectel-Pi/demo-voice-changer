"""Opt-in local reference-audio TTS through a locally installed CosyVoice 3 runtime."""

from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import TYPE_CHECKING
import wave

from .profiles import ProfileStore, VoiceProfile, default_root

if TYPE_CHECKING:
    import numpy as np

MODEL_ENV = "VOICE_CHANGER_COSYVOICE_MODEL"
_ZERO_SHOT_INSTRUCTION = "You are a helpful assistant.<|endofprompt|>"
_REQUIRED_MODEL_ITEMS = (
    "cosyvoice3.yaml",
    "CosyVoice-BlankEN",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "llm.pt",
    "flow.pt",
    "hift.pt",
)


class TtsError(RuntimeError):
    """A local TTS configuration, model, or synthesis error."""


@dataclass(frozen=True)
class GeneratedAudio:
    samples: np.ndarray
    sample_rate: int


def configured_model_dir() -> Path | None:
    value = os.environ.get(MODEL_ENV)
    return Path(value).expanduser() if value else None


def unavailable_reason(model_dir: Path | None = None, python: str | None = None) -> str | None:
    selected = model_dir if model_dir is not None else configured_model_dir()
    if selected is None:
        return "未配置 CosyVoice 3 模型目录 / CosyVoice 3 model directory is not configured"
    try:
        validate_model_dir(selected)
    except TtsError as error:
        return str(error)
    interpreter = python or os.environ.get("VOICE_CHANGER_TTS_PYTHON", sys.executable)
    if interpreter != sys.executable:
        if not Path(interpreter).is_file() or not os.access(interpreter, os.X_OK):
            return "朗读 Python 不存在或不可执行 / Speech Python is missing or not executable"
        # Dependencies in another interpreter are checked by the isolated generation process.
        return None
    if importlib.util.find_spec("cosyvoice") is None:
        return "当前 Python 环境缺少 CosyVoice 运行时 / CosyVoice runtime is missing in this Python environment"
    return None


def _regular_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode)


def validate_model_dir(model_dir: Path) -> Path:
    model_dir = Path(model_dir).expanduser()
    try:
        root_info = model_dir.lstat()
    except OSError as error:
        raise TtsError(f"Cannot access local CosyVoice model directory: {model_dir}: {error}") from error
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise TtsError("CosyVoice model path must be a local directory, not a symbolic link")

    missing = []
    for name in _REQUIRED_MODEL_ITEMS:
        item = model_dir / name
        try:
            info = item.lstat()
        except OSError:
            missing.append(name)
            continue
        if stat.S_ISLNK(info.st_mode) or not (
            stat.S_ISDIR(info.st_mode) if name == "CosyVoice-BlankEN"
            else stat.S_ISREG(info.st_mode)
        ):
            raise TtsError(f"Invalid local CosyVoice model component: {item}")
    if missing:
        raise TtsError(
            "Incomplete local CosyVoice 3 model directory; missing: " + ", ".join(missing)
        )
    return model_dir


def _load_model(model_dir: Path):
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice3
    except ImportError as error:
        raise TtsError(
            "CosyVoice runtime is not installed in this Python environment"
        ) from error
    try:
        return CosyVoice3(
            str(model_dir), load_trt=False, load_vllm=False, fp16=False,
        )
    except (OSError, RuntimeError, ValueError, AssertionError) as error:
        raise TtsError(f"Could not load the local CosyVoice 3 model: {error}") from error


def _profile_from_store(profile: VoiceProfile) -> VoiceProfile:
    if not isinstance(profile, VoiceProfile) or profile.damaged:
        raise TtsError("Choose a valid local voice profile")
    store = ProfileStore(profile.directory.parent)
    found = next((item for item in store.list() if item.profile_id == profile.profile_id), None)
    if found is None or found.damaged or found.directory != profile.directory:
        raise TtsError("Voice profile is no longer a valid owned profile")
    if not found.prompt_text.strip():
        raise TtsError("The reference recording has no matching prompt text")
    if not _regular_file(found.reference_path):
        raise TtsError("The reference audio is missing or is not a regular file")
    return found


def _audio_array(output: object) -> np.ndarray:
    import numpy as np

    value = output
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    samples = np.asarray(value, dtype=np.float32)
    if samples.ndim == 2 and samples.shape[0] == 1:
        samples = samples[0]
    if samples.ndim != 1 or samples.size == 0:
        raise TtsError("CosyVoice returned an empty or non-mono audio chunk")
    if not np.isfinite(samples).all():
        raise TtsError("CosyVoice returned non-finite audio samples")
    return samples


def synthesize(
    profile: VoiceProfile,
    text: str,
    model_dir: Path | None = None,
    *,
    model_loader: Callable[[Path], object] | None = None,
) -> GeneratedAudio:
    """Generate speech from a stored reference and its exact recorded prompt.

    Importing this module does not import CosyVoice, PyTorch, or an audio runtime.
    Model loading happens only when this function is explicitly called.
    """
    text = text.strip() if isinstance(text, str) else ""
    if not text:
        raise TtsError("Enter text to synthesize")
    valid_profile = _profile_from_store(profile)
    selected_dir = model_dir if model_dir is not None else configured_model_dir()
    if selected_dir is None:
        raise TtsError(f"Set {MODEL_ENV} to an installed local CosyVoice 3 model directory")
    local_dir = validate_model_dir(selected_dir)
    loader = model_loader or _load_model
    model = loader(local_dir)
    sample_rate = getattr(model, "sample_rate", None)
    if type(sample_rate) is not int or sample_rate <= 0:
        raise TtsError("CosyVoice reported an invalid output sample rate")

    import numpy as np

    chunks = []
    try:
        generated = model.inference_zero_shot(
            text, _ZERO_SHOT_INSTRUCTION + valid_profile.prompt_text,
            str(valid_profile.reference_path),
            stream=False,
        )
        for item in generated:
            if not isinstance(item, dict) or "tts_speech" not in item:
                raise TtsError("CosyVoice returned an invalid inference result")
            chunks.append(_audio_array(item["tts_speech"]))
    except TtsError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        raise TtsError(f"CosyVoice reference-audio synthesis failed: {error}") from error
    if not chunks:
        raise TtsError("CosyVoice returned no audio")
    return GeneratedAudio(np.concatenate(chunks), sample_rate)


def write_wave(path: Path, audio: GeneratedAudio, *, overwrite: bool = False) -> Path:
    import numpy as np

    path = Path(path).expanduser()
    if type(audio.sample_rate) is not int or not 8000 <= audio.sample_rate <= 192000:
        raise TtsError("Output sample rate is outside the supported PCM WAV range")
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise TtsError("Refusing to save invalid audio samples")
    if path.is_symlink():
        raise TtsError("Refusing to replace a symbolic link")
    if not path.parent.is_dir():
        raise TtsError(f"Output directory does not exist: {path.parent}")
    if path.exists() and not overwrite:
        raise TtsError(f"Output already exists; choose a new path: {path}")
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".voice-changer-tts-", suffix=".wav", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with wave.open(raw, "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(audio.sample_rate)
                writer.writeframes(pcm.tobytes())
            raw.flush()
            os.fsync(raw.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    except (OSError, wave.Error) as error:
        raise TtsError(f"Could not write generated WAV {path}: {error}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use a local CosyVoice 3 model and an authorized local reference profile to synthesize speech."
    )
    parser.add_argument("--model-dir", type=Path, default=configured_model_dir())
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--profile-root", type=Path, default=default_root())
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--text")
    text.add_argument("--text-file", type=Path, help="UTF-8 text file; avoids exposing text in process arguments")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _cli_parser().parse_args(argv)
    try:
        profiles = ProfileStore(arguments.profile_root).list()
        profile = next(
            (item for item in profiles if item.profile_id == arguments.profile_id), None,
        )
        if profile is None:
            raise TtsError(f"Local profile not found: {arguments.profile_id}")
        text = arguments.text_file.read_text(encoding="utf-8") if arguments.text_file else arguments.text
        audio = synthesize(profile, text, arguments.model_dir)
        destination = write_wave(arguments.output, audio, overwrite=arguments.overwrite)
    except (TtsError, OSError, UnicodeError) as error:
        print(f"AI TTS failed: {error}", file=sys.stderr)
        return 1
    print(f"Generated {audio.samples.size} samples at {audio.sample_rate} Hz: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
