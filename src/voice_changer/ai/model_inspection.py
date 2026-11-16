"""Identify explicitly selected local voice models without running inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat


@dataclass(frozen=True)
class ModelInspection:
    model_type: str
    description: str
    frames: int | None = None
    sha256: str | None = None


def _validate_regular_file(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"Cannot access model file: {path}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("Select a regular model file, not a symbolic link or directory")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect_rknn(path: Path, digest: str) -> ModelInspection:
    manifest_path = Path(f"{path}.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"RKNN model requires a valid sidecar manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("RKNN manifest must be a JSON object")
    if manifest.get("model_sha256") != digest:
        raise ValueError("RKNN model SHA-256 does not match its manifest")

    model_format = manifest.get("format")
    if model_format == "voice-changer-llvc-rknn-v1":
        from .llvc import _validate_rknn_package

        _validate_rknn_package(path)
        return ModelInspection(
            "llvc_rknn",
            "LLVC RKNN detected. This experimental backend is not accepted for live conversion.",
            sha256=digest,
        )
    if model_format == "voice-changer-hevc-rknn-v1":
        from .hevc_rknn import read_manifest

        manifest = read_manifest(path)
        return ModelInspection(
            "hevc_rknn",
            "HEVoicechanger RKNN / NPU package detected. Window and simulator accuracy are "
            "validated; live latency and quality still need listening acceptance.",
            frames=manifest.get("frames"),
            sha256=digest,
        )
    if model_format == "voice-changer-rvc-rknn-v1":
        from .rvc_rknn import _read_manifest

        _read_manifest(path)
        return ModelInspection(
            "rvc_rknn",
            "RVC synthesis RKNN detected. Live audio integration and NPU accuracy are not accepted.",
            sha256=digest,
        )
    raise ValueError(f"Unsupported RKNN manifest format: {model_format!r}")


def _inspect_onnx(path: Path, digest: str) -> ModelInspection:
    from .llvc import _onnx_session, _validate_onnx_schema

    session = _onnx_session(path)
    try:
        frames, _, _ = _validate_onnx_schema(session)
    except ValueError:
        pass
    else:
        delay_ms = 52 if frames == 832 else 104
        return ModelInspection(
            "llvc_onnx",
            f"LLVC ONNX detected, {delay_ms} ms model blocks. Interface checked; "
            "numerical equivalence and audio quality are not verified.",
            frames,
            digest,
        )

    from .hevc import validate_hevc_session

    try:
        validate_hevc_session(session)
    except ValueError:
        pass
    else:
        return ModelInspection(
            "hevc_onnx",
            "HEVoicechanger 16 kHz causal ONNX detected. Interface checked without inference; "
            "live conversion is not accepted.",
            sha256=digest,
        )

    from .rvc_frontend import RvcOnnxContentEncoder, RvcOnnxPitchExtractor
    from .rvc_onnx import RvcOnnxSynthesizer

    try:
        RvcOnnxSynthesizer(session)
    except ValueError:
        pass
    else:
        return ModelInspection(
            "rvc_onnx_synthesis",
            "RVC synthesis ONNX detected. The full live frontend/audio path is not accepted.",
            sha256=digest,
        )
    try:
        RvcOnnxContentEncoder(session)
    except ValueError:
        pass
    else:
        return ModelInspection(
            "rvc_onnx_content",
            "RVC content-encoder ONNX component detected; this is not a complete voice model.",
            sha256=digest,
        )
    try:
        RvcOnnxPitchExtractor(session)
    except ValueError:
        pass
    else:
        return ModelInspection(
            "rvc_onnx_pitch",
            "RVC pitch-extractor ONNX component detected; this is not a complete voice model.",
            sha256=digest,
        )
    raise ValueError(
        "ONNX graph does not match a supported LLVC graph or known RVC model component"
    )


def inspect_model_file(model_file: Path) -> ModelInspection:
    """Classify a local model by its validated interface or package metadata."""
    path = Path(model_file).expanduser()
    _validate_regular_file(path)
    suffix = path.suffix.lower()
    digest = _sha256_file(path)
    if suffix == ".onnx":
        return _inspect_onnx(path, digest)
    if suffix == ".rknn":
        return _inspect_rknn(path, digest)
    if suffix == ".pth":
        return ModelInspection(
            "pytorch_checkpoint",
            "PyTorch .pth file detected by extension only; RVC format is unverified. "
            "The file is not loaded or executed.",
            sha256=digest,
        )
    raise ValueError("Unsupported model file extension; select ONNX, RKNN, or RVC .pth")
