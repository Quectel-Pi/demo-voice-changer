"""Fixed-shape RK3576 adapter for an explicitly selected RVC ONNX synthesizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

RVC_RKNN_MANIFEST_FORMAT = "voice-changer-rvc-rknn-v1"
RVC_SYNTHESIS_INPUTS = ("feats", "p_len", "pitch", "pitchf", "sid")
RVC_SYNTHESIS_OUTPUT = "audio"
_MODEL_SAMPLE_RATES = (32000, 40000, 48000)


class RknnRuntime(Protocol):
    def run(self, inputs: list[np.ndarray]) -> list[np.ndarray]: ...

    def close(self) -> None: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(model_file: Path) -> dict[str, object]:
    manifest_file = Path(f"{model_file}.json")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"RVC RKNN manifest is missing or invalid: {manifest_file}") from error
    if not isinstance(manifest, dict):
        raise ValueError("RVC RKNN manifest must be a JSON object")
    if (
        manifest.get("format") != RVC_RKNN_MANIFEST_FORMAT
        or manifest.get("target_platform") != "rk3576"
        or manifest.get("quantized") is not False
        or manifest.get("inputs") != list(RVC_SYNTHESIS_INPUTS)
        or manifest.get("outputs") != [RVC_SYNTHESIS_OUTPUT]
        or manifest.get("model_sha256") != _sha256_file(model_file)
    ):
        raise ValueError("RVC RKNN package does not match the RK3576 synthesis contract")

    frames = manifest.get("frames")
    channels = manifest.get("feature_channels")
    sample_rate = manifest.get("model_sample_rate")
    input_shapes = manifest.get("input_shapes")
    output_shapes = manifest.get("output_shapes")
    if (
        type(frames) is not int or frames < 1
        or channels not in (256, 768)
        or type(sample_rate) is not int or sample_rate not in _MODEL_SAMPLE_RATES
        or input_shapes != [
            [1, frames, channels], [1], [1, frames], [1, frames], [1],
        ]
        or output_shapes != [[1, 1, frames * sample_rate // 100]]
        or manifest.get("feature_dtype") not in ("float32", "float16")
    ):
        raise ValueError("RVC RKNN manifest has invalid fixed-shape tensor metadata")
    return manifest


class _RknnRuntime:
    def __init__(self, model_file: Path) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError as error:
            raise RuntimeError(
                "RVC RKNN inference requires rknn-toolkit-lite2 on RK3576"
            ) from error
        self.runtime = RKNNLite()
        loaded = self.runtime.load_rknn(str(model_file))
        if loaded != 0:
            self.runtime.release()
            raise RuntimeError(f"RKNN failed to load the RVC model (status {loaded})")
        initialized = self.runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if initialized != 0:
            self.runtime.release()
            raise RuntimeError(f"RKNPU runtime initialization failed (status {initialized})")
        self._closed = False

    def run(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        outputs = self.runtime.inference(inputs=inputs)
        if outputs is None:
            raise RuntimeError("RKNPU runtime returned no RVC audio output")
        return [np.asarray(value).copy() for value in outputs]

    def close(self) -> None:
        if not self._closed:
            self.runtime.release()
            self._closed = True


class RvcRknnSynthesizer:
    """Run fixed-frame RVC synthesis on RK3576; feature/F0 extraction stays external."""

    def __init__(self, runtime: RknnRuntime, manifest: dict[str, object]) -> None:
        if not callable(getattr(runtime, "run", None)) or not callable(getattr(runtime, "close", None)):
            raise TypeError("RVC RKNN runtime must provide run and close methods")
        frames = manifest.get("frames")
        channels = manifest.get("feature_channels")
        sample_rate = manifest.get("model_sample_rate")
        feature_dtype = manifest.get("feature_dtype")
        if (
            type(frames) is not int or frames < 1
            or channels not in (256, 768)
            or type(sample_rate) is not int or sample_rate not in _MODEL_SAMPLE_RATES
            or feature_dtype not in ("float32", "float16")
            or manifest.get("inputs") != list(RVC_SYNTHESIS_INPUTS)
            or manifest.get("outputs") != [RVC_SYNTHESIS_OUTPUT]
        ):
            raise ValueError("Invalid RVC RKNN synthesis metadata")
        self._runtime = runtime
        self.frames = frames
        self.channels = channels
        self.model_sample_rate = sample_rate
        self._feature_dtype = np.dtype(feature_dtype)
        self._audio_samples = frames * sample_rate // 100
        self._closed = False

    @classmethod
    def load(cls, model_file: Path) -> RvcRknnSynthesizer:
        model_file = Path(model_file).expanduser()
        if (
            model_file.is_symlink() or not model_file.is_file()
            or model_file.suffix.lower() != ".rknn"
        ):
            raise ValueError("Select an existing regular RVC .rknn model file")
        manifest = _read_manifest(model_file)
        runtime = _RknnRuntime(model_file)
        try:
            return cls(runtime, manifest)
        except Exception:
            runtime.close()
            raise

    def synthesize(
        self,
        features: np.ndarray,
        pitch: np.ndarray,
        pitch_hz: np.ndarray,
        *,
        speaker_id: int = 0,
    ) -> np.ndarray:
        if self._closed:
            raise RuntimeError("RVC RKNN synthesizer is closed")
        if (
            not isinstance(features, np.ndarray)
            or features.shape != (self.frames, self.channels)
            or features.dtype not in (np.dtype("float32"), np.dtype("float16"))
            or not np.isfinite(features).all()
        ):
            raise ValueError(
                f"RVC features must be finite [{self.frames}, {self.channels}] float values"
            )
        if (
            not isinstance(pitch, np.ndarray) or pitch.shape != (self.frames,)
            or not np.issubdtype(pitch.dtype, np.integer)
            or np.any(pitch < 1) or np.any(pitch > 255)
        ):
            raise ValueError(f"RVC coarse pitch must be an integer [{self.frames}] array in [1, 255]")
        if (
            not isinstance(pitch_hz, np.ndarray)
            or pitch_hz.shape != (self.frames,)
            or pitch_hz.dtype not in (np.dtype("float32"), np.dtype("float16"))
            or not np.isfinite(pitch_hz).all() or np.any(pitch_hz < 0)
        ):
            raise ValueError(f"RVC Hz pitch must be finite nonnegative [{self.frames}] float values")
        if (
            isinstance(speaker_id, bool) or not isinstance(speaker_id, int)
            or not 0 <= speaker_id <= np.iinfo(np.int64).max
        ):
            raise ValueError("RVC speaker ID must be a nonnegative int64 integer")
        feature_limit = np.finfo(self._feature_dtype).max
        if np.any(features > feature_limit) or np.any(features < -feature_limit):
            raise ValueError("RVC features exceed the RKNN model floating-point range")

        inputs = [
            np.ascontiguousarray(features[None], dtype=self._feature_dtype),
            np.array([self.frames], dtype=np.int64),
            np.ascontiguousarray(pitch[None], dtype=np.int64),
            np.ascontiguousarray(pitch_hz[None], dtype=np.float32),
            np.array([speaker_id], dtype=np.int64),
        ]
        result = self._runtime.run(inputs)
        if (
            not isinstance(result, (list, tuple))
            or len(result) != 1
            or not isinstance(result[0], np.ndarray)
        ):
            raise RuntimeError("RVC RKNN must return exactly one audio tensor")
        audio = result[0]
        if audio.shape == (1, 1, self._audio_samples):
            audio = audio[0, 0]
        elif audio.shape != (self._audio_samples,):
            raise RuntimeError("RVC RKNN returned an unexpected audio shape")
        if audio.dtype != np.float32 or not np.isfinite(audio).all():
            raise RuntimeError("RVC RKNN returned non-finite or non-float32 audio")
        return np.clip(audio.astype(np.float32, copy=True), -1.0, 1.0)

    def close(self) -> None:
        if not self._closed:
            self._runtime.close()
            self._closed = True
