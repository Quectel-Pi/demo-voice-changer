"""CPU adapter for the w-okada RVC ONNX synthesis interface, not a VC backend.

The caller supplies aligned content features and F0; capture, feature/F0
extraction, resampling and streaming alignment belong to the full pipeline.
Only explicitly selected, trusted models may be loaded. Importing this module
does not import ONNX Runtime or load any model.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class TensorMetadata(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def type(self) -> str: ...

    @property
    def shape(self) -> Sequence[int | str | None]: ...


class SynthesisSession(Protocol):
    def get_inputs(self) -> Sequence[TensorMetadata]: ...

    def get_outputs(self) -> Sequence[TensorMetadata]: ...

    def get_providers(self) -> list[str]: ...

    def run(self, output_names: list[str],
            input_feed: dict[str, np.ndarray]) -> list[np.ndarray]: ...


def load_cpu_session(model_file: Path, threads: int) -> SynthesisSession:
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("RVC ONNX thread count must be a positive integer")
    if model_file.is_symlink() or not model_file.is_file() or model_file.suffix.lower() != ".onnx":
        raise ValueError("Select an existing regular .onnx model file, not a symbolic link")
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        str(model_file), sess_options=options, providers=["CPUExecutionProvider"],
    )


class RvcOnnxSynthesizer:
    """Batch-one, F0-conditioned RVC v1/v2 synthesis from aligned frames."""

    def __init__(self, session: SynthesisSession) -> None:
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("RVC synthesis requires an explicit CPU-only ONNX session")
        inputs = list(session.get_inputs())
        self._inputs = {item.name: item for item in inputs}
        expected = {"feats", "p_len", "pitch", "pitchf", "sid"}
        if len(inputs) != len(expected) or set(self._inputs) != expected:
            raise ValueError("Unsupported RVC ONNX inputs; expected feats, p_len, pitch, pitchf, sid")
        self._feature_dtype = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
        }.get(self._inputs["feats"].type)
        if self._feature_dtype is None:
            raise ValueError("RVC ONNX features must be float32 or float16")
        for name, kind, rank in (
            ("feats", self._inputs["feats"].type, 3),
            ("p_len", "tensor(int64)", 1),
            ("pitch", "tensor(int64)", 2),
            ("pitchf", "tensor(float)", 2),
            ("sid", "tensor(int64)", 1),
        ):
            item = self._inputs[name]
            if item.type != kind or len(item.shape) != rank:
                raise ValueError(f"Unsupported RVC ONNX type or rank for {name}")
            self._check_dimension(item.shape[0], 1, name)
        channels = self._inputs["feats"].shape[-1]
        if isinstance(channels, int) and channels not in (256, 768):
            raise ValueError("RVC ONNX features must have 256 or 768 channels")
        outputs = list(session.get_outputs())
        if len(outputs) != 1 or outputs[0].name != "audio":
            raise ValueError("Unsupported RVC ONNX output; expected audio only")
        audio = outputs[0]
        if audio.type not in ("tensor(float)", "tensor(float16)") or len(audio.shape) not in (1, 3):
            raise ValueError("RVC ONNX audio must be floating point with shape [N] or [1, 1, N]")
        if len(audio.shape) == 3:
            for dimension in audio.shape[:2]:
                self._check_dimension(dimension, 1, "audio")
        self._output = audio
        self._session = session

    @staticmethod
    def _check_dimension(dimension: int | str | None, actual: int, name: str) -> None:
        if isinstance(dimension, int) and dimension != actual:
            raise ValueError(f"RVC ONNX {name} requires dimension {dimension}, got {actual}")

    @classmethod
    def load(cls, model_file: Path, *, threads: int = 2) -> RvcOnnxSynthesizer:
        """Load a trusted ONNX file explicitly; never download or register it."""
        return cls(load_cpu_session(model_file, threads))

    def synthesize(self, features: np.ndarray, pitch: np.ndarray,
                   pitch_hz: np.ndarray, *, speaker_id: int = 0) -> np.ndarray:
        """Return owned, finite, mono float32 audio clipped to [-1, 1].

        Features are [T, 256 or 768], coarse pitch and Hz F0 are [T].
        The caller must validate the speaker against the selected model.
        """
        if not isinstance(features, np.ndarray) or features.ndim != 2:
            raise ValueError("RVC features must be a two-dimensional array")
        frames, channels = features.shape
        if frames == 0 or channels not in (256, 768):
            raise ValueError("RVC features need nonempty frames with 256 or 768 channels")
        if features.dtype not in (np.dtype("float32"), np.dtype("float16")):
            raise ValueError("RVC features must use float32 or float16")
        if not np.isfinite(features).all():
            raise ValueError("RVC features contain non-finite values")
        if (not isinstance(pitch, np.ndarray) or pitch.shape != (frames,)
                or not np.issubdtype(pitch.dtype, np.integer)):
            raise ValueError("RVC coarse pitch must be an integer array aligned with features")
        if np.any(pitch < 1) or np.any(pitch > 255):
            raise ValueError("RVC coarse pitch must be within [1, 255]")
        if (not isinstance(pitch_hz, np.ndarray) or pitch_hz.shape != (frames,)
                or pitch_hz.dtype not in (np.dtype("float32"), np.dtype("float16"))
                or not np.isfinite(pitch_hz).all() or np.any(pitch_hz < 0)):
            raise ValueError("RVC Hz pitch must be finite, nonnegative and aligned with features")
        if (isinstance(speaker_id, bool) or not isinstance(speaker_id, int)
                or not 0 <= speaker_id <= np.iinfo(np.int64).max):
            raise ValueError("RVC speaker ID must be a nonnegative int64 integer")
        feature_limit = np.finfo(self._feature_dtype).max
        if np.any(features > feature_limit) or np.any(features < -feature_limit):
            raise ValueError("RVC features exceed the model's floating-point range")
        feed = {
            "feats": np.ascontiguousarray(features[None], dtype=self._feature_dtype),
            "p_len": np.array([frames], dtype=np.int64),
            "pitch": np.ascontiguousarray(pitch[None], dtype=np.int64),
            "pitchf": np.ascontiguousarray(pitch_hz[None], dtype=np.float32),
            "sid": np.array([speaker_id], dtype=np.int64),
        }
        for name, value in feed.items():
            for dimension, actual in zip(self._inputs[name].shape, value.shape):
                self._check_dimension(dimension, actual, name)
        result = self._session.run(["audio"], feed)
        if len(result) != 1 or not isinstance(result[0], np.ndarray):
            raise RuntimeError("RVC ONNX returned no single audio array")
        audio = result[0]
        if audio.ndim != len(self._output.shape):
            raise RuntimeError("RVC ONNX returned an unexpected audio rank")
        if audio.ndim == 3:
            if audio.shape[:2] != (1, 1):
                raise RuntimeError("RVC ONNX returned non-mono audio")
            audio = audio[0, 0]
        expected_dtype = np.float32 if self._output.type == "tensor(float)" else np.float16
        if audio.dtype != expected_dtype or audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("RVC ONNX returned empty, non-finite or incorrectly typed audio")
        declared_frames = self._output.shape[-1]
        if isinstance(declared_frames, int) and len(audio) != declared_frames:
            raise RuntimeError("RVC ONNX audio length does not match the model output")
        return np.clip(audio.astype(np.float32, copy=True), -1.0, 1.0)
