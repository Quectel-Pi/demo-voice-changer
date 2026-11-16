"""Opt-in fixed-target LLVC ONNX/RKNN streaming; device and model blocks are independent."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    import numpy as np
    from ..audio.device_types import CaptureConfig, StreamConfig
    from .rvc_worker import RvcWorkerProcessor

MODEL_SHA256 = "b890acee384ad2d7c342c16404ac85bebbbe273deb0636f22d2c79f729eb74c5"
INPUTS = ("audio", "enc", "dec", "out", "pre")
OUTPUTS = ("converted", "next_enc", "next_dec", "next_out", "next_pre")
MODEL_FRAMES = 832
STABLE_MODEL_SHA256 = "427e80ee069f65f2196482d08a5a1bfd8e027fdfaacd55fada1d15fc74b9074a"
MODEL_BLOCKS = {MODEL_SHA256: MODEL_FRAMES, STABLE_MODEL_SHA256: 1664}
RKNN_MANIFEST_FORMAT = "voice-changer-llvc-rknn-v2"
RKNN_SIMULATOR_RTOL = 0.001
RKNN_SIMULATOR_ATOL = 0.0001
RKNN_SIMULATOR_BLOCKS_PER_CASE = 3
RKNN_SIMULATOR_CASE_NAMES = (
    "zero_state_silence",
    "fixed_random_audio_zero_state",
    "positive_boundary_zero_state",
    "negative_boundary_zero_state",
    "alternating_boundary_zero_state",
    "random_state_audio",
)
RKNN_INPUT_SHAPES = (
    (1, 1, 1696),
    (1, 512, 510),
    (1, 2, 13, 256),
    (1, 512, 4),
    (1, 1, 24),
)
RKNN_OUTPUT_SHAPES = (
    (1, 1, 1664),
    (1, 512, 510),
    (1, 2, 13, 256),
    (1, 512, 4),
    (1, 1, 24),
)


def _has_valid_host_simulator_equivalence(manifest: dict) -> bool:
    result = manifest.get("host_simulator_equivalence")
    if not isinstance(result, dict):
        return False
    expected_blocks = len(RKNN_SIMULATOR_CASE_NAMES) * RKNN_SIMULATOR_BLOCKS_PER_CASE
    errors = result.get("max_abs_error")
    return (
        result.get("passed") is True
        and result.get("rtol") == RKNN_SIMULATOR_RTOL
        and result.get("atol") == RKNN_SIMULATOR_ATOL
        and result.get("cases") == list(RKNN_SIMULATOR_CASE_NAMES)
        and result.get("continuous_blocks_compared") == expected_blocks
        and result.get("output_tensors_compared") == expected_blocks * len(OUTPUTS)
        and isinstance(errors, dict)
        and set(errors) == set(OUTPUTS)
        and all(
            type(value) in (int, float) and math.isfinite(value) and value >= 0
            for value in errors.values()
        )
    )


def _validate_rknn_package(model_path: Path) -> None:
    manifest_path = Path(f"{model_path}.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"LLVC RKNN manifest is missing or invalid: {manifest_path}") from error
    with model_path.open("rb") as source:
        model_digest = LlvcConverter._hash(source)
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != RKNN_MANIFEST_FORMAT
        or manifest.get("source_sha256") != STABLE_MODEL_SHA256
        or manifest.get("target_platform") != "rk3576"
        or manifest.get("quantized") is not False
        or manifest.get("model_sha256") != model_digest
        or manifest.get("inputs") != list(INPUTS)
        or manifest.get("outputs") != list(OUTPUTS)
        or manifest.get("input_shapes") != [list(shape) for shape in RKNN_INPUT_SHAPES]
        or manifest.get("output_shapes") != [list(shape) for shape in RKNN_OUTPUT_SHAPES]
        or not _has_valid_host_simulator_equivalence(manifest)
    ):
        raise ValueError(
            "LLVC RKNN artifact does not match the validated RK3576 model contract "
            "or lacks passing host-simulator equivalence"
        )


class _RknnSession:
    """Adapt RKNNLite's ordered tensor API to the ONNX session interface."""

    def __init__(self, model_path: Path) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError as error:
            raise RuntimeError(
                "LLVC RKNN requires rknn-toolkit-lite2 on the target board"
            ) from error
        self.runtime = RKNNLite()
        loaded = self.runtime.load_rknn(str(model_path))
        if loaded != 0:
            self.runtime.release()
            raise RuntimeError(f"RKNN failed to load model (status {loaded})")
        initialized = self.runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if initialized != 0:
            self.runtime.release()
            raise RuntimeError(f"RKNPU runtime initialization failed (status {initialized})")
        self.closed = False

    def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        if tuple(names) != OUTPUTS or tuple(feed) != INPUTS:
            raise ValueError("LLVC RKNN tensor ordering mismatch")
        results = self.runtime.inference(inputs=[feed[name] for name in INPUTS])
        if results is None:
            raise RuntimeError("RKNPU runtime returned no outputs")
        import numpy as np

        return [np.asarray(value).copy() for value in results]

    def close(self) -> None:
        if not self.closed:
            self.runtime.release()
            self.closed = True


def configured_model() -> Path | None:
    value = os.environ.get("VOICE_CHANGER_LLVC_MODEL")
    return Path(value).expanduser() if value else None


def _validate_local_onnx_path(model_path: Path) -> None:
    if model_path.suffix.lower() != ".onnx":
        raise ValueError("Select an LLVC ONNX model (.onnx); RVC and RKNN files are not supported here")
    try:
        info = model_path.lstat()
    except OSError as error:
        raise ValueError(f"Cannot access LLVC model: {model_path}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("LLVC model must be a regular file, not a symbolic link")


def _onnx_session(model_path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"],
    )


def _validate_onnx_schema(session) -> tuple[int, tuple[tuple[int, ...], ...],
                                             tuple[tuple[int, ...], ...]]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if tuple(item.name for item in inputs) != INPUTS or tuple(item.name for item in outputs) != OUTPUTS:
        raise ValueError("LLVC model audio/state schema mismatch")
    metadata = (*inputs, *outputs)
    if any(item.type != "tensor(float)" for item in metadata):
        raise ValueError("LLVC requires float32 inputs, outputs, and streaming states")

    def fixed_shape(item) -> tuple[int, ...]:
        if not all(type(size) is int and size > 0 for size in item.shape):
            raise ValueError("LLVC requires fixed-shape audio and streaming states")
        return tuple(item.shape)

    input_shapes = tuple(fixed_shape(item) for item in inputs)
    output_shapes = tuple(fixed_shape(item) for item in outputs)
    frames = input_shapes[0][-1] - 32 if len(input_shapes[0]) == 3 else 0
    if frames not in (MODEL_FRAMES, 1664):
        raise ValueError("LLVC model block size must be 832 or 1664 samples")
    expected_inputs = (
        (1, 1, frames + 32),
        (1, 512, 510),
        (1, 2, 13, 256),
        (1, 512, 4),
        (1, 1, 24),
    )
    expected_outputs = (
        (1, 1, frames),
        *expected_inputs[1:],
    )
    if input_shapes != expected_inputs or output_shapes != expected_outputs:
        raise ValueError("LLVC model shapes do not match the fixed audio/state contract")
    return frames, input_shapes, output_shapes


def inspect_local_onnx_model(model_path: Path) -> int:
    """Validate a user-selected LLVC graph without running inference."""
    _validate_local_onnx_path(model_path)
    session = _onnx_session(model_path)
    frames, _, _ = _validate_onnx_schema(session)
    return frames


def unavailable_reason() -> str | None:
    path = configured_model()
    if path is None:
        return "设置 VOICE_CHANGER_LLVC_MODEL 指向已验证的 LLVC ONNX 模型 / Set the LLVC model path."
    if not path.is_file():
        return f"LLVC 模型不存在 / Model not found: {path}"
    if path.suffix.lower() == ".rknn":
        if not Path(f"{path}.json").is_file():
            return f"LLVC RKNN manifest not found: {path}.json"
        if importlib.util.find_spec("rknnlite") is None:
            return "缺少 rknn-toolkit-lite2，请在 RK3576 板端安装兼容运行时 / Install rknn-toolkit-lite2 on RK3576."
        return None
    if importlib.util.find_spec("onnxruntime") is None:
        return "缺少 onnxruntime，请安装 AI 可选依赖 / Install the optional AI dependencies."
    return None


class LlvcConverter:
    """Convert model-sized 48 kHz blocks with a validated 16 kHz stateful graph."""

    def __init__(self, config: StreamConfig | CaptureConfig, model_path: Path,
                 *, allow_custom_model: bool = False) -> None:
        import numpy as np
        from .resampling import StreamingResampler

        if config.samplerate != 48000 or config.blocksize not in (2496, 4992) or config.channels not in (1, 2):
            raise ValueError("LLVC requires 48000 Hz, 2496 frames and 1 or 2 channels")
        self.is_rknn = model_path.suffix.lower() == ".rknn"
        if allow_custom_model and not self.is_rknn:
            _validate_local_onnx_path(model_path)
            digest = None
        else:
            with model_path.open("rb") as source:
                digest = self._hash(source)
        if self.is_rknn:
            _validate_rknn_package(model_path)
            self.model_frames = MODEL_BLOCKS[STABLE_MODEL_SHA256]
            input_shapes = RKNN_INPUT_SHAPES
            self.output_shapes = list(RKNN_OUTPUT_SHAPES)
            self.session = _RknnSession(model_path)
        else:
            if digest not in MODEL_BLOCKS and not allow_custom_model:
                raise ValueError("LLVC model SHA-256 mismatch; use a validated 52/104 ms graph")
            self.session = _onnx_session(model_path)
            self.model_frames, input_shapes, output_shapes = _validate_onnx_schema(self.session)
            if digest in MODEL_BLOCKS and self.model_frames != MODEL_BLOCKS[digest]:
                raise ValueError("LLVC model block size does not match its validated SHA-256")
            self.output_shapes = list(output_shapes)
        config = replace(config, blocksize=self.model_frames * 3)
        self.state = [np.zeros(shape, dtype=np.float32) for shape in input_shapes[1:]]
        self.config = config
        self.down = StreamingResampler(48000, 16000)
        self.up = StreamingResampler(16000, 48000)
        self.pending = np.empty(0, dtype=np.float32)
        self.context = np.zeros(32, dtype=np.float32)
        self.skip = 16
        # Reserve two blocks for lookahead and both resamplers' FIR context.
        self.ready = np.zeros(config.blocksize * 2, dtype=np.float32)

    @staticmethod
    def _hash(source: BinaryIO) -> str:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        import numpy as np

        if (samples.shape != (self.config.blocksize, self.config.channels)
                or samples.dtype != np.float32 or not np.isfinite(samples).all()):
            raise ValueError("Invalid LLVC device audio")
        mono = samples.mean(axis=1, dtype=np.float32)
        np.clip(mono, -1, 1, out=mono)
        incoming = self.down.process(mono)
        drop = min(self.skip, incoming.size)
        self.skip -= drop
        self.pending = np.concatenate((self.pending, incoming[drop:]))
        while self.pending.size >= self.model_frames:
            block = self.pending[:self.model_frames]
            audio = np.concatenate((self.context, block))[None, None]
            feed = dict(zip(INPUTS, [audio, *self.state]))
            result = self.session.run(list(OUTPUTS), feed)
            if len(result) != len(OUTPUTS):
                raise RuntimeError("LLVC omitted audio or streaming state")
            for expected, value in zip(self.output_shapes, result):
                if (value.shape != expected or value.dtype != np.float32
                        or not np.isfinite(value).all()):
                    raise RuntimeError("LLVC returned invalid audio or streaming state")
            self.state = result[1:]
            self.context = block[-32:].copy()
            self.pending = self.pending[self.model_frames:].copy()
            converted = np.clip(result[0].reshape(-1), -1, 1)
            self.ready = np.concatenate((self.ready, self.up.process(converted)))
        count = self.config.blocksize
        if self.ready.size < count:
            raise RuntimeError("LLVC resampling output underrun; no passthrough is used")
        output = np.repeat(self.ready[:count, None], self.config.channels, axis=1)
        self.ready = self.ready[count:].copy()
        if self.pending.size >= self.model_frames or self.ready.size > count * 2:
            raise RuntimeError("LLVC streaming buffers exceeded their limit")
        return np.clip(output, -1, 1)

    def close(self) -> None:
        close = getattr(self.session, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class LlvcProcessorFactory:
    model_path: Path
    allow_custom_model: bool = False

    def __call__(self, config: StreamConfig | CaptureConfig, _control) -> RvcWorkerProcessor:
        from .rvc_worker import RvcWorkerProcessor

        if config.samplerate != 48000 or config.blocksize not in (1024, 2496):
            raise ValueError("LLVC audio device requires 48000 Hz and 1024 frames")
        conversion_config = replace(config, blocksize=2496)
        converter = LlvcConverter(
            conversion_config, self.model_path, allow_custom_model=self.allow_custom_model,
        )
        return RvcWorkerProcessor(
            config, converter, conversion_frames=converter.config.blocksize,
        )
