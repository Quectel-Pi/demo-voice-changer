"""RKNN (NPU) live backend for HEEVoicechanger 16 kHz causal models.

The GUI runs Python 3.13 on the board, which has no ``rknn-toolkit-lite2``
wheel. The NPU session therefore lives in a small helper process started with
a Python 3.10 interpreter (``VOICE_CHANGER_RKNN_PYTHON``), and this module owns
the audio side: resampling, the fixed window, and the frame protocol.

Nothing here downloads or converts a model. A ``.rknn`` file is only used when
its sidecar manifest matches this module's format, was built with a working
toolkit (``rknn-toolkit2`` 2.5.0 or newer; the 2.3.2 compiler miscompiles this
network on RK3576) and the source ONNX digest of a known HEEVoicechanger graph.
No passthrough or CPU fallback exists: a failing worker raises and the live
engine reports a fault.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import sys

import numpy as np

from ..audio.device_types import CaptureConfig, StreamConfig

MANIFEST_FORMAT = "voice-changer-hevc-rknn-v1"
SAMPLE_RATE = 16000
HISTORY_SAMPLES = 8192
CHUNK_SAMPLES = 3072
WINDOW_SAMPLES = HISTORY_SAMPLES + CHUNK_SAMPLES
CHUNK_MS = 192
CONVERSION_FRAMES = CHUNK_SAMPLES * 3  # 48 kHz device frames per NPU chunk
TARGET_PLATFORM = "rk3576"
# Packages must be compiled with rknn-toolkit2 >= 2.5.0. The 2.3.2 toolkit's
# compiler produces numerically wrong RK3576 code for this network (correct in
# the simulator, garbage on the NPU); 2.5.0-built packages run correctly on the
# board's librknnrt 2.3.2 runtime.
TOOLKIT_VERSION = "2.5.0"
MIN_TOOLKIT_VERSION = (2, 5, 0)
MEASURED_CHUNK_MS = 113.8
MEASURED_RTF = round(MEASURED_CHUNK_MS / CHUNK_MS, 3)
MIN_SIMULATOR_SNR_DB = 40.0
WORKER_PYTHON_ENV = "VOICE_CHANGER_RKNN_PYTHON"

LIVE_NOTE = (
    "实验模型：RKNN NPU 每 192 ms 分块处理，实测约 114 ms（RTF≈0.59），"
    "启动储备（384 ms）与重采样会增加总延迟，约 0.4–0.6 秒；不是低延迟链路，音质需你试听验收。"
    " / Experimental: RKNN NPU processes 192 ms chunks in about 114 ms (RTF≈0.59); "
    "startup reserve (384 ms) and resampling add up to roughly 0.4–0.6 s total delay. "
    "This is not a low-latency path, and quality is for you to judge."
)

_REQUEST = struct.Struct("<4sI")
_HEADER = struct.Struct("<4sI")
_MAGIC = b"HEV1"
_READY = b"HVRD"
_ERROR = b"HVER"
_MAX_MESSAGE = 1 << 20


def worker_script() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "hevc_rknn_worker.py"


def worker_python() -> str:
    value = os.environ.get(WORKER_PYTHON_ENV)
    return value.strip() if value and value.strip() else sys.executable


def manifest_path(model_path: Path) -> Path:
    return Path(f"{model_path}.json")


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _known_source_digests() -> set[str]:
    from .model_catalog import ENTRIES

    digests = set()
    for entry in ENTRIES:
        if entry.model_type == "hevc_onnx":
            digests.update(entry.sha256)
    return digests


def _toolkit_version_ok(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parts = tuple(int(part) for part in value.split(".")[:3])
    except ValueError:
        return False
    return parts >= MIN_TOOLKIT_VERSION


def read_manifest(model_path: Path) -> dict:
    """Validate the sidecar package; raises ValueError with a user-facing reason."""
    path = Path(model_path)
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"无法访问模型文件 / Cannot access model file: {path}: {error}") from error
    import stat

    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError("仅支持普通模型文件，不支持符号链接或目录 / Select a regular model file")
    try:
        manifest = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"需要与模型同名的 sidecar manifest / A sidecar manifest is required: {manifest_path(path)}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("manifest 必须是 JSON 对象 / The manifest must be a JSON object")
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"不支持的 RKNN 包格式 / Unsupported RKNN package: {manifest.get('format')!r}"
        )
    if manifest.get("target_platform") != TARGET_PLATFORM:
        raise ValueError("模型目标平台不是 RK3576 / The package targets another platform")
    if not _toolkit_version_ok(manifest.get("toolkit_version")):
        raise ValueError(
            f"需要 rknn-toolkit2 ≥ {TOOLKIT_VERSION} 生成的模型"
            "（2.3.2 编译器会把该网络编译错） / "
            f"The package must be built with rknn-toolkit2 >= {TOOLKIT_VERSION}; "
            "the 2.3.2 compiler miscompiles this network"
        )
    if manifest.get("model_sha256") != _sha256_file(path):
        raise ValueError("模型 SHA-256 与 manifest 不一致 / Model SHA-256 does not match the manifest")
    if manifest.get("frames") != WINDOW_SAMPLES:
        raise ValueError("模型窗口大小与运行时不一致 / Model window does not match this runtime")
    if manifest.get("history") != HISTORY_SAMPLES or manifest.get("chunk") != CHUNK_SAMPLES:
        raise ValueError("模型窗口划分与运行时不一致 / Model window split does not match this runtime")
    source = manifest.get("source_onnx_sha256")
    if source not in _known_source_digests():
        raise ValueError("manifest 的来源 ONNX 不是已知 HEVC 模型 / Unknown source ONNX digest")
    snr = manifest.get("simulator_snr_db")
    if not isinstance(snr, (int, float)) or not np.isfinite(snr) or snr < MIN_SIMULATOR_SNR_DB:
        raise ValueError(
            f"模拟器精度未达门限 {MIN_SIMULATOR_SNR_DB:.0f} dB / "
            f"Simulator accuracy is below {MIN_SIMULATOR_SNR_DB:.0f} dB"
        )
    return manifest


def unavailable_reason(model_path: Path | None) -> str | None:
    if model_path is None:
        return (
            "选择已转换的 HEVC RKNN 模型（含 sidecar manifest），或先按文档在主机侧转换。"
            " / Select a converted HEVC RKNN package, or convert it first as documented."
        )
    try:
        read_manifest(model_path)
    except ValueError as error:
        return str(error)
    if os.environ.get(WORKER_PYTHON_ENV, "").strip() == "":
        return (
            f"设置 {WORKER_PYTHON_ENV} 指向板端 Python 3.10 环境（含 rknn-toolkit-lite2）。"
            f" / Set {WORKER_PYTHON_ENV} to the board Python 3.10 environment with rknn-toolkit-lite2."
        )
    if not worker_script().is_file():
        return f"缺少 NPU 辅助进程脚本 / Missing NPU helper script: {worker_script()}"
    if not Path(worker_python()).is_file():
        return (
            f"{WORKER_PYTHON_ENV} 指向的解释器不存在 / The configured interpreter does not exist: "
            f"{worker_python()}"
        )
    return None


def _read_exact(stream, count: int, timeout: float) -> bytes:
    import time

    deadline = time.monotonic() + timeout
    data = bytearray()
    fd = stream.fileno()
    while len(data) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("NPU 辅助进程超时 / The NPU helper timed out")
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            continue
        piece = stream.read(count - len(data))
        if not piece:
            raise RuntimeError("NPU 辅助进程已退出 / The NPU helper exited")
        data.extend(piece)
    return bytes(data)


class HevcRknnClient:
    """Persistent helper process that runs one fixed-window RKNN model."""

    def __init__(self, model_path: Path, *, command: list[str] | None = None,
                 startup_timeout: float = 120.0, inference_timeout: float = 10.0) -> None:
        self.model_path = Path(model_path)
        read_manifest(self.model_path)
        self._command = command or [
            worker_python(), str(worker_script()),
            "--model", str(self.model_path),
            "--history", str(HISTORY_SAMPLES),
            "--chunk", str(CHUNK_SAMPLES),
        ]
        self._startup_timeout = startup_timeout
        self._inference_timeout = inference_timeout
        self._process: subprocess.Popen | None = None
        self._start()

    def _start(self) -> None:
        if not Path(self._command[0]).is_file() and not os.path.sep in self._command[0]:
            raise RuntimeError(f"找不到 Python 解释器 / Python interpreter not found: {self._command[0]}")
        self._process = subprocess.Popen(
            self._command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0,
        )
        try:
            magic, length = _HEADER.unpack(_read_exact(self._process.stdout, _HEADER.size, self._startup_timeout))
            if magic == _ERROR:
                message = _read_exact(self._process.stdout, min(length, _MAX_MESSAGE), 5.0).decode(
                    "utf-8", "replace")
                raise RuntimeError(f"NPU 辅助进程启动失败 / NPU helper failed to start: {message}")
            if magic != _READY or length != 0:
                raise RuntimeError("NPU 辅助进程握手无效 / Invalid NPU helper handshake")
        except Exception:
            self.close()
            raise

    def infer(self, window: np.ndarray) -> np.ndarray:
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("NPU 辅助进程未运行 / The NPU helper is not running")
        samples = np.ascontiguousarray(window, dtype=np.float32).reshape(-1)
        if samples.size != WINDOW_SAMPLES:
            raise ValueError(f"HEVC RKNN window must contain {WINDOW_SAMPLES} samples")
        if not np.isfinite(samples).all():
            raise ValueError("HEVC RKNN input contains non-finite samples")
        try:
            self._process.stdin.write(_REQUEST.pack(_MAGIC, samples.size))
            self._process.stdin.write(samples.tobytes())
            self._process.stdin.flush()
            magic, length = _HEADER.unpack(
                _read_exact(self._process.stdout, _HEADER.size, self._inference_timeout))
            if magic == _ERROR:
                message = _read_exact(self._process.stdout, min(length, _MAX_MESSAGE), 5.0).decode(
                    "utf-8", "replace")
                raise RuntimeError(f"NPU 推理失败 / NPU inference failed: {message}")
            if magic != _MAGIC or length != CHUNK_SAMPLES:
                raise RuntimeError("NPU 辅助进程返回了无效响应 / The NPU helper returned an invalid response")
            payload = _read_exact(self._process.stdout, CHUNK_SAMPLES * 4, self._inference_timeout)
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(f"与 NPU 辅助进程通信失败 / NPU helper pipe failure: {error}") from error
        output = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
        if not np.isfinite(output).all():
            raise RuntimeError("NPU 返回了非有限样本 / The NPU returned non-finite samples")
        return output

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        try:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass
        finally:
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except OSError:
                pass

    def __enter__(self) -> "HevcRknnClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class HevcRknnConverter:
    """Convert 48 kHz blocks with a validated 16 kHz RKNN window."""

    def __init__(self, config: StreamConfig | CaptureConfig, model_path: Path,
                 *, client: HevcRknnClient | None = None) -> None:
        from .resampling import StreamingResampler

        if config.samplerate != 48000 or config.blocksize != CONVERSION_FRAMES or config.channels not in (1, 2):
            raise ValueError("HEVC RKNN requires 48000 Hz, 9216 frames and 1 or 2 channels")
        self.config = config
        self.model_path = Path(model_path)
        self.client = client or HevcRknnClient(self.model_path)
        self.down = StreamingResampler(48000, SAMPLE_RATE)
        self.up = StreamingResampler(SAMPLE_RATE, 48000)
        self.pending = np.empty(0, dtype=np.float32)
        self.history = np.zeros(HISTORY_SAMPLES, dtype=np.float32)
        self.skip = 16
        self.ready = np.zeros(config.blocksize * 2, dtype=np.float32)

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        if (samples.shape != (self.config.blocksize, self.config.channels)
                or samples.dtype != np.float32 or not np.isfinite(samples).all()):
            raise ValueError("Invalid HEVC RKNN device audio")
        mono = samples.mean(axis=1, dtype=np.float32)
        np.clip(mono, -1, 1, out=mono)
        incoming = self.down.process(mono)
        drop = min(self.skip, incoming.size)
        self.skip -= drop
        self.pending = np.concatenate((self.pending, incoming[drop:]))
        while self.pending.size >= CHUNK_SAMPLES:
            chunk = self.pending[:CHUNK_SAMPLES]
            window = np.concatenate((self.history, chunk))
            converted = self.client.infer(window)
            self.history = window[CHUNK_SAMPLES:].copy()
            self.pending = self.pending[CHUNK_SAMPLES:].copy()
            self.ready = np.concatenate((self.ready, self.up.process(np.clip(converted, -1, 1))))
        count = self.config.blocksize
        if self.ready.size < count:
            raise RuntimeError("HEVC RKNN resampling underrun; no passthrough is used")
        output = np.repeat(self.ready[:count, None], self.config.channels, axis=1)
        self.ready = self.ready[count:].copy()
        if self.ready.size > count * 2 or self.pending.size >= CHUNK_SAMPLES:
            raise RuntimeError("HEVC RKNN streaming buffers exceeded their limit")
        return np.clip(output, -1, 1)

    def close(self) -> None:
        self.client.close()


@dataclass(frozen=True)
class HevcRknnProcessorFactory:
    model_path: Path

    def __call__(self, config: StreamConfig | CaptureConfig, _control) -> object:
        from dataclasses import replace

        if config.samplerate != 48000 or config.blocksize != 1024:
            raise ValueError("HEVC RKNN audio device requires 48000 Hz and 1024 frames")
        conversion_config = replace(config, blocksize=CONVERSION_FRAMES)
        converter = HevcRknnConverter(conversion_config, self.model_path)
        from .rvc_worker import RvcWorkerProcessor

        return RvcWorkerProcessor(
            config, converter, queue_blocks=6, startup_blocks=2,
            conversion_frames=CONVERSION_FRAMES,
        )
