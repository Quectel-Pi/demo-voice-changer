"""Experimental helper-process client for the Beatrice v2 engine.

The in-process Beatrice path measured about 24.7 ms per 21.3 ms block on the
board once Qt, painting and analysis competed for the interpreter.  This module
runs the same stateful conversion in ``scripts/beatrice_worker.py`` (a separate
OS process) and hands 48 kHz blocks over a pipe, so the GUI process does no
Beatrice math at all.  It is a measurement/integration helper, not a passthrough
fallback: a dead or slow worker raises, and callers must surface the fault.

The helper is started with the current interpreter unless
``VOICE_CHANGER_BEATRICE_PYTHON`` is set; the Beatrice API and models are still
configured by ``VOICE_CHANGER_BEATRICE_API_DIR`` / ``..._MODEL_DIR``.
"""

from __future__ import annotations

import os
import queue
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import beatrice

_MAGIC = b"BEA1"
_HEADER = struct.Struct("<4sI")
_MAX_MESSAGE = 1 << 20
#: Board measurement with the helper process and an RvcWorkerProcessor-like
#: queue (4-block reserve, 16-block capacity): 30 s simulated chain, no underrun.
MEASURED_BLOCK_MS = 19.6
MEASURED_SECONDS = 30.0


def worker_script() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "beatrice_worker.py"


def worker_python() -> str:
    configured = os.environ.get("VOICE_CHANGER_BEATRICE_PYTHON")
    return configured or sys.executable


def _read_exact(stream, count: int, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    data = bytearray()
    while len(data) < count:
        piece = stream.read(count - len(data))
        if not piece:
            raise RuntimeError("Beatrice 辅助进程已退出 / the Beatrice helper exited")
        data.extend(piece)
        if time.monotonic() > deadline:
            raise TimeoutError("Beatrice 辅助进程响应超时 / the Beatrice helper timed out")
    return bytes(data)


class BeatriceWorkerClient:
    """Persistent helper process that owns one stateful Beatrice converter."""

    def __init__(
        self,
        config: "StreamConfig | CaptureConfig",
        engine_config: beatrice.BeatriceConfig,
        *,
        command: list[str] | None = None,
        startup_timeout: float = 180.0,
        inference_timeout: float = 10.0,
    ) -> None:
        reason = beatrice.unavailable_reason(engine_config)
        if reason is not None:
            raise RuntimeError(reason)
        self.engine_config = engine_config
        self.blocksize = config.blocksize
        self._command = command or [
            worker_python(), str(worker_script()),
            "--api-dir", str(engine_config.api_dir),
            "--model-dir", str(engine_config.model_dir),
            "--speaker", str(beatrice.clamp_speaker(engine_config.speaker)),
            "--pitch-shift", f"{beatrice.clamp_pitch_shift(engine_config.pitch_shift_semitone):g}",
            "--formant-shift", f"{beatrice.clamp_formant_shift(engine_config.formant_shift):g}",
            "--blocksize", str(self.blocksize),
        ]
        self._startup_timeout = startup_timeout
        self._inference_timeout = inference_timeout
        self._process: subprocess.Popen | None = None
        self._start()

    def _start(self) -> None:
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0,
        )
        try:
            magic, length = _HEADER.unpack(_read_exact(
                self._process.stdout, _HEADER.size, self._startup_timeout))
            if magic != _MAGIC or length != 0:
                message = _read_exact(
                    self._process.stdout, min(length, _MAX_MESSAGE), 5.0).decode(
                        "utf-8", "replace")
                raise RuntimeError(f"Beatrice 辅助进程启动失败 / helper failed: {message}")
        except Exception:
            self.close()
            raise

    def convert(self, samples: np.ndarray) -> np.ndarray:
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Beatrice 辅助进程未运行 / the Beatrice helper is not running")
        block = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        if block.size != self.blocksize:
            raise ValueError(f"Beatrice worker expects {self.blocksize} frames")
        if not np.isfinite(block).all():
            raise ValueError("Beatrice worker input contains non-finite samples")
        try:
            self._process.stdin.write(_HEADER.pack(_MAGIC, block.size))
            self._process.stdin.write(block.tobytes())
            self._process.stdin.flush()
            magic, length = _HEADER.unpack(_read_exact(
                self._process.stdout, _HEADER.size, self._inference_timeout))
            if magic != _MAGIC or length != self.blocksize:
                message = _read_exact(
                    self._process.stdout, min(length, _MAX_MESSAGE), 5.0).decode(
                        "utf-8", "replace")
                raise RuntimeError(f"Beatrice 转换失败 / conversion failed: {message}")
            payload = _read_exact(self._process.stdout, self.blocksize * 4, self._inference_timeout)
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(f"与 Beatrice 辅助进程通信失败 / pipe failure: {error}") from error
        output = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
        if not np.isfinite(output).all():
            raise RuntimeError("Beatrice 返回了非有限样本 / non-finite samples")
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
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
        finally:
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except OSError:
                pass

    def __enter__(self) -> "BeatriceWorkerClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class _ClientConverter:
    """Adapter that makes the helper process look like a block converter.

    The optional external formant stage runs here, in the GUI process, while the
    helper process keeps running the model. That keeps the two CPU-heavy steps on
    separate cores: on the board the model alone measures about 19.6 ms per 21.3
    ms block, so adding the ~3.5 ms Rubber Band stage to the same call would make
    the live path late. With the stage enabled the model and the formant stage
    run on their own threads and are pipelined, so the steady-state wall time of
    one ``__call__`` is ``max(model, stage)`` instead of the sum. The caller
    hands one block per call and cannot submit ahead, so the converter keeps a
    two-block lookahead of its own: the first two calls return silence and every
    later call returns the block converted two calls earlier. That is two extra
    blocks of latency (~43 ms at 1024 frames), covered by the worker's startup
    reserve.
    """

    #: Blocks of lookahead that let the model and the formant stage overlap.
    PIPELINE_LAG = 2

    def __init__(self, client: BeatriceWorkerClient, config, *, stage=None) -> None:
        import numpy as np

        self._np = np
        self.client = client
        self.blocksize = client.blocksize
        self.channels = config.channels
        engine_config = getattr(client, "engine_config", None)
        formant = beatrice.clamp_external_formant(
            getattr(engine_config, "external_formant", 1.0))
        if stage is not None:
            self._formant = stage
        elif formant != 1.0:
            self._formant = beatrice.ExternalFormant(
                getattr(config, "samplerate", 48000), self.channels,
                self.blocksize, formant)
        else:
            self._formant = None
        self._closed = False
        self._failure: BaseException | None = None
        self._lag_remaining = self.PIPELINE_LAG
        self._threads: tuple[threading.Thread, ...] = ()
        if self._formant is None:
            self._inbox = None
            self._converted = None
            self._outbox = None
        else:
            self._inbox: queue.Queue = queue.Queue(maxsize=2)
            self._converted: queue.Queue = queue.Queue(maxsize=2)
            self._outbox: queue.Queue = queue.Queue(maxsize=2)
            self._threads = (
                threading.Thread(
                    target=self._model_loop,
                    name="voice-changer-beatrice-model", daemon=True),
                threading.Thread(
                    target=self._formant_loop,
                    name="voice-changer-beatrice-formant", daemon=True),
            )
            for thread in self._threads:
                thread.start()

    def __call__(self, samples: "np.ndarray") -> "np.ndarray":
        np = self._np
        block = np.asarray(samples, dtype=np.float32)
        if block.shape != (self.blocksize, self.channels):
            raise ValueError("Beatrice helper converter received an unexpected block shape")
        if self._closed:
            raise RuntimeError("Beatrice helper converter is closed")
        if self._inbox is None:
            if self.channels == 1:
                mono = block[:, 0].copy()
            else:
                mono = block.mean(axis=1, dtype=np.float32)
            np.clip(mono, -1, 1, out=mono)
            converted = self.client.convert(mono)
            if self.channels == 1:
                return converted[:, None].copy()
            return np.repeat(converted[:, None], self.channels, axis=1)
        if self._failure is not None:
            raise RuntimeError(
                f"Beatrice 共振峰后处理失败 / formant pipeline failed: {self._failure}"
            ) from self._failure
        self._inbox.put(block)
        if self._lag_remaining > 0:
            self._lag_remaining -= 1
            return np.zeros((self.blocksize, self.channels), dtype=np.float32)
        output = self._outbox.get()
        if output is None:
            error = self._failure
            raise RuntimeError(
                f"Beatrice 共振峰后处理失败 / formant pipeline failed: {error}"
            ) from error
        return output

    def _split(self, block: "np.ndarray") -> "np.ndarray":
        np = self._np
        if self.channels == 1:
            mono = block[:, 0].copy()
        else:
            mono = block.mean(axis=1, dtype=np.float32)
        np.clip(mono, -1, 1, out=mono)
        return mono

    def _merge(self, converted: "np.ndarray") -> "np.ndarray":
        np = self._np
        if self.channels == 1:
            return converted[:, None].copy()
        return np.repeat(converted[:, None], self.channels, axis=1)

    def _queue_put(self, target: "queue.Queue", block) -> bool:
        while not self._closed:
            try:
                target.put(block, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _model_loop(self) -> None:
        try:
            while True:
                block = self._inbox.get()
                if block is None:
                    break
                converted = self.client.convert(self._split(block))
                if not self._queue_put(self._converted, self._merge(converted)):
                    break
        except BaseException as error:  # noqa: BLE001 - surfaced to the audio thread
            self._failure = error
        finally:
            try:
                self._converted.put(None, timeout=1.0)
            except queue.Full:
                pass

    def _formant_loop(self) -> None:
        try:
            while True:
                block = self._converted.get()
                if block is None:
                    break
                output = self._formant.process(block)
                if not self._queue_put(self._outbox, output):
                    break
        except BaseException as error:  # noqa: BLE001 - surfaced to the audio thread
            self._failure = error
        finally:
            try:
                self._outbox.put(None, timeout=1.0)
            except queue.Full:
                pass

    def close(self) -> None:
        self._closed = True
        if self._inbox is not None:
            try:
                self._inbox.put(None, timeout=1.0)
            except queue.Full:
                pass
            self.client.close()
            for thread in self._threads:
                thread.join(timeout=2.0)
            self._threads = ()
        if self._formant is not None:
            self._formant.close()
            self._formant = None
        self.client.close()


def subprocess_processor(config, engine_config: beatrice.BeatriceConfig):
    """Build the RvcWorkerProcessor bridge backed by the helper process."""
    from .rvc_worker import RvcWorkerProcessor

    if config.samplerate != 48000 or config.blocksize not in (1024, 2496):
        raise ValueError("Beatrice helper path requires 48000 Hz and 1024/2496 frames")
    client = BeatriceWorkerClient(config, engine_config)
    converter = _ClientConverter(client, config)
    return RvcWorkerProcessor(
        config, converter, conversion_frames=client.blocksize,
        queue_blocks=16, startup_blocks=4,
    )


def measure_block_ms(
    config, engine_config: beatrice.BeatriceConfig, *, blocks: int = 10, warmup: int = 2,
) -> float:
    """Measure helper round trips per device block (preflight for the live path)."""
    import numpy as np

    client = BeatriceWorkerClient(config, engine_config)
    try:
        block = np.zeros(config.blocksize, dtype=np.float32)
        for _ in range(warmup):
            client.convert(block)
        began = time.monotonic()
        for _ in range(blocks):
            client.convert(block)
        return (time.monotonic() - began) * 1000.0 / blocks
    finally:
        client.close()
