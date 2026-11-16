"""Bounded asynchronous bridge for a block-to-block RVC audio converter."""

from __future__ import annotations

from collections.abc import Callable
import math
import threading
import time

import numpy as np

from ..audio.device_types import CaptureConfig, StreamConfig
from ..audio.live_processing import FrameRing


class RvcWorkerProcessor:
    """Keep model work out of the audio callback and surface every overrun.

    The converter runs on a dedicated worker and must return one normalized
    float32 block at the same device rate/channel layout. This bridge does not
    resample or hide model latency; callers must provide a tested converter.
    """

    def __init__(
        self,
        config: StreamConfig | CaptureConfig,
        convert: Callable[[np.ndarray], np.ndarray],
        *,
        queue_blocks: int = 8,
        startup_blocks: int = 2,
        conversion_frames: int | None = None,
    ) -> None:
        if not callable(convert):
            raise TypeError("RVC block converter must be callable")
        for name, value in (("queue_blocks", queue_blocks), ("startup_blocks", startup_blocks)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if startup_blocks >= queue_blocks:
            raise ValueError("RVC startup reserve must be smaller than the bounded queue")
        self.config = config
        self.convert = convert
        self.block_frames = config.blocksize
        self.conversion_frames = config.blocksize if conversion_frames is None else conversion_frames
        if type(self.conversion_frames) is not int or self.conversion_frames < self.block_frames:
            raise ValueError("conversion_frames must be an integer at least as large as the device block")
        self.channels = config.channels
        capacity = self.conversion_frames * queue_blocks
        self.input = FrameRing(capacity, config.channels)
        self.output = FrameRing(capacity, config.channels)
        self._block = np.empty((self.conversion_frames, config.channels), dtype=np.float32)
        self._startup_frames = self.conversion_frames * startup_blocks
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._submitted = 0
        self._processed = 0
        self._playing = False
        self._converter_closed = False
        self.error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("Create a new RVC worker for each listening session")
        self._thread = threading.Thread(
            target=self._run, name="voice-changer-rvc-inference", daemon=False,
        )
        self._thread.start()

    def exchange(self, samples: np.ndarray, output: np.ndarray) -> bool:
        if not isinstance(output, np.ndarray):
            self.error = TypeError("RVC audio callback output must be a NumPy array")
            return False
        output.fill(0)
        if self.error is not None:
            return False
        if (not isinstance(samples, np.ndarray)
                or samples.shape != (self.block_frames, self.channels)
                or output.shape != samples.shape
                or samples.dtype != np.float32 or output.dtype != np.float32
                or not np.isfinite(samples).all()
                or np.any(np.abs(samples) > 1)):
            self.error = ValueError("RVC audio callback received invalid block shape or samples")
            return False
        if not self.input.push(samples):
            self.error = RuntimeError("RVC input queue overflow; AI listening must stop")
            return False
        self._submitted += self.block_frames
        if self._thread is None or not self._thread.is_alive():
            self.error = RuntimeError("RVC worker is not running")
            return False
        if not self._playing:
            if self.output.available < self._startup_frames:
                if self._submitted - self._processed > self.input.capacity:
                    self.error = RuntimeError("RVC worker produced no timely output during startup")
                    return False
                return True
            self._playing = True
        if not self.output.pop(output):
            self.error = RuntimeError("RVC output queue underrun; AI listening must stop")
            return False
        return True

    def check_health(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"RVC worker failed: {self.error}") from self.error
        if self._thread is not None and not self._thread.is_alive() and not self._stop.is_set():
            raise RuntimeError("RVC worker exited unexpectedly")

    def close(self, *, timeout: float = 2.0) -> None:
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or timeout <= 0 or timeout > threading.TIMEOUT_MAX
                or not math.isfinite(timeout)):
            raise ValueError("RVC worker close timeout must be positive")
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise RuntimeError("RVC worker did not stop before the timeout")
            self._thread = None
        if not self._converter_closed:
            close = getattr(self.convert, "close", None)
            if close is not None:
                if not callable(close):
                    raise TypeError("RVC converter close attribute must be callable")
                close()
            self._converter_closed = True

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if self.input.available < self.conversion_frames:
                    time.sleep(0.001)
                    continue
                if not self.input.pop(self._block):
                    continue
                converted = self.convert(self._block)
                if (not isinstance(converted, np.ndarray)
                        or converted.shape != self._block.shape
                        or converted.dtype != np.float32
                        or not np.isfinite(converted).all()
                        or np.any(np.abs(converted) > 1)):
                    raise RuntimeError("RVC converter returned invalid audio; no passthrough is used")
                if not self.output.push(converted):
                    raise RuntimeError("RVC output queue overflow; AI listening must stop")
                self._processed += self.conversion_frames
        except Exception as error:
            self.error = error
