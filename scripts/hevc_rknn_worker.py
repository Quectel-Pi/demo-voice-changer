#!/usr/bin/env python3
"""Board-side RKNN helper for the HEEVoicechanger NPU live path.

The GUI runs Python 3.13, which has no ``rknn-toolkit-lite2`` wheel on the Pi
M2 image, so the NPU session lives here in a small Python 3.10 helper process
(started by ``voice_changer.ai.hevc_rknn``). The helper owns one fixed-window
RKNN model and speaks a tiny binary protocol over stdin/stdout:

    request : b"HEV1" | u32 sample_count | sample_count * float32
    ready   : b"HVRD" | u32 0                         (after init, before loop)
    reply   : b"HEV1" | u32 chunk | chunk * float32   (last ``chunk`` samples)
    error   : b"HVER" | u32 byte_length | utf-8 message (then exit)

The request carries the full fixed window (8192-sample history plus the newest
3072-sample chunk, 11264 samples at 16 kHz); the helper is stateless and returns
the model output for the newest 3072 samples. ``sample_count``/``chunk`` are
sample counts in the header, while the error frame header carries a byte length.
No passthrough, no CPU inference: a failing session reports an error frame and
exits, and the GUI shows a fault.
"""

from __future__ import annotations

import argparse
import struct
import sys

import numpy as np

MAGIC = b"HEV1"
READY = b"HVRD"
ERROR = b"HVER"
HEADER = struct.Struct("<4sI")
MAX_MESSAGE = 1 << 20


def write_frame(stream, magic: bytes, payload: bytes, *, count: int | None = None) -> None:
    header_count = len(payload) if count is None else count
    stream.write(HEADER.pack(magic, header_count))
    stream.write(payload)
    stream.flush()


def fail(message: str) -> None:
    try:
        write_frame(sys.stdout.buffer, ERROR, message.encode("utf-8", "replace")[:MAX_MESSAGE])
    except OSError:
        pass
    sys.exit(2)


def _read_exact(stream, count: int) -> bytes:
    data = bytearray()
    while len(data) < count:
        piece = stream.read(count - len(data))
        if not piece:
            raise ValueError("truncated frame")
        data.extend(piece)
    return bytes(data)


def read_request(stream, frames: int) -> bytes | None:
    """Read one request; the header count is a number of float32 samples."""
    header = stream.read(HEADER.size)
    if not header:
        return None
    if len(header) < HEADER.size:
        raise ValueError("truncated header")
    magic, count = HEADER.unpack(header)
    if magic != MAGIC:
        raise ValueError("invalid request magic")
    if count != frames or count * 4 > MAX_MESSAGE:
        raise ValueError("invalid request size")
    return _read_exact(stream, count * 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="path to the converted .rknn model")
    parser.add_argument("--history", type=int, default=8192)
    parser.add_argument("--chunk", type=int, default=3072)
    args = parser.parse_args(argv)
    frames = args.history + args.chunk
    if args.history <= 0 or args.chunk <= 0:
        fail("无效的窗口参数 / invalid window arguments")

    try:
        from rknnlite.api import RKNNLite
    except Exception as error:  # pragma: no cover - board only
        fail(f"缺少 rknn-toolkit-lite2 / rknn-toolkit-lite2 is missing: {error}")

    rknn = RKNNLite(verbose=False)
    try:
        if rknn.load_rknn(args.model) != 0:
            fail(f"无法加载 RKNN 模型 / Cannot load the RKNN model: {args.model}")
        if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            fail("NPU 会话初始化失败 / NPU session initialization failed")
    except SystemExit:
        try:
            rknn.release()
        except Exception:
            pass
        raise
    except Exception as error:
        try:
            rknn.release()
        except Exception:
            pass
        fail(f"NPU 会话初始化失败 / NPU session initialization failed: {error}")

    stdin = sys.stdin.buffer
    write_frame(sys.stdout.buffer, READY, b"")
    try:
        while True:
            payload = read_request(stdin, frames)
            if payload is None:
                break
            window = np.frombuffer(payload, dtype="<f4").astype(np.float32)
            if window.size != frames or not np.isfinite(window).all():
                fail("输入包含非有限样本 / input contains non-finite samples")
            try:
                output = rknn.inference(inputs=[window.reshape(1, 1, frames)], data_format="nchw")[0]
            except Exception as error:  # pragma: no cover - board only
                fail(f"NPU 推理失败 / NPU inference failed: {error}")
            output = np.asarray(output, dtype=np.float32).reshape(-1)
            if output.size != frames:
                fail(f"NPU 输出长度异常 / unexpected NPU output size: {output.size}")
            result = output[-args.chunk:]
            if result.size != args.chunk or not np.isfinite(result).all():
                fail("NPU 输出包含非有限样本 / NPU output contains non-finite samples")
            write_frame(sys.stdout.buffer, MAGIC, result.tobytes(), count=result.size)
    except (BrokenPipeError, ValueError):
        pass
    finally:
        try:
            rknn.release()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
