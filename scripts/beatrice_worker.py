#!/usr/bin/env python3
"""Board-side helper for the experimental Beatrice v2 conversion engine.

The GUI keeps Qt, painting, analysis and the audio callback in one Python
process; the in-process Beatrice conversion measured about 24.7 ms per 21.3 ms
device block on the Pi M2, which underruns after about a second.  This helper
owns the same stateful conversion graph in its own process so the GUI process
can hand blocks over a pipe and keep its GIL free.  It is an experiment: the
hardware still has to prove that the extra pipe round trip fits the budget, and
a failing helper reports an error frame and exits instead of substituting
passthrough audio.

The protocol matches ``scripts/hevc_rknn_worker.py``:

    request : b"BEA1" | u32 sample_count | sample_count * float32 (48 kHz mono)
    ready   : b"BEA1" | u32 0                          (after init, before loop)
    reply   : b"BEA1" | u32 sample_count | sample_count * float32
    error   : b"BEA1" | u32 byte_length | utf-8 message (then exit)

The helper keeps the converter state between requests (that is the whole point:
the engine is stateful), so every request must carry exactly ``blocksize``
frames and is answered with exactly ``blocksize`` frames.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"BEA1"
HEADER = struct.Struct("<4sI")
MAX_MESSAGE = 1 << 20
VERSION = "2.0.0-rc.0"


def _source_root() -> Path:
    """Locate the repository ``src`` directory from this script."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "src" / "voice_changer"
        if candidate.is_dir():
            return parent / "src"
    raise RuntimeError("cannot locate the voice_changer source tree")


def write_frame(stream, magic: bytes, payload: bytes, *, count: int | None = None) -> None:
    header_count = len(payload) if count is None else count
    stream.write(HEADER.pack(magic, header_count))
    stream.write(payload)
    stream.flush()


def fail(message: str) -> None:
    try:
        write_frame(sys.stdout.buffer, MAGIC, message.encode("utf-8", "replace")[:MAX_MESSAGE])
    except OSError:
        pass
    sys.exit(2)


def _read_exact(stream, count: int) -> bytes | None:
    data = bytearray()
    while len(data) < count:
        piece = stream.read(count - len(data))
        if not piece:
            return None
        data.extend(piece)
    return bytes(data)


def build_converter(
    api_dir: Path,
    model_dir: Path,
    speaker: int,
    blocksize: int,
    pitch_shift_semitone: float = 0.0,
    formant_shift: float = 0.0,
):
    """Create the stateful 48 kHz converter used by the worker loop."""
    sys.path.insert(0, str(_source_root()))
    from voice_changer.ai import beatrice as beatrice_module
    from voice_changer.ai.resampling import StreamingResampler

    engine_config = beatrice_module.BeatriceConfig(
        api_dir=api_dir, model_dir=model_dir, speaker=speaker)
    reason = beatrice_module.unavailable_reason(engine_config)
    if reason is not None:
        raise RuntimeError(reason)
    runtime = beatrice_module._load_runtime(engine_config.api_dir)
    try:
        engine = runtime.SimpleBeatrice(
            str(model_dir / "phone_extractor.bin"),
            str(model_dir / "pitch_estimator.bin"),
            str(model_dir / "embedding_setter.bin"),
            str(model_dir / "waveform_generator.bin"),
            str(model_dir / "speaker_embeddings.bin"),
        )
    except Exception as error:  # pragma: no cover - board only
        raise RuntimeError(f"cannot initialize the Beatrice model: {error}") from error
    if not engine.is_ready():
        raise RuntimeError("the Beatrice model is not ready")
    engine.set_target_speaker(beatrice_module.clamp_speaker(engine_config.speaker))
    pitch_shift = beatrice_module.clamp_pitch_shift(engine_config.pitch_shift_semitone)
    if pitch_shift:
        engine.set_pitch_shift_semitone(pitch_shift)
    formant_shift = beatrice_module.clamp_formant_shift(engine_config.formant_shift)
    if formant_shift:
        engine.set_formant_shift(formant_shift)

    from collections import deque

    down = StreamingResampler(48000, 16000)
    up = StreamingResampler(24000, 48000)
    pending: deque[np.ndarray] = deque()
    pending_size = 0
    ready: deque[np.ndarray] = deque()
    ready_size = 0
    ready.append(np.zeros(blocksize * 2, dtype=np.float32))
    ready_size = blocksize * 2

    def convert(samples: np.ndarray) -> np.ndarray:
        nonlocal pending_size, ready_size
        block = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        pending.append(down.process(block))
        pending_size += pending[-1].size
        while pending_size >= 160:
            hop = np.empty(160, dtype=np.float32)
            copied = 0
            while copied < 160 and pending:
                head = pending[0]
                needed = 160 - copied
                if head.size <= needed:
                    hop[copied:copied + head.size] = head
                    copied += head.size
                    pending.popleft()
                else:
                    hop[copied:copied + needed] = head[:needed]
                    pending[0] = head[needed:]
                    copied += needed
            pending_size -= 160
            converted = engine.convert(hop)
            if isinstance(converted, tuple):
                converted = converted[0]
            output = np.asarray(converted, dtype=np.float32).reshape(-1)
            if output.size != 240 or not np.isfinite(output).all():
                raise RuntimeError(f"unexpected engine output: {output.size}")
            error_fn = getattr(engine, "last_error", None)
            if callable(error_fn) and error_fn() != 0:
                raise RuntimeError(f"Beatrice engine error {error_fn()}")
            output = np.clip(output, -1, 1)
            ready.append(up.process(output))
            ready_size += ready[-1].size
        result = np.empty(blocksize, dtype=np.float32)
        copied = 0
        while copied < blocksize and ready and ready_size >= blocksize:
            head = ready[0]
            needed = blocksize - copied
            if head.size <= needed:
                result[copied:copied + head.size] = head
                copied += head.size
                ready.popleft()
            else:
                result[copied:copied + needed] = head[:needed]
                ready[0] = head[needed:]
                copied += needed
            ready_size -= head.size if head.size <= needed else needed
        if copied < blocksize:
            raise RuntimeError("Beatrice output underrun; no passthrough is used")
        return result

    return convert


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-dir", required=True, help="extracted Beatrice API directory")
    parser.add_argument("--model-dir", required=True, help="directory with the five .bin files")
    parser.add_argument("--speaker", type=int, default=0, help="JVS speaker index (0-based)")
    parser.add_argument("--pitch-shift", type=float, default=0.0, help="pitch shift in semitones")
    parser.add_argument("--formant-shift", type=float, default=0.0, help="formant shift in semitones")
    parser.add_argument("--blocksize", type=int, default=1024, choices=(1024, 2496))
    args = parser.parse_args(argv)
    blocksize = args.blocksize

    try:
        convert = build_converter(
            Path(args.api_dir).expanduser(), Path(args.model_dir).expanduser(),
            args.speaker, blocksize, args.pitch_shift, args.formant_shift)
    except Exception as error:  # pragma: no cover - board only
        fail(f"Beatrice 初始化失败 / Beatrice initialization failed: {error}")

    stdin = sys.stdin.buffer
    write_frame(sys.stdout.buffer, MAGIC, b"")
    try:
        while True:
            header = _read_exact(stdin, HEADER.size)
            if header is None:
                break
            magic, count = HEADER.unpack(header)
            if magic != MAGIC or count != blocksize:
                fail("无效的请求帧 / invalid request frame")
            payload = _read_exact(stdin, count * 4)
            if payload is None:
                break
            samples = np.frombuffer(payload, dtype="<f4").astype(np.float32)
            if samples.size != blocksize or not np.isfinite(samples).all():
                fail("输入包含非有限样本 / input contains non-finite samples")
            try:
                output = convert(samples)
            except Exception as error:  # pragma: no cover - board only
                fail(f"Beatrice 转换失败 / Beatrice conversion failed: {error}")
            write_frame(sys.stdout.buffer, MAGIC, output.tobytes(), count=output.size)
    except (BrokenPipeError, ValueError):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
