#!/usr/bin/env python3
"""Paced benchmark for the experimental Beatrice helper-process path.

Drives ``BeatriceWorkerClient`` like an audio callback: every
``blocksize * 1000 / 48000`` ms it sends one 48 kHz block and waits for the
converted block, recording the per-block round-trip latency.  This measures
whether moving the engine into its own process leaves enough headroom for the
GUI process to run Qt and the audio callback.

Run it from the repository root with the board's Python 3.13 environment:

    VOICE_CHANGER_BEATRICE_API_DIR=... VOICE_CHANGER_BEATRICE_MODEL_DIR=... \
      .venv/bin/python scripts/bench_beatrice_worker.py --speaker 5 --seconds 20

The numbers are measurement only: a pass here does not make the engine an
accepted realtime path.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_changer.ai import beatrice  # noqa: E402
from voice_changer.ai.beatrice_subprocess import BeatriceWorkerClient  # noqa: E402


class _Config:
    samplerate = 48000
    channels = 1

    def __init__(self, blocksize: int) -> None:
        self.blocksize = blocksize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--speaker", type=int, default=0, help="JVS speaker index (0-based)")
    parser.add_argument("--blocksize", type=int, default=1024, choices=(1024, 2496))
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--queue", action="store_true",
                        help="simulate the live chain: paced audio callback plus converter thread")
    args = parser.parse_args(argv)

    engine = beatrice.BeatriceConfig(
        api_dir=Path(args.api_dir).expanduser(),
        model_dir=Path(args.model_dir).expanduser(),
        speaker=args.speaker,
    )
    config = _Config(args.blocksize)
    period = args.blocksize / 48000.0
    budget_ms = period * 1000.0

    # Speech-like test signal: 170 Hz plus harmonics and a little noise.
    t = np.arange(args.blocksize, dtype=np.float32) / 48000.0
    block = (0.2 * np.sin(2 * np.pi * 170.0 * t)
             + 0.1 * np.sin(2 * np.pi * 340.0 * t)
             + 0.03 * np.sin(2 * np.pi * 510.0 * t))

    start = time.monotonic()
    if args.queue:
        return _run_queue(client_factory=lambda: BeatriceWorkerClient(config, engine),
                          block_size=args.blocksize, period=period, seconds=args.seconds,
                          budget_ms=budget_ms)
    client = BeatriceWorkerClient(config, engine)
    ready_ms = (time.monotonic() - start) * 1000.0
    print(f"worker ready in {ready_ms:.0f} ms (speaker {args.speaker}, block {args.blocksize})")

    latencies: list[float] = []
    missed = 0
    blocks = int(round(args.seconds / period))
    started = time.monotonic()
    last_rms = 0.0
    try:
        for index in range(blocks):
            deadline = started + index * period
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            began = time.monotonic()
            try:
                output = client.convert(block)
            except Exception as error:  # pragma: no cover - board only
                print(f"block {index}: conversion failed: {error}")
                return 2
            elapsed_ms = (time.monotonic() - began) * 1000.0
            latencies.append(elapsed_ms)
            if elapsed_ms > budget_ms:
                missed += 1
            last_rms = float(np.sqrt(np.mean(output.astype(np.float64) ** 2)))
    finally:
        client.close()

    total = time.monotonic() - started
    ordered = sorted(latencies)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    print(f"blocks={len(latencies)} in {total:.2f} s, budget={budget_ms:.2f} ms/block")
    print(f"latency ms: mean={statistics.fmean(latencies):.3f} p50={p50:.3f} "
          f"p95={p95:.3f} max={max(ordered):.3f}")
    print(f"missed deadlines: {missed} ({100.0 * missed / max(1, len(latencies)):.1f} %), "
          f"last block rms={last_rms:.4f}")
    print("NOTE: helper-process experiment only; not an accepted realtime path.")
    return 0 if missed == 0 else 1


def _run_queue(*, client_factory, block_size: int, period: float, seconds: float,
               budget_ms: float) -> int:
    """Simulate the app: paced callback + a converter thread with a small ring."""
    import queue
    import threading

    client = client_factory()
    in_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=16)
    out_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=16)
    # RvcWorkerProcessor reserves a few converted blocks before the callback
    # starts consuming (startup_blocks=4 in the production factory).
    for _ in range(4):
        out_q.put(np.zeros(block_size, dtype=np.float32))
    stop = threading.Event()
    latencies: list[float] = []
    converted = 0

    def converter() -> None:
        nonlocal converted
        while not stop.is_set():
            try:
                block_in = in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            began = time.monotonic()
            try:
                output = client.convert(block_in)
            except Exception as error:  # pragma: no cover - board only
                print(f"conversion failed: {error}")
                stop.set()
                return
            latencies.append((time.monotonic() - began) * 1000.0)
            converted += 1
            try:
                out_q.put(output, timeout=0.5)
            except queue.Full:
                pass

    thread = threading.Thread(target=converter, daemon=True)
    thread.start()
    t = np.arange(block_size, dtype=np.float32) / 48000.0
    block = (0.2 * np.sin(2 * np.pi * 170.0 * t)
             + 0.1 * np.sin(2 * np.pi * 340.0 * t)
             + 0.03 * np.sin(2 * np.pi * 510.0 * t))
    underruns = 0
    dropped = 0
    ticks = int(round(seconds / period))
    started = time.monotonic()
    for index in range(ticks):
        deadline = started + index * period
        delay = deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        try:
            in_q.put_nowait(block)
        except queue.Full:
            dropped += 1
        try:
            out_q.get_nowait()
        except queue.Empty:
            underruns += 1
    stop.set()
    thread.join(timeout=10)
    client.close()
    ordered = sorted(latencies)
    p50 = statistics.median(ordered) if ordered else 0.0
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))] if ordered else 0.0
    print(f"queue mode: ticks={ticks} converted={converted} budget={budget_ms:.2f} ms")
    if ordered:
        print(f"latency ms: mean={statistics.fmean(ordered):.3f} p50={p50:.3f} "
              f"p95={p95:.3f} max={max(ordered):.3f}")
    print(f"underruns={underruns} ({100.0 * underruns / max(1, ticks):.1f} %), "
          f"input drops={dropped}")
    print("NOTE: helper-process experiment only; not an accepted realtime path.")
    return 0 if underruns == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
