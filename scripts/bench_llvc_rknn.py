#!/usr/bin/env python3
"""Compare fixed-shape LLVC ONNX and RKNN inference using synthetic memory inputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from voice_changer.ai.llvc import (
    INPUTS,
    OUTPUTS,
    RKNN_INPUT_SHAPES,
    RKNN_OUTPUT_SHAPES,
    STABLE_MODEL_SHA256,
    LlvcConverter,
    LlvcProcessorFactory,
    _RknnSession,
    _validate_rknn_package,
)
from voice_changer.ai.rvc_worker import RvcWorkerProcessor
from voice_changer.audio.device_types import DeviceRef, StreamConfig


def run_benchmark(
    onnx_path: Path,
    rknn_path: Path,
    *,
    blocks: int,
    warmup: int,
    rtol: float,
    atol: float,
) -> tuple[dict[str, object], bool]:
    import onnxruntime as ort

    if blocks < 1 or warmup < 0:
        raise ValueError("blocks must be positive and warmup must not be negative")
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be finite and non-negative")
    with onnx_path.open("rb") as source:
        onnx_digest = LlvcConverter._hash(source)
    if onnx_digest != STABLE_MODEL_SHA256:
        raise ValueError("Only the validated fixed-shape 104 ms LLVC graph can be benchmarked")
    _validate_rknn_package(rknn_path)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    cpu = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    if tuple(item.name for item in cpu.get_inputs()) != INPUTS:
        raise ValueError("LLVC ONNX input schema mismatch")
    if tuple(item.name for item in cpu.get_outputs()) != OUTPUTS:
        raise ValueError("LLVC ONNX output schema mismatch")

    npu = _RknnSession(rknn_path)
    try:
        cpu_state = [np.zeros(shape, dtype=np.float32) for shape in RKNN_INPUT_SHAPES[1:]]
        npu_state = [np.zeros(shape, dtype=np.float32) for shape in RKNN_INPUT_SHAPES[1:]]
        generator = np.random.default_rng(41357)
        cpu_ms: list[float] = []
        npu_ms: list[float] = []
        cpu_over_budget = 0
        npu_over_budget = 0
        max_abs = {name: 0.0 for name in OUTPUTS}
        failures = {name: 0 for name in OUTPUTS}
        reset_inputs: list[np.ndarray] = []
        total = blocks + warmup

        for index in range(total):
            audio = generator.normal(0.0, 0.02, RKNN_INPUT_SHAPES[0]).astype(np.float32)
            if index >= warmup and len(reset_inputs) < 3:
                reset_inputs.append(audio.copy())
            cpu_feed = dict(zip(INPUTS, [audio, *cpu_state]))
            npu_feed = dict(zip(INPUTS, [audio, *npu_state]))

            started = time.perf_counter_ns()
            cpu_result = cpu.run(list(OUTPUTS), cpu_feed)
            cpu_elapsed = (time.perf_counter_ns() - started) / 1_000_000
            started = time.perf_counter_ns()
            npu_result = npu.run(list(OUTPUTS), npu_feed)
            npu_elapsed = (time.perf_counter_ns() - started) / 1_000_000

            if len(cpu_result) != len(OUTPUTS) or len(npu_result) != len(OUTPUTS):
                raise RuntimeError("LLVC inference omitted audio or streaming state")
            for name, expected_shape, expected, actual in zip(
                OUTPUTS, RKNN_OUTPUT_SHAPES, cpu_result, npu_result,
            ):
                if (
                    expected.shape != expected_shape
                    or actual.shape != expected_shape
                    or expected.dtype != np.float32
                    or actual.dtype != np.float32
                    or not np.isfinite(expected).all()
                    or not np.isfinite(actual).all()
                ):
                    raise RuntimeError(f"LLVC {name} has invalid shape, dtype, or values")
                if index >= warmup:
                    difference = float(np.max(np.abs(expected - actual)))
                    max_abs[name] = max(max_abs[name], difference)
                    if not np.allclose(expected, actual, rtol=rtol, atol=atol):
                        failures[name] += 1
            cpu_state = cpu_result[1:]
            npu_state = npu_result[1:]
            if index >= warmup:
                cpu_ms.append(cpu_elapsed)
                npu_ms.append(npu_elapsed)
                cpu_over_budget += cpu_elapsed > 104
                npu_over_budget += npu_elapsed > 104

        def replay(session) -> list[list[np.ndarray]]:
            state = [np.zeros(shape, dtype=np.float32) for shape in RKNN_INPUT_SHAPES[1:]]
            sequence = []
            for audio in reset_inputs:
                feed = dict(zip(INPUTS, [audio, *state]))
                result = session.run(list(OUTPUTS), feed)
                if len(result) != len(OUTPUTS):
                    raise RuntimeError("LLVC reset replay omitted audio or streaming state")
                for name, expected_shape, value in zip(OUTPUTS, RKNN_OUTPUT_SHAPES, result):
                    if (
                        value.shape != expected_shape
                        or value.dtype != np.float32
                        or not np.isfinite(value).all()
                    ):
                        raise RuntimeError(f"LLVC reset replay returned invalid {name}")
                sequence.append([value.copy() for value in result])
                state = result[1:]
            return sequence

        cpu_first = replay(cpu)
        cpu_second = replay(cpu)
        npu_first = replay(npu)
        npu_second = replay(npu)
        reset_failures = {name: 0 for name in OUTPUTS}
        for first, second in ((cpu_first, cpu_second), (npu_first, npu_second)):
            for first_block, second_block in zip(first, second):
                for name, expected, actual in zip(OUTPUTS, first_block, second_block):
                    if not np.allclose(expected, actual, rtol=rtol, atol=atol):
                        reset_failures[name] += 1

        def summarize(values: list[float]) -> dict[str, float]:
            return {
                "mean_ms": float(np.mean(values)),
                "p95_ms": float(np.percentile(values, 95)),
                "max_ms": float(np.max(values)),
            }

        report: dict[str, object] = {
            "input": "deterministic synthetic float32 tensors in memory",
            "audio_playback": False,
            "target_block_ms": 104,
            "warmup_blocks": warmup,
            "measured_blocks": blocks,
            "comparison": {"rtol": rtol, "atol": atol, "max_abs_error": max_abs, "failed_blocks": failures},
            "zero_state_reset": {
                "replayed_blocks": len(reset_inputs),
                "failed_blocks": reset_failures,
            },
            "cpu_execution_provider": {
                **summarize(cpu_ms),
                "blocks_over_104ms": cpu_over_budget,
            },
            "rknn_lite": {
                **summarize(npu_ms),
                "blocks_over_104ms": npu_over_budget,
            },
        }
        passed = not any(failures.values()) and not any(reset_failures.values())
        return report, passed
    finally:
        npu.close()


def run_worker_smoke(rknn_path: Path, conversion_blocks: int = 12) -> dict[str, object]:
    device = DeviceRef(0, "synthetic input/output", 0)
    config = StreamConfig(device, device, 48000, 1024, 1, 0.05)
    worker = LlvcProcessorFactory(rknn_path)(config, None)
    if not isinstance(worker, RvcWorkerProcessor):
        raise TypeError("LLVC factory returned an unexpected worker")
    callbacks = 0
    muted = np.zeros((1024, 1), dtype=np.float32)
    samples = np.full((1024, 1), 0.01, dtype=np.float32)
    try:
        worker.start()
        target_frames = conversion_blocks * worker.conversion_frames
        deadline = time.monotonic() + conversion_blocks * 2.0 + 5.0
        while worker._processed < target_frames:
            worker.check_health()
            if time.monotonic() >= deadline:
                raise TimeoutError("Synthetic LLVC worker did not process the requested model blocks")
            output = np.empty_like(muted)
            if not worker.exchange(samples, output):
                worker.check_health()
                raise RuntimeError("Synthetic LLVC worker exchange failed")
            output.fill(0)
            if np.any(output):
                raise RuntimeError("Synthetic worker output was not forced to silence")
            callbacks += 1
            time.sleep(config.blocksize / config.samplerate)
        worker.check_health()
        return {
            "conversion_blocks": conversion_blocks,
            "device_callbacks": callbacks,
            "output_forced_silent": True,
            "audio_devices_opened": False,
            "audio_saved": False,
        }
    finally:
        worker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_model", type=Path)
    parser.add_argument("rknn_model", type=Path)
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=0.001)
    parser.add_argument("--atol", type=float, default=0.0001)
    arguments = parser.parse_args()
    report, passed = run_benchmark(
        arguments.onnx_model.expanduser(),
        arguments.rknn_model.expanduser(),
        blocks=arguments.blocks,
        warmup=arguments.warmup,
        rtol=arguments.rtol,
        atol=arguments.atol,
    )
    report["worker_smoke"] = run_worker_smoke(arguments.rknn_model.expanduser())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
