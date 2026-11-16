#!/usr/bin/env python3
"""Convert the validated 104 ms LLVC ONNX graph for the RK3576 NPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_changer.ai.llvc import (
    INPUTS,
    OUTPUTS,
    RKNN_INPUT_SHAPES,
    RKNN_MANIFEST_FORMAT,
    RKNN_OUTPUT_SHAPES,
    RKNN_SIMULATOR_BLOCKS_PER_CASE,
    RKNN_SIMULATOR_CASE_NAMES,
    STABLE_MODEL_SHA256,
)
from voice_changer.ai.llvc_rknn_graph import (
    ATOL,
    RTOL,
    prepare_llvc_rknn_graph,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_rknn_simulator(runtime: Any, graph: Path) -> dict[str, Any]:
    """Reject RKNN builds that fail continuous host-simulator equivalence."""
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    reference = ort.InferenceSession(
        str(graph), options, providers=["CPUExecutionProvider"],
    )
    if tuple(item.name for item in reference.get_inputs()) != INPUTS:
        raise ValueError("RKNN simulator input schema does not match LLVC")
    if tuple(item.name for item in reference.get_outputs()) != OUTPUTS:
        raise ValueError("RKNN simulator output schema does not match LLVC")

    initialized = runtime.init_runtime(target=None)
    if initialized != 0:
        raise RuntimeError(
            f"RKNN host simulator initialization failed (status {initialized})"
        )

    generator = np.random.default_rng(41357)
    state_shapes = RKNN_INPUT_SHAPES[1:]
    cases = (
        ("zero_state_silence", "zeros", "zeros"),
        ("fixed_random_audio_zero_state", "zeros", "random"),
        ("positive_boundary_zero_state", "zeros", "positive"),
        ("negative_boundary_zero_state", "zeros", "negative"),
        ("alternating_boundary_zero_state", "zeros", "alternating"),
        ("random_state_audio", "random", "random"),
    )
    case_names = [name for name, _, _ in cases]
    if case_names != list(RKNN_SIMULATOR_CASE_NAMES):
        raise RuntimeError("RKNN simulator case contract is inconsistent")
    max_abs_error = {name: 0.0 for name in OUTPUTS}
    blocks_compared = 0

    def initial_states(mode: str) -> list:
        if mode == "zeros":
            return [np.zeros(shape, dtype=np.float32) for shape in state_shapes]
        return [
            generator.normal(0.0, 0.02, shape).astype(np.float32)
            for shape in state_shapes
        ]

    def audio_block(mode: str) -> np.ndarray:
        shape = RKNN_INPUT_SHAPES[0]
        if mode == "zeros":
            return np.zeros(shape, dtype=np.float32)
        if mode == "random":
            return generator.normal(0.0, 0.02, shape).astype(np.float32)
        if mode == "alternating":
            return np.tile(
                np.asarray([-1.0, 1.0], dtype=np.float32), 848,
            ).reshape(shape)
        value = 1.0 if mode == "positive" else -1.0
        return np.full(shape, value, dtype=np.float32)

    for case_name, state_mode, audio_mode in cases:
        reference_state = initial_states(state_mode)
        simulator_state = [value.copy() for value in reference_state]
        for block in range(RKNN_SIMULATOR_BLOCKS_PER_CASE):
            audio = audio_block(audio_mode)
            reference_feed = dict(zip(INPUTS, [audio, *reference_state]))
            simulator_feed = dict(zip(INPUTS, [audio, *simulator_state]))
            expected = reference.run(list(OUTPUTS), reference_feed)
            actual = runtime.inference(
                inputs=[simulator_feed[name] for name in INPUTS],
                data_format=["nhwc", "nhwc", "nchw", "nhwc", "nhwc"],
            )
            if (
                expected is None
                or len(expected) != len(OUTPUTS)
                or actual is None
                or len(actual) != len(OUTPUTS)
            ):
                raise RuntimeError(
                    f"LLVC reference or RKNN simulator omitted outputs in "
                    f"{case_name} block {block}"
                )

            for name, shape, expected_value, actual_value in zip(
                OUTPUTS, RKNN_OUTPUT_SHAPES, expected, actual,
            ):
                expected_value = np.asarray(expected_value)
                actual_value = np.asarray(actual_value)
                if (
                    expected_value.shape != shape
                    or actual_value.shape != shape
                    or expected_value.dtype != np.float32
                    or actual_value.dtype != np.float32
                    or not np.isfinite(expected_value).all()
                    or not np.isfinite(actual_value).all()
                ):
                    raise RuntimeError(
                        f"RKNN simulator returned invalid {name} in "
                        f"{case_name} block {block}"
                    )
                difference = float(np.max(np.abs(expected_value - actual_value)))
                max_abs_error[name] = max(max_abs_error[name], difference)
                if not np.allclose(
                    expected_value, actual_value, rtol=RTOL, atol=ATOL,
                ):
                    raise RuntimeError(
                        f"RKNN host simulator equivalence failed for {name} "
                        f"in {case_name} block {block}: "
                        f"max_abs_error={difference} "
                        f"(rtol={RTOL}, atol={ATOL})"
                    )

            reference_state = expected[1:]
            simulator_state = actual[1:]
            blocks_compared += 1

    return {
        "passed": True,
        "rtol": RTOL,
        "atol": ATOL,
        "continuous_blocks_compared": blocks_compared,
        "output_tensors_compared": blocks_compared * len(OUTPUTS),
        "cases": case_names,
        "max_abs_error": max_abs_error,
    }


def convert(source: Path, destination: Path) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source.suffix.lower() != ".onnx" or not source.is_file():
        raise ValueError("Source must be an existing .onnx file")
    if sha256_file(source) != STABLE_MODEL_SHA256:
        raise ValueError("Only the validated fixed-shape 104 ms LLVC graph can be converted")
    if destination.suffix.lower() != ".rknn":
        raise ValueError("Destination must use the .rknn extension")
    if not destination.parent.is_dir():
        raise ValueError(f"Destination directory does not exist: {destination.parent}")
    manifest_path = Path(f"{destination}.json")
    if destination.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing RKNN artifact: {destination}")

    from rknn.api import RKNN

    with tempfile.TemporaryDirectory(prefix=".llvc-rknn-", dir=destination.parent) as temporary:
        temporary_model = Path(temporary) / destination.name
        prepared_graph = Path(temporary) / "llvc-prepared.onnx"
        runtime = RKNN(verbose=True)
        previous_cwd = Path.cwd()
        try:
            os.chdir(temporary)
            graph_preparation = prepare_llvc_rknn_graph(source, prepared_graph)
            if (
                graph_preparation.get("source_sha256") != STABLE_MODEL_SHA256
                or graph_preparation.get("equivalence", {}).get("passed") is not True
                or graph_preparation.get("graph_sha256") != sha256_file(prepared_graph)
            ):
                raise RuntimeError("Prepared LLVC graph failed its provenance or equivalence gate")
            configured = runtime.config(target_platform="rk3576")
            if configured != 0:
                raise RuntimeError(f"RKNN configuration failed (status {configured})")
            loaded = runtime.load_onnx(model=str(prepared_graph))
            if loaded != 0:
                raise RuntimeError(f"RKNN ONNX load failed (status {loaded})")
            built = runtime.build(do_quantization=False)
            if built != 0:
                raise RuntimeError(f"RKNN build failed (status {built})")
            simulator_equivalence = verify_rknn_simulator(runtime, prepared_graph)
            exported = runtime.export_rknn(str(temporary_model))
            if exported != 0 or not temporary_model.is_file():
                raise RuntimeError(f"RKNN export failed (status {exported})")
        finally:
            try:
                runtime.release()
            finally:
                os.chdir(previous_cwd)

        manifest = {
            "format": RKNN_MANIFEST_FORMAT,
            "source_sha256": STABLE_MODEL_SHA256,
            "target_platform": "rk3576",
            "quantized": False,
            "model_sha256": sha256_file(temporary_model),
            "inputs": list(INPUTS),
            "outputs": list(OUTPUTS),
            "input_shapes": [list(shape) for shape in RKNN_INPUT_SHAPES],
            "output_shapes": [list(shape) for shape in RKNN_OUTPUT_SHAPES],
            "toolkit_version": importlib.metadata.version("rknn-toolkit2"),
            "conversion_graph_sha256": graph_preparation["graph_sha256"],
            "graph_preparation": graph_preparation,
            "host_simulator_equivalence": simulator_equivalence,
        }
        temporary_manifest = Path(temporary) / f"{destination.name}.json"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists() or manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing RKNN artifact: {destination}")
        os.link(temporary_model, destination)
        try:
            os.link(temporary_manifest, manifest_path)
        except OSError:
            destination.unlink(missing_ok=True)
            raise
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="validated llvc-stateful-8.onnx")
    parser.add_argument("destination", type=Path, help="new .rknn output path")
    arguments = parser.parse_args()
    manifest = convert(arguments.source.expanduser(), arguments.destination.expanduser())
    print(f"RKNN model and manifest created: {manifest}")
    print("Inspect the conversion report and run board-side output/state equivalence tests before use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
