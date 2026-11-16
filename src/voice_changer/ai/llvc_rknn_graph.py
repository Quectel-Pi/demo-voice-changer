"""Prepare and numerically validate the fixed LLVC graph for RKNN conversion."""

from __future__ import annotations

import hashlib
import importlib.metadata
from itertools import product
import os
from pathlib import Path
import tempfile
from typing import Any

from .llvc import (
    INPUTS,
    OUTPUTS,
    RKNN_INPUT_SHAPES,
    RKNN_OUTPUT_SHAPES,
    RKNN_SIMULATOR_ATOL,
    RKNN_SIMULATOR_RTOL,
    STABLE_MODEL_SHA256,
)

TRANSFORM_ID = "onnxsim-static-shapes-scatternd-slice-concat-v1"
RTOL = RKNN_SIMULATOR_RTOL
ATOL = RKNN_SIMULATOR_ATOL


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _static_shape(value_info: Any) -> tuple[int, ...] | None:
    dimensions = tuple(
        dimension.dim_value
        for dimension in value_info.type.tensor_type.shape.dim
    )
    return dimensions if dimensions and all(size > 0 for size in dimensions) else None


def rewrite_static_scatter_nd(model: Any) -> int:
    """Replace contiguous constant-index ScatterND updates with Slice/Concat."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model.CopyFrom(onnx.shape_inference.infer_shapes(model))
    value_info = {
        item.name: item
        for item in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }

    def shape(name: str) -> tuple[int, ...] | None:
        item = value_info.get(name)
        return _static_shape(item) if item is not None else None

    constants = {
        item.name: numpy_helper.to_array(item)
        for item in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    constants[node.output[0]] = numpy_helper.to_array(attribute.t)

    nodes = []
    rewritten = 0
    serial = 0

    def append_slice(
        data: str,
        axis: int,
        start: int,
        end: int,
        label: str,
        pieces: list[str],
    ) -> None:
        nonlocal serial
        name = f"__llvc_scatter_{serial}_{label}"
        serial += 1
        values = (
            np.asarray([start], dtype=np.int64),
            np.asarray([end], dtype=np.int64),
            np.asarray([axis], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
        )
        suffixes = ("starts", "ends", "axes", "steps")
        initializer_names = [f"{name}_{suffix}" for suffix in suffixes]
        model.graph.initializer.extend(
            numpy_helper.from_array(value, name=initializer_name)
            for value, initializer_name in zip(values, initializer_names)
        )
        nodes.append(helper.make_node(
            "Slice",
            [data, *initializer_names],
            [name],
            name=f"{name}_slice",
        ))
        pieces.append(name)

    for node in model.graph.node:
        if node.op_type != "ScatterND":
            nodes.append(node)
            continue
        if len(node.input) != 3 or node.attribute:
            raise ValueError(f"Unsupported ScatterND form: {node.name}")

        data, indices, updates = node.input
        if indices not in constants:
            raise ValueError(f"ScatterND indices are not constant: {node.name}")
        data_shape = shape(data) or shape(node.output[0])
        updates_shape = shape(updates)
        if data_shape is None or updates_shape is None:
            raise ValueError(f"ScatterND has a non-static shape: {node.name}")

        index_values = np.asarray(constants[indices])
        if not np.issubdtype(index_values.dtype, np.integer) or index_values.ndim < 1:
            raise ValueError(f"ScatterND indices have an invalid type or rank: {node.name}")
        index_depth = index_values.shape[-1]
        if not 1 <= index_depth <= len(data_shape):
            raise ValueError(f"ScatterND index depth is invalid: {node.name}")
        rows = index_values.astype(np.int64, copy=False).reshape(-1, index_depth)
        coordinates = [
            sorted(np.unique(rows[:, axis]).tolist())
            for axis in range(index_depth)
        ]
        partial_axes = []
        for axis, (dimension, values) in enumerate(zip(data_shape, coordinates)):
            if any(value < 0 or value >= dimension for value in values):
                raise ValueError(f"ScatterND index is out of bounds: {node.name}")
            if values == list(range(dimension)):
                continue
            if values and values == list(range(values[0], values[-1] + 1)):
                partial_axes.append(axis)
                continue
            raise ValueError(f"ScatterND indices are not contiguous: {node.name}")

        expected_count = int(np.prod([len(values) for values in coordinates]))
        if (
            len(rows) != expected_count
            or rows.tolist() != [list(row) for row in product(*coordinates)]
        ):
            raise ValueError(
                f"ScatterND indices are not an ordered contiguous block: {node.name}"
            )
        if len(partial_axes) > 1:
            raise ValueError(
                f"ScatterND updates span multiple partial axes: {node.name}"
            )

        scatter_updates_shape = tuple(index_values.shape[:-1]) + data_shape[index_depth:]
        if updates_shape != scatter_updates_shape:
            raise ValueError(
                f"ScatterND updates shape mismatch for {node.name}: "
                f"{updates_shape} != {scatter_updates_shape}"
            )
        if partial_axes:
            axis = partial_axes[0]
            start = coordinates[axis][0]
            end = coordinates[axis][-1] + 1
            replacement_shape = tuple(
                end - start if index == axis else dimension
                for index, dimension in enumerate(data_shape)
            )
            pieces: list[str] = []
            if start:
                append_slice(data, axis, 0, start, "prefix", pieces)
            replacement = updates
            if updates_shape != replacement_shape:
                replacement = f"__llvc_scatter_{serial}_updates"
                serial += 1
                shape_name = f"{replacement}_shape"
                model.graph.initializer.append(numpy_helper.from_array(
                    np.asarray(replacement_shape, dtype=np.int64),
                    name=shape_name,
                ))
                nodes.append(helper.make_node(
                    "Reshape",
                    [updates, shape_name],
                    [replacement],
                    name=f"{replacement}_reshape",
                ))
            pieces.append(replacement)
            if end < data_shape[axis]:
                append_slice(data, axis, end, data_shape[axis], "suffix", pieces)
            nodes.append(helper.make_node(
                "Concat",
                pieces,
                [node.output[0]],
                axis=axis,
                name=f"{node.name}_slice_concat",
            ))
        else:
            if updates_shape == data_shape:
                nodes.append(helper.make_node(
                    "Identity",
                    [updates],
                    [node.output[0]],
                    name=f"{node.name}_identity",
                ))
            else:
                shape_name = f"__llvc_scatter_{serial}_full_shape"
                serial += 1
                model.graph.initializer.append(numpy_helper.from_array(
                    np.asarray(data_shape, dtype=np.int64),
                    name=shape_name,
                ))
                nodes.append(helper.make_node(
                    "Reshape",
                    [updates, shape_name],
                    [node.output[0]],
                    name=f"{node.name}_reshape",
                ))
        rewritten += 1

    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    model.CopyFrom(onnx.shape_inference.infer_shapes(model))
    onnx.checker.check_model(model)
    return rewritten


def _verify_equivalence(source: Path, candidate: Path) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    original = ort.InferenceSession(
        str(source), options, providers=["CPUExecutionProvider"],
    )
    prepared = ort.InferenceSession(
        str(candidate), options, providers=["CPUExecutionProvider"],
    )
    if tuple(item.name for item in original.get_inputs()) != INPUTS:
        raise ValueError("Source LLVC input names or order do not match the fixed graph")
    if tuple(item.name for item in original.get_outputs()) != OUTPUTS:
        raise ValueError("Source LLVC output names or order do not match the fixed graph")
    if tuple(item.name for item in prepared.get_inputs()) != INPUTS:
        raise ValueError("Prepared LLVC input names or order changed")
    if tuple(item.name for item in prepared.get_outputs()) != OUTPUTS:
        raise ValueError("Prepared LLVC output names or order changed")

    state_shapes = RKNN_INPUT_SHAPES[1:]
    generator = np.random.default_rng(41357)
    max_abs_error = {name: 0.0 for name in OUTPUTS}
    cases = (
        ("zero_state_silence", "zeros", "zeros"),
        ("random_state_audio", "random", "random"),
        ("positive_boundary", "positive", "positive"),
        ("negative_boundary", "negative", "negative"),
        ("zero_state_alternating", "zeros", "alternating"),
    )
    blocks_compared = 0

    def initial_states(mode: str) -> list[np.ndarray]:
        if mode == "zeros":
            return [np.zeros(shape, dtype=np.float32) for shape in state_shapes]
        if mode == "random":
            return [
                generator.normal(0.0, 0.02, shape).astype(np.float32)
                for shape in state_shapes
            ]
        value = 1.0 if mode == "positive" else -1.0
        return [np.full(shape, value, dtype=np.float32) for shape in state_shapes]

    for _, state_mode, audio_mode in cases:
        source_state = initial_states(state_mode)
        prepared_state = [value.copy() for value in source_state]
        for block in range(3):
            if audio_mode == "zeros":
                audio = np.zeros(RKNN_INPUT_SHAPES[0], dtype=np.float32)
            elif audio_mode == "positive":
                audio = np.ones(RKNN_INPUT_SHAPES[0], dtype=np.float32)
            elif audio_mode == "negative":
                audio = -np.ones(RKNN_INPUT_SHAPES[0], dtype=np.float32)
            elif audio_mode == "alternating":
                audio = np.tile(
                    np.asarray([-1.0, 1.0], dtype=np.float32), 848,
                ).reshape(RKNN_INPUT_SHAPES[0])
            else:
                audio = generator.normal(
                    0.0, 0.02, RKNN_INPUT_SHAPES[0],
                ).astype(np.float32)

            source_feed = dict(zip(INPUTS, [audio, *source_state]))
            prepared_feed = dict(zip(INPUTS, [audio, *prepared_state]))
            source_result = original.run(list(OUTPUTS), source_feed)
            prepared_result = prepared.run(list(OUTPUTS), prepared_feed)
            if len(source_result) != len(OUTPUTS) or len(prepared_result) != len(OUTPUTS):
                raise RuntimeError("LLVC equivalence check omitted an output or state")
            for name, expected_shape, expected, actual in zip(
                OUTPUTS, RKNN_OUTPUT_SHAPES, source_result, prepared_result,
            ):
                if (
                    expected.shape != expected_shape
                    or actual.shape != expected_shape
                    or expected.dtype != np.float32
                    or actual.dtype != np.float32
                    or not np.isfinite(expected).all()
                    or not np.isfinite(actual).all()
                ):
                    raise RuntimeError(
                        f"LLVC equivalence check returned invalid {name} at block {block}"
                    )
                difference = float(np.max(np.abs(expected - actual)))
                max_abs_error[name] = max(max_abs_error[name], difference)
                if not np.allclose(expected, actual, rtol=RTOL, atol=ATOL):
                    raise RuntimeError(
                        f"LLVC graph rewrite failed for {name} at block {block}: "
                        f"max_abs_error={difference}"
                    )
            source_state = source_result[1:]
            prepared_state = prepared_result[1:]
            blocks_compared += 1

    return {
        "passed": True,
        "rtol": RTOL,
        "atol": ATOL,
        "continuous_blocks_compared": blocks_compared,
        "output_tensors_compared": blocks_compared * len(OUTPUTS),
        "cases": [name for name, _, _ in cases],
        "max_abs_error": max_abs_error,
    }


def prepare_llvc_rknn_graph(source: Path, destination: Path) -> dict[str, Any]:
    """Simplify and validate the pinned LLVC graph without changing its interface."""
    import onnx
    from onnxsim import simplify

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file() or sha256_file(source) != STABLE_MODEL_SHA256:
        raise ValueError("Only the SHA-256-pinned 104 ms LLVC graph can be prepared")
    if destination.suffix.lower() != ".onnx":
        raise ValueError("Prepared graph destination must use the .onnx extension")
    if not destination.parent.is_dir():
        raise ValueError(f"Destination directory does not exist: {destination.parent}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite prepared graph: {destination}")

    original = onnx.load(str(source))
    if tuple(item.name for item in original.graph.input) != INPUTS:
        raise ValueError("LLVC source input names or order do not match the fixed graph")
    if tuple(item.name for item in original.graph.output) != OUTPUTS:
        raise ValueError("LLVC source output names or order do not match the fixed graph")
    simplified, valid = simplify(
        original,
        overwrite_input_shapes={
            name: list(shape)
            for name, shape in zip(INPUTS, RKNN_INPUT_SHAPES)
        },
        perform_optimization=True,
    )
    if not valid:
        raise RuntimeError("ONNX simplifier rejected the fixed LLVC graph")
    rewritten_scatter_count = rewrite_static_scatter_nd(simplified)

    final_inputs = tuple(item.name for item in simplified.graph.input)
    final_outputs = tuple(item.name for item in simplified.graph.output)
    if final_inputs != INPUTS or final_outputs != OUTPUTS:
        raise ValueError("Prepared LLVC tensor names or ordering changed")
    for item, expected_shape in zip(simplified.graph.input, RKNN_INPUT_SHAPES):
        if _static_shape(item) != expected_shape or item.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"Prepared LLVC input contract changed: {item.name}")
    for item, expected_shape in zip(simplified.graph.output, RKNN_OUTPUT_SHAPES):
        if _static_shape(item) != expected_shape or item.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"Prepared LLVC output contract changed: {item.name}")

    with tempfile.TemporaryDirectory(
        prefix=".llvc-graph-",
        dir=destination.parent,
    ) as temporary:
        temporary_graph = Path(temporary) / destination.name
        onnx.save(simplified, str(temporary_graph))
        equivalence = _verify_equivalence(source, temporary_graph)
        if not equivalence["passed"]:
            raise RuntimeError("Prepared LLVC graph failed ONNX Runtime equivalence")
        os.link(temporary_graph, destination)

    operator_counts: dict[str, int] = {}
    for node in simplified.graph.node:
        operator_counts[node.op_type] = operator_counts.get(node.op_type, 0) + 1
    return {
        "transform": TRANSFORM_ID,
        "source_sha256": STABLE_MODEL_SHA256,
        "graph_sha256": sha256_file(destination),
        "onnx_version": importlib.metadata.version("onnx"),
        "onnxsim_version": importlib.metadata.version("onnxsim"),
        "onnxruntime_version": importlib.metadata.version("onnxruntime"),
        "rewritten_scatternd_count": rewritten_scatter_count,
        "node_count": len(simplified.graph.node),
        "operator_counts": operator_counts,
        "equivalence": equivalence,
    }
