#!/usr/bin/env python3
"""Convert a HEEVoicechanger ONNX model into a validated RK3576 RKNN package.

This script runs on the host (x86_64) with ``rknn-toolkit2>=2.5.0``. Packages
compiled by the 2.3.2 toolkit are numerically wrong on the RK3576 NPU for this
network (the simulator agrees with ONNX, the device does not), while 2.5.0
packages run correctly on the board's ``librknnrt 2.3.2`` runtime. It never runs
on the board and never downloads weights: the source ONNX is a local file that
the user provides (the repository does not bundle HEVC weights).

Pipeline:
  1. fix ``audio_in`` to the runtime window (8192 history + 3072 chunk),
  2. simplify the graph, rewrite its constant-index ``ScatterND`` nodes and
     stabilise the LayerNorm blocks so fp16 activations stay in range,
  3. compare the rewritten graph against the original with ONNX Runtime,
  4. build a non-quantized fp16 RKNN model for ``rk3576``,
  5. compare the RKNN simulator against the rewritten graph,
  6. write ``<model>_rknn.rknn`` plus a ``<model>_rknn.rknn.json`` sidecar that
     the app validates before the model may enter the live chain.

Every gate is strict on purpose: a package whose simulator accuracy is below
40 dB SNR is rejected instead of being shipped as "working".
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxsim
from onnx import TensorProto, helper, numpy_helper, shape_inference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_changer.ai import hevc_rknn  # noqa: E402
from voice_changer.ai.model_catalog import ENTRIES  # noqa: E402

FRAMES = hevc_rknn.WINDOW_SAMPLES
HISTORY = hevc_rknn.HISTORY_SAMPLES
CHUNK = hevc_rknn.CHUNK_SAMPLES
SNR_GATE_DB = hevc_rknn.MIN_SIMULATOR_SNR_DB
EQUIVALENCE_GATE = 0.01
SIMULATOR_MAX_ABS = 1e-2
SPEECH_CASES = ("speech", "random")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _known_sources() -> dict[str, str]:
    return {
        digest: entry.entry_id
        for entry in ENTRIES
        if entry.model_type == "hevc_onnx"
        for digest in entry.sha256
    }


def fix_input_shapes(model: onnx.ModelProto, frames: int) -> onnx.ModelProto:
    for item in model.graph.input:
        if item.name != "audio_in":
            continue
        dims = item.type.tensor_type.shape.dim
        dims[0].dim_value = 1
        dims[1].dim_value = 1
        dims[2].dim_value = frames
    return model


def tensor_shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for value in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        dims = value.type.tensor_type.shape.dim
        if dims and all(dim.HasField("dim_value") for dim in dims):
            shapes[value.name] = tuple(dim.dim_value for dim in dims)
    for init in model.graph.initializer:
        shapes[init.name] = tuple(init.dims)
    return shapes


def _rewrite_scatter_nd(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Replace constant-index ScatterND with Slice/Concat/Reshape.

    RKNN's compiler rejects the ScatterND pattern this graph uses for its
    rolling buffers. All indices are initializers and every update is one
    ordered, contiguous block, so the replacement is exact.
    """
    consts = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    shapes = tensor_shapes(model)
    nodes: list = []
    nodes_init: list = []
    serial = 0
    rewritten = 0

    def slice_piece(data: str, axis: int, start: int, end: int) -> str:
        nonlocal serial
        names = []
        for value in (start, end):
            name = f"__hevc_scatter_{serial}_{value}"
            serial += 1
            nodes_init.append(numpy_helper.from_array(np.asarray([value], dtype=np.int64), name=name))
            names.append(name)
        axis_name = f"__hevc_scatter_axis_{axis}"
        if axis_name not in consts:
            nodes_init.append(numpy_helper.from_array(np.asarray([axis], dtype=np.int64), name=axis_name))
            consts[axis_name] = np.asarray([axis], dtype=np.int64)
        piece = f"__hevc_scatter_{serial}_piece"
        serial += 1
        nodes.append(
            helper.make_node("Slice", [data, names[0], names[1], axis_name], [piece], name=f"{piece}_node")
        )
        return piece

    for node in model.graph.node:
        if node.op_type != "ScatterND":
            nodes.append(node)
            continue
        data, indices, updates = node.input
        index_values = consts.get(indices)
        if index_values is None:
            raise ValueError(f"non-constant ScatterND indices: {node.name}")
        index_values = np.asarray(index_values)
        depth = index_values.shape[-1]
        rows = index_values.reshape(-1, depth)
        data_shape = shapes.get(data)
        updates_shape = shapes.get(updates)
        if data_shape is None or updates_shape is None:
            raise ValueError(f"missing shapes for {node.name}: {data_shape} {updates_shape}")
        coordinates = [sorted({int(row[d]) for row in rows}) for d in range(depth)]
        expected = int(np.prod([len(items) for items in coordinates]))
        wanted = [list(row) for row in _cartesian(coordinates)]
        if len(rows) != expected or rows.tolist() != wanted:
            raise ValueError(f"not an ordered contiguous block: {node.name}")
        partial = [d for d in range(depth) if coordinates[d][-1] - coordinates[d][0] + 1 != data_shape[d]]
        if len(partial) > 1:
            raise ValueError(f"multiple partial axes: {node.name}")
        if partial:
            axis = partial[0]
            start = coordinates[axis][0]
            end = coordinates[axis][-1] + 1
            replacement_shape = tuple(end - start if i == axis else dim for i, dim in enumerate(data_shape))
            pieces = []
            if start:
                pieces.append(slice_piece(data, axis, 0, start))
            replacement = updates
            if tuple(updates_shape) != replacement_shape:
                shape_name = f"__hevc_scatter_{serial}_updates_shape"
                serial += 1
                nodes_init.append(
                    numpy_helper.from_array(np.asarray(replacement_shape, dtype=np.int64), name=shape_name)
                )
                replacement = f"__hevc_scatter_{serial}_updates"
                nodes.append(
                    helper.make_node("Reshape", [updates, shape_name], [replacement], name=f"{replacement}_node")
                )
            pieces.append(replacement)
            if end < data_shape[axis]:
                pieces.append(slice_piece(data, axis, end, data_shape[axis]))
            nodes.append(
                helper.make_node("Concat", pieces, [node.output[0]], axis=axis, name=f"{node.name}_concat")
            )
        else:
            if tuple(updates_shape) == data_shape:
                nodes.append(helper.make_node("Identity", [updates], [node.output[0]], name=f"{node.name}_identity"))
            else:
                shape_name = f"__hevc_scatter_{serial}_full_shape"
                serial += 1
                nodes_init.append(numpy_helper.from_array(np.asarray(data_shape, dtype=np.int64), name=shape_name))
                nodes.append(
                    helper.make_node("Reshape", [updates, shape_name], [node.output[0]], name=f"{node.name}_reshape")
                )
        rewritten += 1

    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(nodes_init)
    model = shape_inference.infer_shapes(model, strict_mode=False)
    onnx.checker.check_model(model)
    return model, rewritten


def _cartesian(coordinates: list[list[int]]):
    from itertools import product

    return product(*coordinates)

LN_SCALE = 0.0625  # 1/16: keeps the LayerNorm Pow(2) outputs inside fp16 range.


def _stabilise_layer_norm(model: onnx.ModelProto, scale: float = LN_SCALE) -> dict:
    """Rewrite ``LayerNorm`` so the squared deviation stays inside fp16 range.

    The HiFi-GAN generator computes ``norm(x - mean) / sqrt(mean((x - mean)^2) + eps)``.
    With the HEVC activations some LayerNorm inputs reach ~1e3, so the ``Pow(2)``
    output reaches ~1.4e6 and overflows fp16 on the RK3576 NPU (the toolkit
    simulator computes it in fp32, so it does not see the problem). Scaling the
    deviation by ``s`` and the epsilon by ``s^2`` keeps the LayerNorm result
    unchanged while making the squared values ``s^2`` smaller:

        sqrt(mean((s*d)^2) + s^2*eps) = s * sqrt(mean(d^2) + eps)

    The divisor is scaled by ``s`` as well, so ``s*d / s*sqrt(...)`` is exactly
    the original LayerNorm in real arithmetic.
    """

    graph = model.graph
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for output in node.output:
            producer[output] = node
    initializers = {init.name: init for init in graph.initializer}
    constants = {name: numpy_helper.to_array(init) for name, init in initializers.items()}

    chosen: list[tuple[onnx.NodeProto, onnx.NodeProto, onnx.NodeProto]] = []
    for node in graph.node:
        if node.op_type != "Pow" or len(node.input) != 2:
            continue
        exponent = constants.get(node.input[1])
        if exponent is None or exponent.size != 1 or float(np.ravel(exponent)[0]) != 2.0:
            continue
        subtract = producer.get(node.input[0])
        if subtract is None or subtract.op_type != "Sub":
            continue
        divisors = [
            consumer
            for consumer in _consumers(graph, subtract.output[0])
            if consumer.op_type == "Div" and consumer.input[0] == subtract.output[0]
        ]
        if len(divisors) != 1:
            continue
        squared_mean = [
            consumer for consumer in _consumers(graph, node.output[0]) if consumer.op_type == "ReduceMean"
        ]
        if len(squared_mean) != 1:
            continue
        adds = [
            consumer
            for consumer in _consumers(graph, squared_mean[0].output[0])
            if consumer.op_type == "Add" and any(name in constants for name in consumer.input)
        ]
        if len(adds) != 1:
            continue
        add = adds[0]
        epsilon_name = next(name for name in add.input if name in constants)
        square_roots = [consumer for consumer in _consumers(graph, add.output[0]) if consumer.op_type == "Sqrt"]
        if not square_roots or divisors[0].input[1] != square_roots[0].output[0]:
            continue
        chosen.append((node, add, epsilon_name))

    if not chosen:
        return {"layer_norms_scaled": 0, "scale": scale}

    inserted: dict[int, onnx.NodeProto] = {}
    counter = 0
    for pow_node, add, epsilon_name in chosen:
        epsilon_value = np.asarray(constants[epsilon_name], dtype=np.float32)
        scaled_epsilon = (epsilon_value * np.float32(scale * scale)).astype(np.float32)
        epsilon_scaled_name = f"{epsilon_name}_ln{counter}_scaled"
        graph.initializer.append(numpy_helper.from_array(scaled_epsilon, epsilon_scaled_name))
        for index, name in enumerate(add.input):
            if name == epsilon_name:
                add.input[index] = epsilon_scaled_name

        scale_name = f"ln_scale_{counter}"
        graph.initializer.append(
            numpy_helper.from_array(np.asarray([scale], dtype=np.float32), scale_name)
        )
        subtract = producer[pow_node.input[0]]
        mul_name = f"{subtract.name or subtract.output[0]}_lnscale"
        mul_output = f"{mul_name}_output_0"
        mul = helper.make_node("Mul", [subtract.output[0], scale_name], [mul_output], name=mul_name)
        pow_node.input[0] = mul_output
        for consumer in _consumers(graph, subtract.output[0]):
            if consumer.op_type == "Div" and consumer.input[0] == subtract.output[0]:
                consumer.input[0] = mul_output
                break
        inserted[id(pow_node)] = mul
        counter += 1

    rewritten: list[onnx.NodeProto] = []
    for node in graph.node:
        mul = inserted.get(id(node))
        if mul is not None:
            rewritten.append(mul)
        rewritten.append(node)
    graph.ClearField("node")
    graph.node.extend(rewritten)
    return {"layer_norms_scaled": counter, "scale": scale}


def _consumers(graph: onnx.GraphProto, tensor: str) -> list[onnx.NodeProto]:
    return [node for node in graph.node if tensor in node.input]


def _rewrite_conv_transpose(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Replace ConvTranspose with Slice/Transpose/MatMul phases.

    The RK3576 device computes this graph's final ConvTranspose incorrectly even
    though the toolkit simulator agrees with ONNX Runtime (verified on device:
    correlation 0.012 for a two-node ConvTranspose+Tanh model).  A transposed
    convolution with stride S and symmetric padding is exactly a bank of S
    polyphase filters, so each phase is a matrix multiply on a shifted slice of
    the input; that formulation matches ONNX Runtime bit-closely and computes
    correctly on the NPU.
    """
    initializers = {item.name: item for item in model.graph.initializer}
    value_info = {item.name: item for item in model.graph.value_info}
    value_info.update({item.name: item for item in model.graph.output})
    replaced = 0
    rewritten_nodes: list[onnx.NodeProto] = []
    extra_initializers: list[onnx.TensorProto] = []
    for node in model.graph.node:
        if node.op_type != "ConvTranspose":
            rewritten_nodes.append(node)
            continue
        attributes = {attribute.name: attribute for attribute in node.attribute}
        if "strides" not in attributes or len(attributes["strides"].ints) != 1:
            rewritten_nodes.append(node)
            continue
        stride = attributes["strides"].ints[0]
        pads = attributes["pads"].ints
        if len(pads) != 2 or pads[0] != pads[1] or pads[0] % stride:
            rewritten_nodes.append(node)
            continue
        kernel = attributes["kernel_shape"].ints[0]
        taps = (kernel + stride - 1) // stride
        lead = pads[0] // stride
        weight = numpy_helper.to_array(initializers[node.input[1]])
        if weight.ndim != 3 or weight.shape[1] != 1:
            rewritten_nodes.append(node)
            continue
        source = value_info.get(node.input[0])
        target = value_info.get(node.output[0])
        if source is None or target is None or source.type.tensor_type.shape.dim[2].dim_value <= 0:
            rewritten_nodes.append(node)
            continue
        source_length = source.type.tensor_type.shape.dim[2].dim_value
        output_length = target.type.tensor_type.shape.dim[2].dim_value
        if output_length <= 0 or output_length % stride:
            rewritten_nodes.append(node)
            continue
        phases = output_length // stride
        if lead - (taps - 1) < 0 or lead + phases > source_length:
            rewritten_nodes.append(node)
            continue
        prefix = f"{node.name or 'conv_transpose'}__phase"
        terms: list[str] = []
        for step in range(taps):
            matrix = np.ascontiguousarray(weight[:, 0, [phase + stride * step for phase in range(stride)]], dtype=np.float32)
            matrix_name = f"{prefix}_w{step}"
            extra_initializers.append(numpy_helper.from_array(matrix, matrix_name))
            for suffix, value in (("start", lead - step), ("end", lead - step + phases)):
                name = f"{prefix}_{suffix}{step}"
                extra_initializers.append(numpy_helper.from_array(np.array([value], dtype=np.int64), name))
            axes_name = f"{prefix}_axes{step}"
            if not any(item.name == axes_name for item in extra_initializers):
                extra_initializers.append(numpy_helper.from_array(np.array([2], dtype=np.int64), axes_name))
            sliced, transposed, product = f"{prefix}_x{step}", f"{prefix}_xt{step}", f"{prefix}_p{step}"
            rewritten_nodes.append(helper.make_node("Slice", [node.input[0], f"{prefix}_start{step}", f"{prefix}_end{step}", axes_name], [sliced], name=f"{prefix}_slice{step}"))
            rewritten_nodes.append(helper.make_node("Transpose", [sliced], [transposed], name=f"{prefix}_transpose{step}", perm=[0, 2, 1]))
            rewritten_nodes.append(helper.make_node("MatMul", [transposed, matrix_name], [product], name=f"{prefix}_matmul{step}"))
            terms.append(product)
        accumulator = terms[0]
        for index, term in enumerate(terms[1:], start=1):
            node_name = f"{prefix}_add{index}"
            rewritten_nodes.append(helper.make_node("Add", [accumulator, term], [node_name], name=node_name))
            accumulator = node_name
        shape_name = f"{prefix}_shape"
        extra_initializers.append(numpy_helper.from_array(np.array([1, 1, output_length], dtype=np.int64), shape_name))
        rewritten_nodes.append(helper.make_node("Reshape", [accumulator, shape_name], [node.output[0]], name=f"{prefix}_reshape"))
        replaced += 1
    model.graph.ClearField("node")
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(extra_initializers)
    return model, replaced


def _rewrite_transpose_conv_sandwich(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Replace Transpose/Relu/Conv1x1/Transpose with Relu/MatMul.

    The RK3576 device miscomputes this graph's 1x1 convolution when it is fed by
    a transpose inside the large encoder graph, while the toolkit simulator and
    every isolated op match ONNX Runtime (verified on device: the feeding tensor
    reads back correctly but the convolution result has correlation 0.016).
    With T the transpose of the last two axes, the sandwich

        y = T(Conv1x1(Relu(T(x))))

    equals ``Relu(x) @ W^T + bias`` in the untransposed frame, where the weight
    has shape [Cout, Cin, 1] and the matrix is ``weight[:, :, 0].T``.  The
    rewrite is exact and removes both transposes and the 1x1 convolution.
    """
    graph = model.graph
    initializers = {item.name: item for item in graph.initializer}
    producers: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for name in node.output:
            producers[name] = node

    def attribute(node: onnx.NodeProto, name: str):
        return next((item for item in node.attribute if item.name == name), None)

    def is_transpose_swap(node: onnx.NodeProto) -> bool:
        if node.op_type != "Transpose":
            return False
        perm = attribute(node, "perm")
        return perm is not None and list(perm.ints) == [0, 2, 1]

    def single_consumer(tensor: str) -> onnx.NodeProto | None:
        consumers = [node for node in graph.node if tensor in node.input]
        return consumers[0] if len(consumers) == 1 else None

    drops: set[int] = set()
    replacements: dict[str, list[onnx.NodeProto]] = {}
    count = 0
    for node in graph.node:
        if not is_transpose_swap(node) or id(node) in drops:
            continue
        source = node.input[0]
        relu = single_consumer(node.output[0])
        if relu is None or relu.op_type != "Relu" or id(relu) in drops:
            continue
        conv = single_consumer(relu.output[0])
        if conv is None or conv.op_type != "Conv" or id(conv) in drops:
            continue
        kernel = attribute(conv, "kernel_shape")
        pads = attribute(conv, "pads")
        strides = attribute(conv, "strides")
        group = attribute(conv, "group")
        if kernel is None or list(kernel.ints) != [1]:
            continue
        if pads is not None and any(pads.ints):
            continue
        if strides is not None and any(value != 1 for value in strides.ints):
            continue
        if group is not None and group.i != 1:
            continue
        weight_proto = initializers.get(conv.input[1])
        if weight_proto is None:
            continue
        weight = numpy_helper.to_array(weight_proto)
        if weight.ndim != 3 or weight.shape[2] != 1:
            continue
        transpose_after = single_consumer(conv.output[0])
        if transpose_after is None or not is_transpose_swap(transpose_after):
            continue
        matrix_name = f"{conv.name or 'conv1x1'}__matmul_weight_t"
        matrix = np.ascontiguousarray(weight[:, :, 0].T, dtype=np.float32)
        graph.initializer.append(numpy_helper.from_array(matrix, matrix_name))
        relu_out = f"{relu.output[0]}__untransposed"
        product = f"{conv.output[0]}__matmul"
        nodes = [
            helper.make_node("Relu", [source], [relu_out], name=f"{relu.name or 'relu'}__untransposed"),
            helper.make_node("MatMul", [relu_out, matrix_name], [product], name=f"{conv.name or 'conv'}__matmul"),
        ]
        if len(conv.input) > 2 and initializers.get(conv.input[2]) is not None:
            nodes.append(helper.make_node("Add", [product, conv.input[2]], [transpose_after.output[0]], name=f"{transpose_after.name or 'transpose'}__bias"))
        else:
            nodes.append(helper.make_node("Identity", [product], [transpose_after.output[0]], name=f"{transpose_after.name or 'transpose'}__untransposed"))
        replacements[transpose_after.output[0]] = nodes
        drops.update({id(node), id(relu), id(conv), id(transpose_after)})
        count += 1
    if not count:
        return model, 0
    rewritten: list[onnx.NodeProto] = []
    dropped_ids = set(drops)
    for node in graph.node:
        if id(node) in dropped_ids:
            if node.output and node.output[0] in replacements:
                rewritten.extend(replacements[node.output[0]])
            continue
        # The first transpose of a sandwich is dropped as well, but its output
        # replacement is attached to the trailing transpose above.
        rewritten.append(node)
    graph.ClearField("node")
    graph.node.extend(rewritten)
    return model, count


def prepare(source: Path, destination: Path, frames: int) -> dict:
    model = onnx.load(str(source))
    fix_input_shapes(model, frames)
    model = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    simplified, ok = onnxsim.simplify(model, overwrite_input_shapes={"audio_in": [1, 1, frames]})
    if not ok:
        raise RuntimeError("onnxsim simplify failed")
    simplified = shape_inference.infer_shapes(simplified, strict_mode=False, data_prop=True)
    nodes_before = len(simplified.graph.node)
    simplified, rewritten = _rewrite_scatter_nd(simplified)
    stabilisation = _stabilise_layer_norm(simplified)
    simplified, transposed = _rewrite_conv_transpose(simplified)
    simplified, sandwiches = _rewrite_transpose_conv_sandwich(simplified)
    simplified = shape_inference.infer_shapes(simplified, strict_mode=False)
    onnx.checker.check_model(simplified)
    onnx.save(simplified, str(destination))
    return {
        "nodes_before_scatter_rewrite": nodes_before,
        "nodes_after": len(simplified.graph.node),
        "scatter_rewritten": rewritten,
        "conv_transpose_rewritten": transposed,
        "transpose_conv_sandwiches": sandwiches,
        "layer_norms_scaled": stabilisation["layer_norms_scaled"],
        "layer_norm_scale": stabilisation["scale"],
        "frames": frames,
    }


def _ort_session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = reference.astype(np.float64).reshape(-1)
    candidate = candidate.astype(np.float64).reshape(-1)
    noise = candidate - reference
    signal_power = float(np.mean(reference**2))
    noise_power = float(np.mean(noise**2))
    if noise_power <= 0:
        return 120.0
    if signal_power <= 0:
        return -120.0
    ratio = 10.0 * np.log10(signal_power / noise_power)
    return float(min(ratio, 120.0))


def _cases(frames: int, real_audio: np.ndarray | None) -> dict[str, np.ndarray]:
    cases = {
        "silence": np.zeros(frames, dtype=np.float32),
        "random": (np.random.default_rng(7).standard_normal(frames) * 0.1).astype(np.float32),
        "boundary": np.concatenate(
            [
                np.full(frames // 2, 0.9, dtype=np.float32),
                np.full(frames - frames // 2, -0.9, dtype=np.float32),
            ]
        ),
    }
    if real_audio is not None and real_audio.size:
        if real_audio.size >= frames:
            cases["speech"] = real_audio[:frames].astype(np.float32)
        else:
            cases["speech"] = np.pad(real_audio.astype(np.float32), (0, frames - real_audio.size))
    return cases


def compare_ort(original: Path, candidate: Path, cases: dict[str, np.ndarray]) -> dict:
    want_session = _ort_session(original)
    got_session = _ort_session(candidate)
    report = {}
    for name, samples in cases.items():
        batch = samples.reshape(1, 1, -1)
        want = want_session.run(["audio_out"], {"audio_in": batch})[0]
        got = got_session.run(["audio_out"], {"audio_in": batch})[0]
        diff = np.abs(want - got)
        report[name] = {
            "max_abs": float(diff.max()),
            "snr_db": _snr_db(want, got),
        }
    return report


def build_and_compare(prepared: Path, output: Path, cases: dict[str, np.ndarray]) -> tuple[dict, dict]:
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform=hevc_rknn.TARGET_PLATFORM,
        optimization_level=3,
        disable_rules=["convert_rs_add_rs_to_rs_gather_elements"],
    )
    try:
        loaded = rknn.load_onnx(
            model=str(prepared), inputs=["audio_in"], input_size_list=[[1, 1, FRAMES]], outputs=["audio_out"]
        )
        if loaded != 0:
            raise RuntimeError(f"load_onnx failed with code {loaded}")
        built = rknn.build(do_quantization=False)
        if built != 0:
            raise RuntimeError(f"rknn build failed with code {built}")
        exported = rknn.export_rknn(str(output))
        if exported != 0:
            raise RuntimeError(f"export_rknn failed with code {exported}")
        if rknn.init_runtime() != 0:
            raise RuntimeError("simulator init_runtime failed")
        prepared_session = _ort_session(prepared)
        report = {}
        for name, samples in cases.items():
            batch = samples.reshape(1, 1, -1)
            want = prepared_session.run(["audio_out"], {"audio_in": batch})[0]
            got = rknn.inference(inputs=[batch], data_format="nchw")[0]
            got = np.asarray(got).reshape(-1)
            if not np.isfinite(got).all():
                raise RuntimeError(f"simulator returned non-finite samples for case {name!r}")
            report[name] = {
                "max_abs": float(np.max(np.abs(want.reshape(-1) - got))),
                "snr_db": _snr_db(want, got),
            }
        return report, {"toolkit": _toolkit_version()}
    finally:
        rknn.release()


MIN_TOOLKIT_VERSION = (2, 5, 0)


def _toolkit_version() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("rknn-toolkit2")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - toolkit env only
        return "unknown"


def _toolkit_version_tuple() -> tuple[int, int, int]:
    try:
        parts = [int(part) for part in _toolkit_version().split(".")[:3]]
    except ValueError:  # pragma: no cover - toolkit env only
        return (0, 0, 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)  # type: ignore[return-value]


def _require_toolkit() -> None:
    found = _toolkit_version_tuple()
    if found < MIN_TOOLKIT_VERSION:
        raise SystemExit(
            f"rknn-toolkit2 {_toolkit_version()} 版本过旧：2.3.2 编译器会把该网络编译错。"
            " 请使用 rknn-toolkit2>=2.5.0。"
            " / rknn-toolkit2 is too old: the 2.3.2 compiler miscompiles this network "
            "on RK3576. Use rknn-toolkit2>=2.5.0."
        )


def _read_reference_audio(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    import scipy.io.wavfile as wav
    import scipy.signal as sig

    rate, data = wav.read(str(path))
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        data = data.astype(np.float64) / max(abs(info.min), info.max)
    data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != 16000:
        data = sig.resample_poly(data, 16000, rate).astype(np.float32)
    return data


def convert(source: Path, output_dir: Path, reference_audio: Path | None) -> dict:
    entries = _known_sources()
    digest = _sha256(source)
    entry_id = entries.get(digest)
    if entry_id is None:
        raise SystemExit(
            f"{source.name}: 不是已知的 HEEVoicechanger ONNX（SHA-256 {digest}）。"
            " / Not a known HEEVoicechanger ONNX; the repository does not convert unknown weights."
        )
    _require_toolkit()
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    prepared_path = output_dir / f"{stem}_prepared.onnx"
    model_path = output_dir / f"{stem}_rknn.rknn"
    manifest_file = hevc_rknn.manifest_path(model_path)
    cases = _cases(FRAMES, _read_reference_audio(reference_audio))

    prep_report = prepare(source, prepared_path, FRAMES)
    equivalence = compare_ort(source, prepared_path, cases)
    prepared_snr = {name: value["snr_db"] for name, value in equivalence.items() if name != "silence"}
    if any(value < SNR_GATE_DB for value in prepared_snr.values()):
        raise SystemExit(f"rewritten graph failed the {SNR_GATE_DB:.0f} dB gate: {prepared_snr}")
    if any(value["max_abs"] > EQUIVALENCE_GATE for value in equivalence.values()):
        raise SystemExit(f"rewritten graph failed the waveform gate: {equivalence}")

    simulator, toolkit = build_and_compare(prepared_path, model_path, cases)
    # The square-wave boundary case has extreme high-frequency content; it is
    # gated on absolute error, while speech and noise are gated on SNR.
    simulator_snr = {name: simulator[name]["snr_db"] for name in SPEECH_CASES}
    worst = min(simulator_snr.values())
    worst_abs = max(value["max_abs"] for value in simulator.values())
    if worst < SNR_GATE_DB or worst_abs > SIMULATOR_MAX_ABS:
        model_path.unlink(missing_ok=True)
        raise SystemExit(
            f"RKNN simulator accuracy failed (worst {worst:.1f} dB, max |err| {worst_abs:.2e}): {simulator}"
        )

    manifest = {
        "format": hevc_rknn.MANIFEST_FORMAT,
        "entry_id": entry_id,
        "toolkit_version": (
            toolkit["toolkit"]
            if toolkit["toolkit"] != "unknown"
            else hevc_rknn.TOOLKIT_VERSION
        ),
        "toolkit_reported": toolkit["toolkit"],
        "target_platform": hevc_rknn.TARGET_PLATFORM,
        "source_onnx": source.name,
        "source_onnx_sha256": digest,
        "model_sha256": _sha256(model_path),
        "frames": FRAMES,
        "history": HISTORY,
        "chunk": CHUNK,
        "sample_rate": hevc_rknn.SAMPLE_RATE,
        "conversion_frames": hevc_rknn.CONVERSION_FRAMES,
        "bit_depth": "fp16",
        "quantized": False,
        "simulator_snr_db": round(worst, 2),
        "simulator_max_abs": round(worst_abs, 8),
        "simulator_cases": {
            name: {"max_abs": round(value["max_abs"], 8), "snr_db": round(value["snr_db"], 2)}
            for name, value in simulator.items()
        },
        "onnx_equivalence_cases": {
            name: {"max_abs": round(value["max_abs"], 10), "snr_db": round(value["snr_db"], 2)}
            for name, value in equivalence.items()
        },
        "graph": prep_report,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": (
            "Public HEEVoicechanger weights (GPL-3.0) converted locally; not part of the repository. "
            "Accuracy is measured against the rewritten ONNX graph on the host simulator, not against "
            "the original voice recording. Live latency is measured separately on the board."
        ),
    }
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "entry_id": entry_id,
        "model": str(model_path),
        "manifest": str(manifest_file),
        "model_bytes": model_path.stat().st_size,
        "prepared": prep_report,
        "onnx_equivalence": equivalence,
        "simulator": simulator,
        "seconds": round(time.time() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", type=Path, help="HEEVoicechanger ONNX file(s)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "rknn-models")
    parser.add_argument("--reference-audio", type=Path, default=None,
                        help="optional speech WAV used as one accuracy case")
    parser.add_argument("--list", action="store_true", help="list known source models and exit")
    args = parser.parse_args(argv)
    if args.list:
        for entry in ENTRIES:
            if entry.model_type == "hevc_onnx":
                print(f"{entry.entry_id:12s} {entry.filenames[0]:24s} {entry.sha256[0]}")
        return 0
    failed = 0
    for model in args.models:
        if not model.is_file():
            print(f"{model}: 文件不存在 / not found", file=sys.stderr)
            failed += 1
            continue
        try:
            report = convert(model, args.output_dir, args.reference_audio)
        except SystemExit as error:
            print(str(error), file=sys.stderr)
            failed += 1
            continue
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
