#!/usr/bin/env python3
"""Convert one fixed-frame RVC ONNX synthesizer to a non-quantized RK3576 model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_changer.ai.rvc_rknn import (
    RVC_RKNN_MANIFEST_FORMAT,
    RVC_SYNTHESIS_INPUTS,
    RVC_SYNTHESIS_OUTPUT,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dimension_value(dimension) -> int | None:
    return dimension.dim_value if dimension.dim_value > 0 else None


def inspect_model(
    source: Path,
    frames: int,
    model_sample_rate: int,
) -> tuple[int, list[list[int]], str]:
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(source))
    onnx.checker.check_model(model)
    graph_inputs = {item.name: item for item in model.graph.input}
    initializers = {item.name for item in model.graph.initializer}
    graph_inputs = {name: item for name, item in graph_inputs.items() if name not in initializers}
    if tuple(graph_inputs) != RVC_SYNTHESIS_INPUTS:
        raise ValueError(
            "RVC ONNX inputs must be ordered feats, p_len, pitch, pitchf and sid"
        )
    expected_types = {
        "feats": (TensorProto.FLOAT, TensorProto.FLOAT16),
        "p_len": (TensorProto.INT64,),
        "pitch": (TensorProto.INT64,),
        "pitchf": (TensorProto.FLOAT,),
        "sid": (TensorProto.INT64,),
    }
    channels = None
    shapes: list[list[int]] = []
    for name in RVC_SYNTHESIS_INPUTS:
        value = graph_inputs[name]
        tensor = value.type.tensor_type
        dimensions = [_dimension_value(item) for item in tensor.shape.dim]
        if tensor.elem_type not in expected_types[name]:
            raise ValueError(f"Unsupported RVC ONNX input dtype for {name}")
        if name == "feats":
            if len(dimensions) != 3 or dimensions[0] not in (None, 1):
                raise ValueError("RVC feats must have shape [1, frames, channels]")
            if dimensions[1] not in (None, frames) or dimensions[2] not in (256, 768):
                raise ValueError("RVC feats has incompatible frame or feature dimensions")
            channels = dimensions[2]
            shape = [1, frames, channels]
        elif name in ("pitch", "pitchf"):
            if len(dimensions) != 2 or dimensions[0] not in (None, 1):
                raise ValueError(f"RVC {name} must have shape [1, frames]")
            if dimensions[1] not in (None, frames):
                raise ValueError(f"RVC {name} frame count differs from --frames")
            shape = [1, frames]
        else:
            if dimensions != [None] and dimensions != [1]:
                raise ValueError(f"RVC {name} must have shape [1]")
            shape = [1]
        shapes.append(shape)

    outputs = {item.name: item for item in model.graph.output}
    if len(outputs) != 1 or RVC_SYNTHESIS_OUTPUT not in outputs:
        raise ValueError("RVC ONNX must expose exactly one output named audio")
    audio = outputs[RVC_SYNTHESIS_OUTPUT].type.tensor_type
    if audio.elem_type not in (TensorProto.FLOAT, TensorProto.FLOAT16):
        raise ValueError("RVC audio output must be float32 or float16")
    output_dims = [_dimension_value(item) for item in audio.shape.dim]
    samples = frames * model_sample_rate // 100
    if len(output_dims) == 3:
        if (
            any(dimension not in (None, 1) for dimension in output_dims[:2])
            or output_dims[2] not in (None, samples)
        ):
            raise ValueError("RVC audio output must have shape [1, 1, frames * sample_rate / 100]")
    elif len(output_dims) == 1:
        if output_dims[0] not in (None, samples):
            raise ValueError("RVC audio output length differs from the selected fixed frame count")
    else:
        raise ValueError("RVC audio output must have shape [N] or [1, 1, N]")
    feature_dtype = "float16" if graph_inputs["feats"].type.tensor_type.elem_type == TensorProto.FLOAT16 else "float32"
    return channels, shapes, feature_dtype


def convert(
    source: Path,
    destination: Path,
    *,
    frames: int,
    model_sample_rate: int,
) -> Path:
    source = source.expanduser()
    destination = destination.expanduser()
    if source.is_symlink() or not source.is_file() or source.suffix.lower() != ".onnx":
        raise ValueError("Source must be an existing regular RVC .onnx file")
    if destination.is_symlink():
        raise ValueError("Destination must not be a symbolic link")
    source = source.resolve()
    destination = destination.resolve()
    if destination.suffix.lower() != ".rknn":
        raise ValueError("Destination must use the .rknn extension")
    if type(frames) is not int or frames < 1 or frames > 1000:
        raise ValueError("frames must be between 1 and 1000")
    if type(model_sample_rate) is not int or model_sample_rate not in (32000, 40000, 48000):
        raise ValueError("RVC model sample rate must be 32000, 40000 or 48000 Hz")
    if not destination.parent.is_dir():
        raise ValueError(f"Destination directory does not exist: {destination.parent}")
    manifest_path = Path(f"{destination}.json")
    if destination.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing RVC RKNN artifact: {destination}")

    channels, input_shapes, feature_dtype = inspect_model(source, frames, model_sample_rate)
    from rknn.api import RKNN

    with tempfile.TemporaryDirectory(prefix=".rvc-rknn-", dir=destination.parent) as temporary:
        temporary_model = Path(temporary) / destination.name
        runtime = RKNN(verbose=True)
        previous_cwd = Path.cwd()
        try:
            os.chdir(temporary)
            runtime.config(
                target_platform="rk3576",
                dynamic_input=[input_shapes],
            )
            loaded = runtime.load_onnx(model=str(source))
            if loaded != 0:
                raise RuntimeError(f"RKNN ONNX load failed (status {loaded})")
            built = runtime.build(do_quantization=False)
            if built != 0:
                raise RuntimeError(f"RKNN build failed (status {built})")
            exported = runtime.export_rknn(str(temporary_model))
            if exported != 0 or not temporary_model.is_file():
                raise RuntimeError(f"RKNN export failed (status {exported})")
        finally:
            try:
                runtime.release()
            finally:
                os.chdir(previous_cwd)

        manifest = {
            "format": RVC_RKNN_MANIFEST_FORMAT,
            "source_sha256": sha256_file(source),
            "target_platform": "rk3576",
            "quantized": False,
            "model_sha256": sha256_file(temporary_model),
            "inputs": list(RVC_SYNTHESIS_INPUTS),
            "outputs": [RVC_SYNTHESIS_OUTPUT],
            "input_shapes": input_shapes,
            "output_shapes": [[1, 1, frames * model_sample_rate // 100]],
            "frames": frames,
            "feature_channels": channels,
            "feature_dtype": feature_dtype,
            "model_sample_rate": model_sample_rate,
            "toolkit_version": importlib.metadata.version("rknn-toolkit2"),
            "numerical_equivalence_verified": False,
        }
        temporary_manifest = Path(temporary) / f"{destination.name}.json"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists() or manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing RVC RKNN artifact: {destination}")
        os.link(temporary_model, destination)
        try:
            os.link(temporary_manifest, manifest_path)
        except OSError:
            destination.unlink(missing_ok=True)
            raise
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="trusted RVC synthesis ONNX model")
    parser.add_argument("destination", type=Path, help="new RK3576 .rknn output path")
    parser.add_argument("--frames", type=int, required=True, help="fixed 100 Hz feature/F0 frame count")
    parser.add_argument(
        "--model-sample-rate", type=int, choices=(32000, 40000, 48000), required=True,
    )
    arguments = parser.parse_args()
    manifest = convert(
        arguments.source,
        arguments.destination,
        frames=arguments.frames,
        model_sample_rate=arguments.model_sample_rate,
    )
    print(f"RVC RKNN model and manifest created: {manifest}")
    print("Inspect conversion logs and verify against ONNX before using the candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
