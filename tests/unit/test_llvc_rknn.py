import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from voice_changer.ai import llvc
from voice_changer.audio.device_types import DeviceRef, StreamConfig


def stream_config():
    device = DeviceRef(0, "test", 0)
    return StreamConfig(device, device, 48000, 2496, 1, 0.5)


def valid_host_simulator_equivalence():
    blocks = len(llvc.RKNN_SIMULATOR_CASE_NAMES) * llvc.RKNN_SIMULATOR_BLOCKS_PER_CASE
    return {
        "passed": True,
        "rtol": llvc.RKNN_SIMULATOR_RTOL,
        "atol": llvc.RKNN_SIMULATOR_ATOL,
        "continuous_blocks_compared": blocks,
        "output_tensors_compared": blocks * len(llvc.OUTPUTS),
        "cases": list(llvc.RKNN_SIMULATOR_CASE_NAMES),
        "max_abs_error": dict.fromkeys(llvc.OUTPUTS, 0.0),
    }


class FakeRknnLite:
    NPU_CORE_0 = 1
    instances = []
    load_status = 0
    init_status = 0

    def __init__(self):
        self.calls = 0
        self.released = 0
        self.state_seen = []
        self.core_mask = None
        type(self).instances.append(self)

    def load_rknn(self, path):
        self.model_path = path
        return type(self).load_status

    def init_runtime(self, *, core_mask):
        self.core_mask = core_mask
        return type(self).init_status

    def inference(self, *, inputs):
        self.calls += 1
        if self.calls <= 2:
            self.state_seen.append([value.copy() for value in inputs[1:]])
        return [
            np.full((1, 1, 1664), 0.1, dtype=np.float32),
            *[value + np.float32(1) for value in inputs[1:]],
        ]

    def release(self):
        self.released += 1


class LlvcRknnTests(unittest.TestCase):
    def setUp(self):
        FakeRknnLite.instances.clear()
        FakeRknnLite.load_status = 0
        FakeRknnLite.init_status = 0
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.model = self.directory / "llvc.rknn"
        self.model.write_bytes(b"test-rknn-model")
        self.onnx_hash = llvc.STABLE_MODEL_SHA256
        self.write_manifest()
        package = types.ModuleType("rknnlite")
        package.__path__ = []
        api = types.ModuleType("rknnlite.api")
        api.RKNNLite = FakeRknnLite
        self.rknn_patch = patch.dict(sys.modules, {"rknnlite": package, "rknnlite.api": api})
        self.rknn_patch.start()
        self.addCleanup(self.rknn_patch.stop)

    def write_manifest(self, **updates):
        manifest = {
            "format": llvc.RKNN_MANIFEST_FORMAT,
            "source_sha256": self.onnx_hash,
            "target_platform": "rk3576",
            "quantized": False,
            "model_sha256": hashlib.sha256(self.model.read_bytes()).hexdigest(),
            "inputs": list(llvc.INPUTS),
            "outputs": list(llvc.OUTPUTS),
            "input_shapes": [list(shape) for shape in llvc.RKNN_INPUT_SHAPES],
            "output_shapes": [list(shape) for shape in llvc.RKNN_OUTPUT_SHAPES],
            "host_simulator_equivalence": valid_host_simulator_equivalence(),
        }
        manifest.update(updates)
        Path(f"{self.model}.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_converter_keeps_all_state_tensors_and_releases_runtime(self):
        with patch.dict(sys.modules, {"onnxruntime": None}):
            converter = llvc.LlvcConverter(stream_config(), self.model)
        audio = np.full((4992, 1), 0.1, dtype=np.float32)
        for _ in range(3):
            output = converter(audio)
            self.assertEqual(output.shape, audio.shape)
            self.assertEqual(output.dtype, np.float32)
            self.assertTrue(np.isfinite(output).all())

        runtime = FakeRknnLite.instances[-1]
        self.assertEqual(runtime.calls, 2)
        self.assertEqual(runtime.core_mask, FakeRknnLite.NPU_CORE_0)
        self.assertEqual(len(runtime.state_seen[0]), 4)
        for initial, following in zip(runtime.state_seen[0], runtime.state_seen[1]):
            np.testing.assert_array_equal(initial, np.zeros(initial.shape, dtype=np.float32))
            np.testing.assert_array_equal(following, np.ones(following.shape, dtype=np.float32))
        converter.close()
        converter.close()
        self.assertEqual(runtime.released, 1)

    def test_manifest_binds_output_to_validated_unquantized_rk3576_source(self):
        self.write_manifest(model_sha256="wrong")
        with self.assertRaisesRegex(ValueError, "model contract"):
            llvc.LlvcConverter(stream_config(), self.model)
        self.assertEqual(FakeRknnLite.instances, [])

    def test_manifest_requires_passing_host_simulator_equivalence(self):
        self.write_manifest(host_simulator_equivalence={"passed": False})
        with self.assertRaisesRegex(ValueError, "host-simulator equivalence"):
            llvc.LlvcConverter(stream_config(), self.model)
        self.assertEqual(FakeRknnLite.instances, [])

    def test_runtime_initialization_failure_releases_loaded_artifact(self):
        FakeRknnLite.init_status = -1
        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            llvc.LlvcConverter(stream_config(), self.model)
        self.assertEqual(FakeRknnLite.instances[-1].released, 1)


class LlvcRknnConversionTests(unittest.TestCase):
    def test_host_simulator_gate_checks_outputs_and_carries_state(self):
        script = Path(__file__).resolve().parents[2] / "scripts" / "convert_llvc_rknn.py"
        spec = importlib.util.spec_from_file_location("convert_llvc_rknn_simulator", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class FakeSession:
            def get_inputs(self):
                return [types.SimpleNamespace(name=name) for name in module.INPUTS]

            def get_outputs(self):
                return [types.SimpleNamespace(name=name) for name in module.OUTPUTS]

            def run(self, names, _feed):
                values = {
                    name: np.zeros(shape, dtype=np.float32)
                    if name == "converted"
                    else np.ones(shape, dtype=np.float32)
                    for name, shape in zip(module.OUTPUTS, llvc.RKNN_OUTPUT_SHAPES)
                }
                return [values[name] for name in names]

        class FakeRuntime:
            def __init__(self, *, mismatch=False):
                self.calls = []
                self.mismatch = mismatch

            def init_runtime(self, *, target):
                self.target = target
                return 0

            def inference(self, *, inputs, data_format):
                self.calls.append(([value.copy() for value in inputs], data_format))
                values = [
                    np.zeros(llvc.RKNN_OUTPUT_SHAPES[0], dtype=np.float32),
                    *[
                        np.ones(shape, dtype=np.float32)
                        for shape in llvc.RKNN_OUTPUT_SHAPES[1:]
                    ],
                ]
                if self.mismatch and len(self.calls) == 1:
                    values[1] = np.full(
                        llvc.RKNN_OUTPUT_SHAPES[1], 2.0, dtype=np.float32,
                    )
                return values

        ort_stub = types.SimpleNamespace(
            SessionOptions=type("SessionOptions", (), {}),
            ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=0),
            InferenceSession=lambda *_args, **_kwargs: FakeSession(),
        )
        runtime = FakeRuntime()
        with patch.dict(sys.modules, {"onnxruntime": ort_stub}):
            result = module.verify_rknn_simulator(runtime, Path("prepared.onnx"))

        self.assertIsNone(runtime.target)
        self.assertEqual(len(runtime.calls), 18)
        self.assertEqual(runtime.calls[0][1], ["nhwc", "nhwc", "nchw", "nhwc", "nhwc"])
        for state in runtime.calls[1][0][1:]:
            np.testing.assert_array_equal(state, np.ones(state.shape, dtype=np.float32))
        self.assertTrue(result["passed"])
        self.assertEqual(result["continuous_blocks_compared"], 18)
        self.assertEqual(result["output_tensors_compared"], 90)
        self.assertEqual(result["max_abs_error"], dict.fromkeys(module.OUTPUTS, 0.0))
        bad_runtime = FakeRuntime(mismatch=True)
        with patch.dict(sys.modules, {"onnxruntime": ort_stub}):
            with self.assertRaisesRegex(
                RuntimeError, "next_enc in zero_state_silence block 0",
            ):
                module.verify_rknn_simulator(bad_runtime, Path("prepared.onnx"))
        self.assertEqual(len(bad_runtime.calls), 1)

    def test_conversion_is_fixed_hash_nonquantized_and_refuses_overwrite(self):
        script = Path(__file__).resolve().parents[2] / "scripts" / "convert_llvc_rknn.py"
        spec = importlib.util.spec_from_file_location("convert_llvc_rknn", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "llvc.onnx"
            source.write_bytes(b"validated-onnx")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            source_patch = patch.object(module, "STABLE_MODEL_SHA256", source_hash)
            source_patch.start()
            self.addCleanup(source_patch.stop)

            class FakeRknn:
                options = None
                quantization = None
                loaded_model = None

                def __init__(self, **_kwargs):
                    pass

                def config(self, **kwargs):
                    type(self).options = kwargs
                    return 0

                def load_onnx(self, **kwargs):
                    type(self).loaded_model = Path(kwargs["model"]).read_bytes()
                    return 0

                def build(self, **kwargs):
                    type(self).quantization = kwargs
                    Path("check0_base_optimize.onnx").write_bytes(b"diagnostic")
                    return 0

                def export_rknn(self, output):
                    Path(output).write_bytes(b"compiled-rknn")
                    return 0

                def release(self):
                    pass

            root = types.ModuleType("rknn")
            root.__path__ = []
            api = types.ModuleType("rknn.api")
            api.RKNN = FakeRknn

            def prepare_graph(_source, destination):
                destination.write_bytes(b"prepared-onnx")
                return {
                    "transform": "test-transform",
                    "source_sha256": source_hash,
                    "graph_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "equivalence": {"passed": True},
                }

            with patch.dict(sys.modules, {"rknn": root, "rknn.api": api}), patch.object(
                module.importlib.metadata, "version", return_value="2.5.0",
            ), patch.object(module, "prepare_llvc_rknn_graph", side_effect=prepare_graph), patch.object(
                module,
                "verify_rknn_simulator",
                return_value={
                    "passed": True,
                    "rtol": 0.001,
                    "atol": 0.0001,
                    "continuous_blocks_compared": 18,
                },
            ) as simulator_validation:
                destination = directory / "converted.rknn"
                original_cwd = Path.cwd()
                manifest_path = module.convert(source, destination)
                simulator_validation.side_effect = RuntimeError(
                    "RKNN host simulator equivalence failed for next_enc"
                )
                rejected_destination = directory / "rejected.rknn"
                with self.assertRaisesRegex(RuntimeError, "equivalence failed"):
                    module.convert(source, rejected_destination)
                self.assertFalse(rejected_destination.exists())
                self.assertFalse(Path(f"{rejected_destination}.json").exists())

            self.assertEqual(Path.cwd(), original_cwd)
            self.assertEqual(FakeRknn.options, {"target_platform": "rk3576"})
            self.assertEqual(FakeRknn.quantization, {"do_quantization": False})
            self.assertEqual(FakeRknn.loaded_model, b"prepared-onnx")
            self.assertEqual(simulator_validation.call_count, 2)
            self.assertFalse((directory / "check0_base_optimize.onnx").exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_sha256"], source_hash)
            self.assertEqual(manifest["target_platform"], "rk3576")
            self.assertFalse(manifest["quantized"])
            self.assertEqual(manifest["inputs"], list(module.INPUTS))
            self.assertEqual(manifest["outputs"], list(module.OUTPUTS))
            self.assertEqual(
                manifest["conversion_graph_sha256"],
                hashlib.sha256(b"prepared-onnx").hexdigest(),
            )
            self.assertEqual(manifest["graph_preparation"]["transform"], "test-transform")
            self.assertTrue(manifest["host_simulator_equivalence"]["passed"])
            with self.assertRaises(FileExistsError):
                module.convert(source, destination)

    def test_worker_smoke_uses_memory_and_forces_silence(self):
        script = Path(__file__).resolve().parents[2] / "scripts" / "bench_llvc_rknn.py"
        spec = importlib.util.spec_from_file_location("bench_llvc_rknn", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        FakeRknnLite.instances.clear()
        FakeRknnLite.load_status = 0
        FakeRknnLite.init_status = 0
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "smoke.rknn"
            model.write_bytes(b"fake-model")
            manifest = {
                "format": llvc.RKNN_MANIFEST_FORMAT,
                "source_sha256": llvc.STABLE_MODEL_SHA256,
                "target_platform": "rk3576",
                "quantized": False,
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "inputs": list(llvc.INPUTS),
                "outputs": list(llvc.OUTPUTS),
                "input_shapes": [list(shape) for shape in llvc.RKNN_INPUT_SHAPES],
                "output_shapes": [list(shape) for shape in llvc.RKNN_OUTPUT_SHAPES],
                "host_simulator_equivalence": valid_host_simulator_equivalence(),
            }
            Path(f"{model}.json").write_text(json.dumps(manifest), encoding="utf-8")
            package = types.ModuleType("rknnlite")
            package.__path__ = []
            api = types.ModuleType("rknnlite.api")
            api.RKNNLite = FakeRknnLite
            with patch.dict(sys.modules, {"rknnlite": package, "rknnlite.api": api}):
                result = module.run_worker_smoke(model, conversion_blocks=2)
        self.assertEqual(result["conversion_blocks"], 2)
        self.assertTrue(result["output_forced_silent"])
        self.assertFalse(result["audio_devices_opened"])
        self.assertFalse(result["audio_saved"])


if __name__ == "__main__":
    unittest.main()
