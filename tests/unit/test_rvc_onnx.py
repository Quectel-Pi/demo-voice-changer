from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from voice_changer.ai.rvc_onnx import RvcOnnxSynthesizer


@dataclass
class TensorInfo:
    name: str
    type: str
    shape: list[int | str | None]


class FakeSession:
    def __init__(self, *, channels=768, half=False, flat=False):
        self.inputs = [
            TensorInfo("feats", "tensor(float16)" if half else "tensor(float)", [1, "T", channels]),
            TensorInfo("p_len", "tensor(int64)", [1]),
            TensorInfo("pitch", "tensor(int64)", [1, "T"]),
            TensorInfo("pitchf", "tensor(float)", [1, "T"]),
            TensorInfo("sid", "tensor(int64)", [1]),
        ]
        self.outputs = [TensorInfo("audio", "tensor(float)", ["N"] if flat else [1, 1, "N"])]
        self.providers = ["CPUExecutionProvider"]
        self.result = [np.array([-2, 0.25, 2], dtype=np.float32)]
        if not flat:
            self.result[0] = self.result[0][None, None]
        self.calls = []

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_providers(self):
        return self.providers

    def run(self, output_names, input_feed):
        self.calls.append((output_names, input_feed))
        return self.result


class RvcOnnxTests(unittest.TestCase):
    def inputs(self, channels=768):
        return (
            np.ones((4, channels), dtype=np.float32),
            np.array([1, 20, 100, 255], dtype=np.int32),
            np.array([0, 80, 120, 400], dtype=np.float32),
        )

    def test_synthesis_feeds_exact_interface_without_changing_caller_arrays(self):
        for channels in (256, 768):
            for half in (False, True):
                for flat in (False, True):
                    with self.subTest(channels=channels, half=half, flat=flat):
                        session = FakeSession(channels=channels, half=half, flat=flat)
                        arrays = self.inputs(channels)
                        before = [value.copy() for value in arrays]
                        audio = RvcOnnxSynthesizer(session).synthesize(*arrays, speaker_id=2)
                        names, feed = session.calls[0]
                        self.assertEqual(names, ["audio"])
                        self.assertEqual(set(feed), {"feats", "p_len", "pitch", "pitchf", "sid"})
                        self.assertEqual(feed["feats"].shape, (1, 4, channels))
                        self.assertEqual(feed["feats"].dtype, np.float16 if half else np.float32)
                        for name in ("p_len", "pitch", "sid"):
                            self.assertEqual(feed[name].dtype, np.int64)
                        self.assertEqual(feed["pitchf"].dtype, np.float32)
                        self.assertEqual(feed["p_len"].tolist(), [4])
                        self.assertEqual(feed["sid"].tolist(), [2])
                        np.testing.assert_array_equal(feed["pitch"][0], arrays[1])
                        np.testing.assert_array_equal(feed["pitchf"][0], arrays[2])
                        self.assertTrue(all(value.flags.c_contiguous for value in feed.values()))
                        np.testing.assert_array_equal(audio, [-1, 0.25, 1])
                        self.assertEqual(audio.dtype, np.float32)
                        self.assertFalse(np.shares_memory(audio, session.result[0]))
                        for previous, value in zip(before, arrays):
                            np.testing.assert_array_equal(previous, value)

    def test_invalid_inputs_never_call_runtime(self):
        features, pitch, hz = self.inputs()
        cases = [
            (features.tolist(), pitch, hz),
            (features[0], pitch, hz),
            (features[:0], pitch[:0], hz[:0]),
            (features[:, :20], pitch, hz),
            (features.astype(np.float64), pitch, hz),
            (features * np.nan, pitch, hz),
            (features, pitch[:-1], hz),
            (features, pitch.astype(np.float32), hz),
            (features, pitch * 0, hz),
            (features, pitch + 1, hz),
            (features, pitch, hz[:, None]),
            (features, pitch, np.full_like(hz, np.inf)),
            (features, pitch, hz - 1),
            (features, pitch, hz.astype(np.int64)),
        ]
        for values in cases:
            with self.subTest(shapes=[np.shape(value) for value in values]):
                session = FakeSession()
                with self.assertRaises(ValueError):
                    RvcOnnxSynthesizer(session).synthesize(*values)
                self.assertEqual(session.calls, [])
        for speaker in (-1, True, 0.5, 2**63):
            session = FakeSession()
            with self.subTest(speaker=speaker), self.assertRaises(ValueError):
                RvcOnnxSynthesizer(session).synthesize(features, pitch, hz, speaker_id=speaker)
            self.assertEqual(session.calls, [])

    def test_static_shapes_and_half_overflow_are_checked_before_inference(self):
        for name, dimension in (("feats", 1), ("pitch", 1), ("pitchf", 1)):
            session = FakeSession()
            next(item for item in session.inputs if item.name == name).shape[dimension] = 8
            with self.subTest(name=name), self.assertRaises(ValueError):
                RvcOnnxSynthesizer(session).synthesize(*self.inputs())
            self.assertEqual(session.calls, [])
        session = FakeSession(channels=256)
        with self.assertRaises(ValueError):
            RvcOnnxSynthesizer(session).synthesize(*self.inputs())
        self.assertEqual(session.calls, [])
        features, pitch, hz = self.inputs()
        session = FakeSession(half=True)
        with self.assertRaisesRegex(ValueError, "range"):
            RvcOnnxSynthesizer(session).synthesize(features * 70000, pitch, hz)
        self.assertEqual(session.calls, [])

    def test_unsupported_models_fail_during_construction(self):
        def extra_input(session):
            session.inputs.append(TensorInfo("rnd", "tensor(float)", [1]))

        def duplicate_input(session):
            session.inputs[-1] = session.inputs[0]

        changes = [
            lambda s: s.providers.append("CUDAExecutionProvider"),
            lambda s: setattr(s.inputs[0], "type", "tensor(double)"),
            lambda s: setattr(s.inputs[0], "shape", [1, "T", 10]),
            lambda s: setattr(s.inputs[0], "shape", [2, "T", 768]),
            lambda s: setattr(s.inputs[2], "type", "tensor(float)"),
            lambda s: setattr(s.inputs[3], "shape", [1]),
            lambda s: setattr(s.outputs[0], "name", "waveform"),
            lambda s: setattr(s.outputs[0], "type", "tensor(int64)"),
            lambda s: setattr(s.outputs[0], "shape", [1, 2, "N"]),
            lambda s: setattr(s.outputs[0], "shape", [1, "N"]),
            extra_input,
            duplicate_input,
        ]
        for change in changes:
            session = FakeSession()
            change(session)
            with self.subTest(change=change), self.assertRaises(ValueError):
                RvcOnnxSynthesizer(session)
            self.assertEqual(session.calls, [])

    def test_invalid_outputs_are_explicit_errors(self):
        results = [
            [],
            [None],
            [np.zeros((1, 2, 3), dtype=np.float32)],
            [np.zeros((1, 1, 0), dtype=np.float32)],
            [np.zeros(3, dtype=np.float32)],
            [np.full((1, 1, 3), np.nan, dtype=np.float32)],
            [np.full((1, 1, 3), np.inf, dtype=np.float32)],
            [np.zeros((1, 1, 3), dtype=np.int64)],
            [np.zeros((1, 1, 3), dtype=np.float16)],
        ]
        for result in results:
            session = FakeSession()
            session.result = result
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                RvcOnnxSynthesizer(session).synthesize(*self.inputs())
        session = FakeSession()
        session.outputs[0].shape[-1] = 5
        with self.assertRaisesRegex(RuntimeError, "length"):
            RvcOnnxSynthesizer(session).synthesize(*self.inputs())

    def test_half_audio_and_noncontiguous_features_are_supported(self):
        session = FakeSession()
        session.outputs[0].type = "tensor(float16)"
        session.result[0] = session.result[0].astype(np.float16)
        features, pitch, hz = self.inputs()
        features = features[:, ::-1]
        output = RvcOnnxSynthesizer(session).synthesize(features, pitch, hz)
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(session.calls[0][1]["feats"].flags.c_contiguous)

    def test_runtime_failures_are_not_replaced_by_passthrough(self):
        session = FakeSession()
        with patch.object(session, "run", side_effect=RuntimeError("inference failed")):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                RvcOnnxSynthesizer(session).synthesize(*self.inputs())

    def test_explicit_loader_is_cpu_only_and_does_not_register_a_backend(self):
        from voice_changer.ai.availability import availability

        session = FakeSession()
        runtime = SimpleNamespace(
            SessionOptions=lambda: SimpleNamespace(),
            InferenceSession=Mock(return_value=session),
        )
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "fixture.onnx"
            model.write_bytes(b"test placeholder, not a model")
            with patch.dict(sys.modules, {"onnxruntime": runtime}):
                RvcOnnxSynthesizer.load(model, threads=2)
                call = runtime.InferenceSession.call_args
                self.assertEqual(call.args, (str(model),))
                self.assertEqual(call.kwargs["providers"], ["CPUExecutionProvider"])
                self.assertEqual(call.kwargs["sess_options"].intra_op_num_threads, 2)
                self.assertEqual(call.kwargs["sess_options"].inter_op_num_threads, 1)
                for threads in (0, -1, True, 1.5):
                    with self.subTest(threads=threads), self.assertRaises(ValueError):
                        RvcOnnxSynthesizer.load(model, threads=threads)
                for invalid in (model.with_suffix(".pth"), Path(directory)):
                    with self.assertRaises(ValueError):
                        RvcOnnxSynthesizer.load(invalid)
                link = Path(directory) / "link.onnx"
                link.symlink_to(model)
                with self.assertRaises(ValueError):
                    RvcOnnxSynthesizer.load(link)
                self.assertEqual(runtime.InferenceSession.call_count, 1)
        self.assertFalse(availability("vc").available)

    def test_import_never_loads_onnxruntime_torch_or_qt(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             "import sys; import voice_changer.ai.rvc_onnx; "
             "import voice_changer.ai.rvc_frontend; import voice_changer.ai.resampling; "
             "import voice_changer.ai.rvc_rknn; "
             "assert not any(n.split('.')[0] in ('onnxruntime', 'torch', 'PySide6') "
             "for n in sys.modules); "
             "assert 'rknnlite' not in sys.modules"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
