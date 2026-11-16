import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from voice_changer.ai.model_inspection import inspect_model_file


class Tensor:
    def __init__(self, name, kind, shape):
        self.name = name
        self.type = kind
        self.shape = shape


class Session:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_providers(self):
        return ["CPUExecutionProvider"]


class ModelInspectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _inspect_onnx(self, inputs, outputs):
        path = self.root / "candidate.onnx"
        path.write_bytes(b"onnx fixture")
        session = Session(inputs, outputs)
        with patch("voice_changer.ai.llvc._onnx_session", return_value=session):
            return inspect_model_file(path)

    def test_llvc_onnx_is_detected_from_its_fixed_audio_and_state_schema(self):
        input_shapes = (
            (1, 1, 864),
            (1, 512, 510),
            (1, 2, 13, 256),
            (1, 512, 4),
            (1, 1, 24),
        )
        inputs = [
            Tensor(name, "tensor(float)", shape)
            for name, shape in zip(("audio", "enc", "dec", "out", "pre"), input_shapes)
        ]
        outputs = [
            Tensor(name, "tensor(float)", shape)
            for name, shape in zip(
                ("converted", "next_enc", "next_dec", "next_out", "next_pre"),
                ((1, 1, 832), *input_shapes[1:]),
            )
        ]
        result = self._inspect_onnx(inputs, outputs)
        self.assertEqual((result.model_type, result.frames), ("llvc_onnx", 832))

    def test_hevc_onnx_is_detected_by_its_causal_audio_contract(self):
        result = self._inspect_onnx(
            [Tensor("audio_in", "tensor(float)", [1, 1, "samples"])],
            [Tensor("audio_out", "tensor(float)", ["samples", 1, "samples"])],
        )
        self.assertEqual(result.model_type, "hevc_onnx")
        self.assertEqual(result.sha256, hashlib.sha256(b"onnx fixture").hexdigest())
        self.assertIn("not accepted", result.description)

        dynamic_batch = self._inspect_onnx(
            [Tensor("audio_in", "tensor(float)", ["batch_size", 1, "samples"])],
            [Tensor("audio_out", "tensor(float)", ["batch_size", 1, "samples"])],
        )
        self.assertEqual(dynamic_batch.model_type, "hevc_onnx")

    def test_hevc_like_graphs_with_a_wrong_interface_are_not_accepted(self):
        for inputs, outputs in (
            (
                [Tensor("audio", "tensor(float)", [1, 1, "samples"])],
                [Tensor("audio_out", "tensor(float)", ["samples", 1, "samples"])],
            ),
            (
                [Tensor("audio_in", "tensor(float16)", [1, 1, "samples"])],
                [Tensor("audio_out", "tensor(float16)", ["samples", 1, "samples"])],
            ),
            (
                [
                    Tensor("audio_in", "tensor(float)", [1, 1, "samples"]),
                    Tensor("extra", "tensor(float)", [1, 1]),
                ],
                [Tensor("audio_out", "tensor(float)", ["samples", 1, "samples"])],
            ),
        ):
            with self.assertRaisesRegex(ValueError, "does not match a supported"):
                self._inspect_onnx(inputs, outputs)

    def test_rvc_synthesis_onnx_is_detected_by_tensor_contract(self):
        inputs = [
            Tensor("feats", "tensor(float)", [1, "frames", 768]),
            Tensor("p_len", "tensor(int64)", [1]),
            Tensor("pitch", "tensor(int64)", [1, "frames"]),
            Tensor("pitchf", "tensor(float)", [1, "frames"]),
            Tensor("sid", "tensor(int64)", [1]),
        ]
        result = self._inspect_onnx(
            inputs, [Tensor("audio", "tensor(float)", [1, 1, "samples"])],
        )
        self.assertEqual(result.model_type, "rvc_onnx_synthesis")

    def test_rvc_frontend_components_are_detected_but_not_treated_as_full_models(self):
        content = self._inspect_onnx(
            [Tensor("source", "tensor(float)", [1, "samples"])],
            [Tensor("hidden", "tensor(float)", [1, "frames", 768])],
        )
        self.assertEqual(content.model_type, "rvc_onnx_content")

        pitch = self._inspect_onnx(
            [
                Tensor("waveform", "tensor(float)", [1, "samples"]),
                Tensor("threshold", "tensor(float)", [1]),
            ],
            [Tensor("pitchf", "tensor(float)", [1, "frames"])],
        )
        self.assertEqual(pitch.model_type, "rvc_onnx_pitch")

    def test_pth_is_identified_without_importing_or_loading_torch(self):
        path = self.root / "rvc.pth"
        path.write_bytes(b"not a serialized checkpoint")
        with patch.dict("sys.modules", {"torch": None}):
            result = inspect_model_file(path)
        self.assertEqual(result.model_type, "pytorch_checkpoint")
        self.assertIn("RVC format is unverified", result.description)

    def test_rknn_type_comes_from_hash_bound_manifest(self):
        path = self.root / "candidate.rknn"
        path.write_bytes(b"rknn fixture")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        for manifest_format, validator, expected in (
            ("voice-changer-llvc-rknn-v1", "voice_changer.ai.llvc._validate_rknn_package",
             "llvc_rknn"),
            ("voice-changer-rvc-rknn-v1", "voice_changer.ai.rvc_rknn._read_manifest",
             "rvc_rknn"),
        ):
            Path(f"{path}.json").write_text(json.dumps({
                "format": manifest_format,
                "model_sha256": digest,
            }), encoding="utf-8")
            with patch(validator):
                result = inspect_model_file(path)
            self.assertEqual(result.model_type, expected)

    def test_unknown_formats_and_symbolic_links_are_rejected(self):
        unknown = self.root / "unknown.bin"
        unknown.write_bytes(b"model")
        with self.assertRaisesRegex(ValueError, "Unsupported model file extension"):
            inspect_model_file(unknown)

        with self.assertRaisesRegex(ValueError, "does not match a supported"):
            self._inspect_onnx(
                [Tensor("input", "tensor(float)", [1, 3, 224, 224])],
                [Tensor("output", "tensor(float)", [1, 1000])],
            )

        alias = self.root / "alias.pth"
        alias.symlink_to(unknown)
        with self.assertRaisesRegex(ValueError, "regular model file"):
            inspect_model_file(alias)


    def test_hevc_rknn_package_is_detected_from_its_sidecar_manifest(self):
        from voice_changer.ai import hevc_rknn

        path = self.root / "female_voice_hfg_new_rknn.rknn"
        path.write_bytes(b"hevc rknn fixture")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "format": hevc_rknn.MANIFEST_FORMAT,
            "target_platform": hevc_rknn.TARGET_PLATFORM,
            "toolkit_version": hevc_rknn.TOOLKIT_VERSION,
            "model_sha256": digest,
            "frames": hevc_rknn.WINDOW_SAMPLES,
            "history": hevc_rknn.HISTORY_SAMPLES,
            "chunk": hevc_rknn.CHUNK_SAMPLES,
            "source_onnx_sha256": (
                "e0084cf7eaa0956f284037dad6ac8344563c936132c47d8ed3cb74202d0633b6"
            ),
            "simulator_snr_db": 60.0,
        }
        Path(f"{path}.json").write_text(json.dumps(manifest), encoding="utf-8")

        result = inspect_model_file(path)
        self.assertEqual(result.model_type, "hevc_rknn")
        self.assertEqual(result.frames, 11264)
        self.assertEqual(result.sha256, digest)
        self.assertIn("HEVoicechanger RKNN / NPU package detected", result.description)

    def test_hevc_rknn_package_with_a_broken_manifest_is_rejected(self):
        path = self.root / "female_voice_hfg_new_rknn.rknn"
        path.write_bytes(b"hevc rknn fixture")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        Path(f"{path}.json").write_text(json.dumps({
            "format": "voice-changer-hevc-rknn-v1",
            "target_platform": "rk3576",
            "toolkit_version": "2.0.0",
            "model_sha256": digest,
            "frames": 11264,
            "history": 8192,
            "chunk": 3072,
            "source_onnx_sha256": (
                "e0084cf7eaa0956f284037dad6ac8344563c936132c47d8ed3cb74202d0633b6"
            ),
            "simulator_snr_db": 60.0,
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "rknn-toolkit2"):
            inspect_model_file(path)


if __name__ == "__main__":
    unittest.main()
