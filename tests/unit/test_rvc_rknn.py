import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from voice_changer.ai.rvc_rknn import (
    RVC_RKNN_MANIFEST_FORMAT,
    RVC_SYNTHESIS_INPUTS,
    RvcRknnSynthesizer,
    _read_manifest,
)
from voice_changer.ai.rvc_frontend import RvcOnnxPipeline


class FakeRuntime:
    def __init__(self, result=None):
        self.result = [np.zeros((1, 1, 1280), dtype=np.float32)] if result is None else result
        self.calls = []
        self.closed = 0

    def run(self, inputs):
        self.calls.append(inputs)
        return self.result

    def close(self):
        self.closed += 1


def manifest(*, frames=4, channels=768, rate=32000, feature_dtype="float32"):
    return {
        "format": RVC_RKNN_MANIFEST_FORMAT,
        "target_platform": "rk3576",
        "quantized": False,
        "inputs": list(RVC_SYNTHESIS_INPUTS),
        "outputs": ["audio"],
        "frames": frames,
        "feature_channels": channels,
        "feature_dtype": feature_dtype,
        "model_sample_rate": rate,
    }


class RvcRknnTests(unittest.TestCase):
    def inputs(self, channels=768, frames=4):
        return (
            np.ones((frames, channels), dtype=np.float32),
            np.array([1, 20, 100, 255][:frames], dtype=np.int64),
            np.array([0, 80, 120, 400][:frames], dtype=np.float32),
        )

    def test_synthesis_passes_fixed_shape_tensors_and_returns_owned_mono_audio(self):
        runtime = FakeRuntime()
        synthesizer = RvcRknnSynthesizer(runtime, manifest())
        features, pitch, pitch_hz = self.inputs()
        before = [value.copy() for value in (features, pitch, pitch_hz)]
        audio = synthesizer.synthesize(features, pitch, pitch_hz, speaker_id=3)

        inputs = runtime.calls[0]
        self.assertEqual(len(inputs), 5)
        self.assertEqual(inputs[0].shape, (1, 4, 768))
        self.assertEqual(inputs[0].dtype, np.float32)
        self.assertEqual(inputs[1].tolist(), [4])
        self.assertEqual(inputs[1].dtype, np.int64)
        self.assertEqual(inputs[2].shape, (1, 4))
        self.assertEqual(inputs[2].dtype, np.int64)
        self.assertEqual(inputs[3].shape, (1, 4))
        self.assertEqual(inputs[3].dtype, np.float32)
        self.assertEqual(inputs[4].tolist(), [3])
        self.assertTrue(all(value.flags.c_contiguous for value in inputs))
        self.assertEqual(audio.shape, (1280,))
        self.assertEqual(audio.dtype, np.float32)
        self.assertFalse(np.shares_memory(audio, runtime.result[0]))
        for expected, actual in zip(before, (features, pitch, pitch_hz)):
            np.testing.assert_array_equal(expected, actual)

    def test_feature_float16_and_supported_output_sample_rates(self):
        for rate in (32000, 40000, 48000):
            with self.subTest(rate=rate):
                samples = 4 * rate // 100
                runtime = FakeRuntime([np.zeros((1, 1, samples), dtype=np.float32)])
                synthesizer = RvcRknnSynthesizer(
                    runtime, manifest(rate=rate, feature_dtype="float16"),
                )
                features, pitch, pitch_hz = self.inputs()
                audio = synthesizer.synthesize(
                    features.astype(np.float16), pitch, pitch_hz,
                )
                self.assertEqual(audio.shape, (samples,))
                self.assertEqual(runtime.calls[0][0].dtype, np.float16)

    def test_rknn_synthesizer_composes_with_the_existing_frontend_pipeline(self):
        class Frontend:
            def extract(self, audio, *, semitones):
                self.audio = audio
                self.semitones = semitones
                return self_features, self_pitch, self_pitch_hz

        self_features, self_pitch, self_pitch_hz = self.inputs()
        runtime = FakeRuntime()
        synthesizer = RvcRknnSynthesizer(runtime, manifest())
        frontend = Frontend()
        pipeline = RvcOnnxPipeline(frontend, synthesizer, model_sample_rate=32000)
        audio = np.zeros(640, dtype=np.float32)
        result = pipeline.convert(audio, semitones=2, speaker_id=1)
        self.assertEqual(result.shape, (1280,))
        self.assertIs(frontend.audio, audio)
        self.assertEqual(frontend.semitones, 2)

    def test_bad_inputs_fail_before_runtime_call(self):
        cases = [
            (np.ones((3, 768), np.float32), *self.inputs()[1:]),
            (np.ones((4, 512), np.float32), *self.inputs()[1:]),
            (np.ones((4, 768), np.float64), *self.inputs()[1:]),
            (np.full((4, 768), np.nan, np.float32), *self.inputs()[1:]),
            (self.inputs()[0], np.array([0, 20, 100, 255]), self.inputs()[2]),
            (self.inputs()[0], self.inputs()[1], np.full(4, np.inf, np.float32)),
        ]
        for features, pitch, pitch_hz in cases:
            runtime = FakeRuntime()
            synthesizer = RvcRknnSynthesizer(runtime, manifest())
            with self.subTest(feature_shape=np.shape(features)), self.assertRaises(ValueError):
                synthesizer.synthesize(features, pitch, pitch_hz)
            self.assertEqual(runtime.calls, [])
        for speaker_id in (-1, True, 1.5, 2**63):
            runtime = FakeRuntime()
            synthesizer = RvcRknnSynthesizer(runtime, manifest())
            with self.subTest(speaker_id=speaker_id), self.assertRaises(ValueError):
                synthesizer.synthesize(*self.inputs(), speaker_id=speaker_id)
            self.assertEqual(runtime.calls, [])

    def test_bad_runtime_outputs_fail_and_close_is_idempotent(self):
        invalid_results = [
            [],
            [None],
            [np.zeros((1, 1, 1279), dtype=np.float32)],
            [np.zeros((1, 2, 1280), dtype=np.float32)],
            [np.zeros((1, 1, 1280), dtype=np.float16)],
            [np.full((1, 1, 1280), np.nan, dtype=np.float32)],
        ]
        for result in invalid_results:
            runtime = FakeRuntime(result)
            synthesizer = RvcRknnSynthesizer(runtime, manifest())
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                synthesizer.synthesize(*self.inputs())
        runtime = FakeRuntime()
        synthesizer = RvcRknnSynthesizer(runtime, manifest())
        synthesizer.close()
        synthesizer.close()
        self.assertEqual(runtime.closed, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            synthesizer.synthesize(*self.inputs())

    def test_manifest_validation_checks_hash_and_tensor_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.rknn"
            model.write_bytes(b"synthetic-rknn-artifact")
            package = manifest()
            package.update({
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "input_shapes": [[1, 4, 768], [1], [1, 4], [1, 4], [1]],
                "output_shapes": [[1, 1, 1280]],
            })
            Path(f"{model}.json").write_text(json.dumps(package), encoding="utf-8")
            self.assertEqual(_read_manifest(model)["frames"], 4)
            package["frames"] = 5
            Path(f"{model}.json").write_text(json.dumps(package), encoding="utf-8")
            with self.assertRaises(ValueError):
                _read_manifest(model)


if __name__ == "__main__":
    unittest.main()
