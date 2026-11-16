from dataclasses import dataclass
import unittest

import numpy as np

from voice_changer.ai.rvc_frontend import (
    RvcOnnxContentEncoder,
    RvcOnnxFrontEnd,
    RvcOnnxPitchExtractor,
    RvcOnnxPipeline,
    upsample_content_features,
)
from voice_changer.ai.rvc_onnx import RvcOnnxSynthesizer


@dataclass
class TensorInfo:
    name: str
    type: str
    shape: list[int | str | None]


class FakeSession:
    def __init__(self, inputs, outputs, result, providers=None):
        self.inputs = inputs
        self.outputs = outputs
        self.result = result
        self.providers = providers or ["CPUExecutionProvider"]
        self.calls = []

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_providers(self):
        return self.providers

    def run(self, names, feed):
        self.calls.append((names, feed))
        return self.result


def content_session(*, padding=True, channels=768):
    inputs = [TensorInfo("source", "tensor(float)", [1, "N"])]
    if padding:
        inputs.append(TensorInfo("padding_mask", "tensor(bool)", [1, "N"]))
    output = TensorInfo("hidden", "tensor(float)", [1, "T", channels])
    values = np.arange(5 * channels, dtype=np.float32).reshape(1, 5, channels)
    return FakeSession(inputs, [output], [values]), values


def pitch_session(frames=10):
    inputs = [
        TensorInfo("waveform", "tensor(float)", [1, "N"]),
        TensorInfo("threshold", "tensor(float)", [1]),
    ]
    output = TensorInfo("pitchf", "tensor(float)", [1, "T"])
    values = np.array([0, 0, 100, 120, 0, 150, 160, 180, 0, 200],
                      dtype=np.float32)[None, :frames]
    return FakeSession(inputs, [output], [values]), values


class ContentEncoderTests(unittest.TestCase):
    def test_selects_explicit_contentvec_input_and_layer(self):
        inputs = [TensorInfo("audio", "tensor(float)", [1, "N"])]
        outputs = [
            TensorInfo("units9", "tensor(float)", [1, "T", 256]),
            TensorInfo("unit12", "tensor(float)", [1, "T", 768]),
            TensorInfo("unit12s", "tensor(float)", [1, "T", 768]),
        ]
        values = np.ones((1, 5, 768), dtype=np.float32)
        session = FakeSession(inputs, outputs, [values])
        encoder = RvcOnnxContentEncoder(session, input_name="audio", output_name="unit12")
        np.testing.assert_array_equal(encoder.extract(np.zeros(1600, dtype=np.float32)), values[0])
        self.assertEqual(session.calls[0][0], ["unit12"])
        self.assertEqual(set(session.calls[0][1]), {"audio"})
        with self.assertRaises(ValueError):
            RvcOnnxContentEncoder(session, input_name="audio")
        with self.assertRaises(ValueError):
            RvcOnnxContentEncoder(session, input_name="audio", output_name="missing")
        with self.assertRaises(ValueError):
            RvcOnnxContentEncoder(session, input_name="other", output_name="unit12")

    def test_extracts_valid_finite_mono_content(self):
        for padding in (False, True):
            for channels in (256, 768):
                with self.subTest(padding=padding, channels=channels):
                    session, expected = content_session(padding=padding, channels=channels)
                    encoder = RvcOnnxContentEncoder(session)
                    audio = np.linspace(-1, 1, 1600, dtype=np.float32)
                    features = encoder.extract(audio)
                    np.testing.assert_array_equal(features, expected[0])
                    names, feed = session.calls[0]
                    self.assertEqual(names, ["hidden"])
                    self.assertEqual(set(feed), {"source", "padding_mask"} if padding else {"source"})
                    np.testing.assert_array_equal(feed["source"], audio[None])
                    if padding:
                        self.assertEqual(feed["padding_mask"].dtype, np.bool_)
                        self.assertFalse(feed["padding_mask"].any())
                    self.assertTrue(feed["source"].flags.c_contiguous)

    def test_rejects_bad_schema_at_construction(self):
        invalid = []
        s, _ = content_session()
        s.providers = ["CUDAExecutionProvider"]
        invalid.append(s)
        s, _ = content_session()
        s.inputs[0].name = "waveform"
        invalid.append(s)
        s, _ = content_session()
        s.inputs[1].type = "tensor(int64)"
        invalid.append(s)
        s, _ = content_session()
        s.outputs[0].shape[-1] = 512
        invalid.append(s)
        s, _ = content_session()
        s.outputs.append(s.outputs[0])
        invalid.append(s)
        for session in invalid:
            with self.subTest(inputs=[x.name for x in session.inputs]), self.assertRaises(ValueError):
                RvcOnnxContentEncoder(session)

    def test_rejects_bad_waveform_and_model_outputs(self):
        session, _ = content_session()
        encoder = RvcOnnxContentEncoder(session)
        bad_inputs = (
            np.ones(1600, dtype=np.float64),
            np.ones((1, 1600), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.full(1600, 1.1, dtype=np.float32),
            np.full(1600, np.nan, dtype=np.float32),
        )
        for audio in bad_inputs:
            with self.subTest(dtype=audio.dtype, shape=audio.shape), self.assertRaises(ValueError):
                encoder.extract(audio)
        for result in (
            [],
            [np.zeros((1, 5, 768), dtype=np.float64)],
            [np.zeros((2, 5, 768), dtype=np.float32)],
            [np.zeros((1, 0, 768), dtype=np.float32)],
            [np.zeros((1, 5, 512), dtype=np.float32)],
            [np.full((1, 5, 768), np.inf, dtype=np.float32)],
        ):
            session.result = result
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                encoder.extract(np.zeros(1600, dtype=np.float32))


class PitchExtractorTests(unittest.TestCase):
    def test_extracts_aligned_f0_and_coarse_pitch_with_shift(self):
        session, raw = pitch_session()
        extractor = RvcOnnxPitchExtractor(session)
        audio = np.zeros(1600, dtype=np.float32)
        before = audio.copy()
        coarse, continuous = extractor.extract(audio, semitones=12)
        self.assertEqual(continuous.shape, (10,))
        np.testing.assert_array_equal(continuous, raw[0] * 2)
        self.assertEqual(coarse.dtype, np.int64)
        self.assertEqual(coarse[0], 1)
        self.assertEqual(coarse[1], 1)
        self.assertGreater(coarse[-1], coarse[2])
        np.testing.assert_array_equal(audio, before)
        names, feed = session.calls[0]
        self.assertEqual(names, ["pitchf"])
        self.assertEqual(feed["waveform"].shape, (1, 1600))
        self.assertAlmostEqual(float(feed["threshold"][0]), 0.3, places=6)

    def test_short_pitch_output_is_aligned_to_end_and_long_output_is_cropped(self):
        session, _ = pitch_session(frames=3)
        extractor = RvcOnnxPitchExtractor(session)
        coarse, f0 = extractor.extract(np.zeros(1600, dtype=np.float32))
        np.testing.assert_array_equal(f0[:7], 0)
        np.testing.assert_array_equal(f0[7:], [0, 0, 100])
        session, _ = pitch_session(frames=15)
        coarse, f0 = RvcOnnxPitchExtractor(session).extract(np.zeros(1600, dtype=np.float32))
        self.assertEqual(f0.shape, (10,))
        np.testing.assert_array_equal(f0, [0, 0, 100, 120, 0, 150, 160, 180, 0, 200])

    def test_invalid_f0_schema_inputs_and_outputs_fail_explicitly(self):
        for change in (
            lambda s: s.providers.__setitem__(0, "CUDAExecutionProvider"),
            lambda s: setattr(s.inputs[0], "type", "tensor(double)"),
            lambda s: setattr(s.inputs[1], "shape", [2]),
            lambda s: setattr(s.outputs[0], "name", "f0"),
            lambda s: setattr(s.outputs[0], "shape", [1, 1, "T"]),
        ):
            session, _ = pitch_session()
            change(session)
            with self.subTest(change=change), self.assertRaises(ValueError):
                RvcOnnxPitchExtractor(session)
        extractor = RvcOnnxPitchExtractor(pitch_session()[0])
        for audio, semitones in (
            (np.zeros(159, dtype=np.float32), 0),
            (np.zeros(161, dtype=np.float32), 0),
            (np.zeros(160, dtype=np.float64), 0),
            (np.full(160, 2, dtype=np.float32), 0),
            (np.full(160, np.nan, dtype=np.float32), 0),
            (np.zeros(160, dtype=np.float32), True),
            (np.zeros(160, dtype=np.float32), 49),
        ):
            with self.subTest(size=audio.size, semitones=semitones), self.assertRaises(ValueError):
                extractor.extract(audio, semitones=semitones)
        for result in (
            [],
            [np.zeros((2, 10), dtype=np.float32)],
            [np.full((1, 10), np.nan, dtype=np.float32)],
            [np.full((1, 10), -1, dtype=np.float32)],
            [np.full((1, 10), np.finfo(np.float32).max, dtype=np.float32)],
        ):
            session, _ = pitch_session()
            session.result = result
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                RvcOnnxPitchExtractor(session).extract(
                    np.zeros(1600, dtype=np.float32), semitones=48,
                )


class FeatureAlignmentTests(unittest.TestCase):
    def test_explicit_contentvec_alignment_accepts_one_missing_50hz_frame(self):
        for source_frames, target_frames in ((9, 20), (49, 100)):
            with self.subTest(source_frames=source_frames):
                features = np.arange(source_frames * 768, dtype=np.float32).reshape(source_frames, 768)
                with self.assertRaises(ValueError):
                    upsample_content_features(features, target_frames)
                result = upsample_content_features(features, target_frames, align_to_input_frames=True)
                positions = np.clip(
                    (np.arange(target_frames) + 0.5) * source_frames / target_frames - 0.5,
                    0, source_frames - 1,
                )
                expected = np.interp(positions, np.arange(source_frames), features[:, 0])
                np.testing.assert_allclose(result[:, 0], expected, rtol=1e-6, atol=1e-4)
                np.testing.assert_array_equal(result[0], features[0])
                np.testing.assert_array_equal(result[-1], features[-1])
                with self.assertRaises(ValueError):
                    upsample_content_features(features, target_frames + 1, align_to_input_frames=True)
        with self.assertRaises(ValueError):
            upsample_content_features(features, 100, align_to_input_frames=1)

    def test_frontend_explicit_alignment_with_missing_frame(self):
        content, _ = content_session()
        content.result = [np.zeros((1, 4, 768), dtype=np.float32)]
        pitch, _ = pitch_session()
        encoder, extractor = RvcOnnxContentEncoder(content), RvcOnnxPitchExtractor(pitch)
        with self.assertRaises(ValueError):
            RvcOnnxFrontEnd(encoder, extractor).extract(np.zeros(1600, dtype=np.float32))
        frontend = RvcOnnxFrontEnd(encoder, extractor, align_to_input_frames=True)
        features, coarse, continuous = frontend.extract(np.zeros(1600, dtype=np.float32))
        self.assertEqual((features.shape, coarse.shape, continuous.shape), ((10, 768), (10,), (10,)))
        with self.assertRaises(ValueError):
            RvcOnnxFrontEnd(encoder, extractor, align_to_input_frames=1)

    def test_interpolation_matches_half_pixel_100hz_grid(self):
        features = np.zeros((3, 256), dtype=np.float32)
        features[:, :2] = [[0, 10], [2, 14], [6, 18]]
        result = upsample_content_features(features, 6)
        np.testing.assert_allclose(result[:, :2], [
            [0, 10], [0.5, 11], [1.5, 13], [3, 15], [5, 17], [6, 18],
        ])
        self.assertTrue(result.flags.c_contiguous)

    def test_target_uses_end_aligned_frames_and_rejects_too_few(self):
        features = np.zeros((4, 768), dtype=np.float32)
        features[:, 0] = np.arange(4, dtype=np.float32)
        np.testing.assert_array_equal(upsample_content_features(features, 6),
                                      upsample_content_features(features, 8)[-6:])
        for values, target in (
            (np.zeros((0, 768), dtype=np.float32), 1),
            (np.zeros((2, 768), dtype=np.float64), 4),
            (np.zeros((2, 512), dtype=np.float32), 4),
            (np.zeros((2, 768), dtype=np.float32), 5),
            (np.zeros((2, 768), dtype=np.float32), 0),
        ):
            with self.subTest(shape=values.shape, target=target), self.assertRaises(ValueError):
                upsample_content_features(values, target)

    def test_frontend_produces_matching_feature_and_f0_grids(self):
        content, values = content_session(channels=256)
        pitch, raw = pitch_session()
        frontend = RvcOnnxFrontEnd(
            RvcOnnxContentEncoder(content), RvcOnnxPitchExtractor(pitch),
        )
        features, coarse, continuous = frontend.extract(np.zeros(1600, dtype=np.float32), semitones=1)
        self.assertEqual(features.shape, (10, 256))
        self.assertEqual(coarse.shape, (10,))
        self.assertEqual(continuous.shape, (10,))
        np.testing.assert_allclose(continuous, raw[0] * (2 ** (1 / 12)), rtol=1e-7)
        with self.assertRaises(ValueError):
            frontend.extract(np.zeros(159, dtype=np.float32))

    def test_frontend_propagates_model_failure_without_fallback(self):
        content, _ = content_session()
        pitch, _ = pitch_session()
        pitch.run = lambda *_: (_ for _ in ()).throw(RuntimeError("pitch model failed"))
        frontend = RvcOnnxFrontEnd(
            RvcOnnxContentEncoder(content), RvcOnnxPitchExtractor(pitch),
        )
        with self.assertRaisesRegex(RuntimeError, "pitch model failed"):
            frontend.extract(np.zeros(1600, dtype=np.float32))

    def test_pipeline_runs_one_chunk_and_enforces_exact_model_duration(self):
        content, _ = content_session(channels=256)
        pitch, _ = pitch_session()
        frontend = RvcOnnxFrontEnd(
            RvcOnnxContentEncoder(content), RvcOnnxPitchExtractor(pitch),
        )
        inputs = [
            TensorInfo("feats", "tensor(float)", [1, "T", 256]),
            TensorInfo("p_len", "tensor(int64)", [1]),
            TensorInfo("pitch", "tensor(int64)", [1, "T"]),
            TensorInfo("pitchf", "tensor(float)", [1, "T"]),
            TensorInfo("sid", "tensor(int64)", [1]),
        ]
        output = TensorInfo("audio", "tensor(float)", [1, 1, "N"])
        synthesis = FakeSession(inputs, [output], [
            np.ones((1, 1, 3200), dtype=np.float32),
        ])
        model = RvcOnnxPipeline(
            frontend, RvcOnnxSynthesizer(synthesis), model_sample_rate=32000,
        )
        result = model.convert(np.zeros(1600, dtype=np.float32), speaker_id=0)
        self.assertEqual(result.shape, (3200,))
        self.assertEqual(content.calls[0][0], ["hidden"])
        self.assertEqual(pitch.calls[0][0], ["pitchf"])
        self.assertEqual(synthesis.calls[0][0], ["audio"])
        synthesis.result = [np.zeros((1, 1, 3199), dtype=np.float32)]
        with self.assertRaisesRegex(RuntimeError, "expected 3200"):
            model.convert(np.zeros(1600, dtype=np.float32))
        with self.assertRaises(ValueError):
            RvcOnnxPipeline(frontend, RvcOnnxSynthesizer(FakeSession(
                inputs, [output], [np.zeros((1, 1, 1), dtype=np.float32)],
            )), model_sample_rate=44100)


if __name__ == "__main__":
    unittest.main()
