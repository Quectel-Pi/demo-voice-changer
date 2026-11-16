import unittest
from unittest.mock import Mock

import numpy as np

from voice_changer.ai.experimental_rvc import ExperimentalRvc


class ExperimentalRvcTests(unittest.TestCase):
    def converter(self):
        converter = ExperimentalRvc.__new__(ExperimentalRvc)
        converter._content = Mock()
        converter._content.extract.return_value = np.zeros((9, 768), dtype=np.float32)
        converter._pre = Mock()
        converter._pre.run.return_value = [np.zeros((1, 1, 32, 128), dtype=np.float32)]
        converter._post = Mock()
        converter._post.run.return_value = [np.full((1, 21), 180, dtype=np.float32)]
        converter._pitch_runtime = Mock()
        converter._pitch_runtime.inference.return_value = [np.zeros((1, 3, 32, 128), dtype=np.float32)]
        converter._pitch_manifest = {"input_name": "mel", "output_name": "cnn"}
        converter._rng = np.random.default_rng(1)
        converter.synthesize_feed = Mock(return_value=np.zeros(8000, dtype=np.float32))
        return converter

    def test_complete_feed_shape_and_shift(self):
        converter = self.converter()
        audio, timing = converter.convert(np.zeros(3200, dtype=np.float32), semitones=12)
        feed = converter.synthesize_feed.call_args.args[0]
        self.assertEqual(feed["feats"].shape, (1, 20, 768))
        self.assertEqual(feed["latent_noise"].shape, (1, 192, 20))
        self.assertEqual(feed["source_noise"].shape, (1, 8000, 1))
        np.testing.assert_array_equal(feed["pitchf"], np.full((1, 20), 360, dtype=np.float32))
        self.assertEqual(audio.shape, (8000,))
        self.assertEqual(set(timing), {"content_ms", "f0_ms", "synthesis_ms", "inference_ms"})
        self.assertGreaterEqual(timing["inference_ms"], 0)

    def test_rejects_invalid_input_and_pitch(self):
        converter = self.converter()
        for audio in (np.zeros(3199, dtype=np.float32), np.zeros(3200, dtype=np.float64),
                      np.full(3200, np.nan, dtype=np.float32),
                      np.full(3200, 2, dtype=np.float32)):
            with self.assertRaises(ValueError):
                converter.convert(audio)
        for shift in (True, 25, -25, 1.5):
            with self.assertRaises(ValueError):
                converter.convert(np.zeros(3200, dtype=np.float32), semitones=shift)
        converter.synthesize_feed.assert_not_called()

    def test_npu_failure_is_not_passthrough(self):
        converter = self.converter()
        converter._pitch_runtime.inference.return_value = None
        with self.assertRaisesRegex(RuntimeError, "NPU"):
            converter.convert(np.zeros(3200, dtype=np.float32))
        converter.synthesize_feed.assert_not_called()

    def test_invalid_f0_is_rejected(self):
        converter = self.converter()
        converter._post.run.return_value = [np.full((1, 21), np.nan, dtype=np.float32)]
        with self.assertRaisesRegex(RuntimeError, "F0 output"):
            converter.convert(np.zeros(3200, dtype=np.float32))

    def test_close_releases_npu_once(self):
        converter = self.converter()
        runtime = converter._pitch_runtime
        converter.close()
        converter.close()
        runtime.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
