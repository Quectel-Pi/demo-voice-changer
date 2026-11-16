import unittest
from unittest.mock import patch

import numpy as np

from voice_changer.ai.resampling import StreamingResampler


class StreamingResamplerTests(unittest.TestCase):
    def test_chunking_is_invariant_and_output_count_tracks_rate(self):
        source = (0.5 * np.sin(2 * np.pi * 997 * np.arange(24000) / 48000)).astype(np.float32)
        for output_rate in (16000, 32000, 40000, 48000):
            with self.subTest(output_rate=output_rate):
                whole = StreamingResampler(48000, output_rate).process(source)
                streaming = StreamingResampler(48000, output_rate)
                pieces = [
                    streaming.process(source[start:start + size])
                    for start, size in ((0, 257), (257, 1024), (1281, 8191), (9472, 14528))
                ]
                chunked = np.concatenate(pieces)
                np.testing.assert_array_equal(chunked, whole)
                expected_max = len(source) * output_rate // 48000
                self.assertLessEqual(len(whole), expected_max)
                self.assertGreater(len(whole), expected_max - 100)
                self.assertEqual(whole.dtype, np.float32)
                self.assertTrue(np.isfinite(whole).all())

    def test_passband_dc_is_preserved_and_alias_band_is_attenuated(self):
        resampler = StreamingResampler(48000, 16000)
        dc = np.full(12000, 0.25, dtype=np.float32)
        output = resampler.process(dc)
        self.assertAlmostEqual(float(np.mean(output[100:])), 0.25, places=5)

        t = np.arange(24000, dtype=np.float64) / 48000
        in_band = (0.5 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
        out_band = (0.5 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32)
        in_rms = np.sqrt(np.mean(StreamingResampler(48000, 16000).process(in_band)[100:] ** 2))
        out_rms = np.sqrt(np.mean(StreamingResampler(48000, 16000).process(out_band)[100:] ** 2))
        self.assertGreater(float(in_rms), 0.3)
        self.assertLess(float(out_rms), 0.01)

    def test_equal_rate_is_exact_and_independent(self):
        source = np.array([0, 0.25, -0.5, 1], dtype=np.float32)
        resampler = StreamingResampler(48000, 48000)
        output = resampler.process(source)
        np.testing.assert_array_equal(output, source)
        self.assertFalse(np.shares_memory(output, source))
        self.assertEqual(resampler.group_delay_seconds, 0)

    def test_device_conversion_reuses_filter_coefficients(self):
        source = np.full(2496, 0.1, dtype=np.float32)
        for rates in ((48000, 16000), (16000, 48000)):
            resampler = StreamingResampler(*rates)
            self.assertLessEqual(len(resampler._weights), 3)
            with patch.object(resampler, "_filter", side_effect=AssertionError("recomputed filter")):
                for _ in range(10):
                    self.assertTrue(np.isfinite(resampler.process(source)).all())

    def test_unusual_coprime_rates_do_not_allocate_an_unbounded_phase_table(self):
        resampler = StreamingResampler(48000, 47999)
        self.assertIsNone(resampler._weights)
        source = np.full(1024, 0.2, dtype=np.float32)
        output = resampler.process(source)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_allclose(output[100:], 0.2, atol=1e-6)

    def test_invalid_rates_filter_sizes_and_samples_fail(self):
        for args in ((0, 16000), (192001, 16000), (16000, True), (16000, 7999)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                StreamingResampler(*args)
        for taps in (14, 256, True, 13):
            with self.subTest(taps=taps), self.assertRaises(ValueError):
                StreamingResampler(48000, 16000, taps=taps)
        resampler = StreamingResampler(48000, 16000)
        for invalid in (
            np.ones(10, dtype=np.float64),
            np.ones((1, 10), dtype=np.float32),
            np.full(10, 2, dtype=np.float32),
            np.full(10, np.nan, dtype=np.float32),
        ):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype), self.assertRaises(ValueError):
                resampler.process(invalid)


if __name__ == "__main__":
    unittest.main()
