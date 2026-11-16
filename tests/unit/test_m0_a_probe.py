import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


spec = importlib.util.spec_from_file_location(
    "m0_a_probe", Path(__file__).resolve().parents[2] / "scripts/m0_a_probe.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class SamplesTests(unittest.TestCase):
    @staticmethod
    def _source_volume(left, right):
        return {
            "mute": False,
            "volume": {
                "front-left": {"value": left},
                "front-right": {"value": right},
            },
        }

    def test_source_full_scale_thresholds_follow_cubic_volume_mapping(self):
        full_scale = probe.source_full_scale_thresholds(self._source_volume(65536, 65536))
        scaled = probe.source_full_scale_thresholds(self._source_volume(49108, 32768))
        self.assertAlmostEqual(full_scale[0], 0.999)
        self.assertAlmostEqual(full_scale[1], 0.999)
        self.assertAlmostEqual(scaled[0], 0.999 * (49108 / 65536) ** 3)
        self.assertAlmostEqual(scaled[1], 0.999 * 0.5 ** 3)

    def test_invalid_or_muted_source_volume_fails_closed(self):
        for source in (
            {"mute": True, "volume": self._source_volume(65536, 65536)["volume"]},
            {"volume": self._source_volume(65536, 65536)["volume"]},
            {"mute": False},
            self._source_volume(0, 65536),
            self._source_volume("49108", 65536),
        ):
            with self.subTest(source=source):
                with self.assertRaises(RuntimeError):
                    probe.source_full_scale_thresholds(source)

    def test_scaled_full_scale_count_uses_per_channel_thresholds(self):
        threshold = probe.source_full_scale_thresholds(self._source_volume(49108, 65536))
        samples = probe.Samples("capture", 48000, 256, 1,
                                input_full_scale_thresholds=threshold)
        data = np.zeros((256, 2), dtype=np.float32)
        data[:, 0] = threshold[0] + 0.0001
        data[:, 1] = threshold[1] - 0.0001
        self.assertTrue(samples.process(data, None, 256))
        self.assertEqual(samples.input_near_full_scale_samples, 256)

    def test_source_volume_change_is_rejected(self):
        initial = {"source_volume": {"front-left": {"value": 49108}}}
        changed = {"source_volume": {"front-left": {"value": 50000}}}
        with self.assertRaisesRegex(RuntimeError, "source volume changed"):
            probe.ensure_source_volume_unchanged(initial, changed)

    def test_stereo_mapping_gain_and_ceiling(self):
        samples = probe.Samples("through", 48000, 256, 1)
        samples.position = 2000
        data = np.full((256, 2), (0.2, -0.4), dtype=np.float32)
        out = np.empty_like(data)
        self.assertTrue(samples.process(data, out, 256))
        np.testing.assert_allclose(out[0], (0.02, -0.04), atol=1e-7)
        data.fill(10)
        samples.process(data, out, 256)
        self.assertEqual(samples.input_near_full_scale_samples, 512)
        self.assertLessEqual(float(out.max()), float(np.float32(0.1)))

    def test_fades_and_finite_end(self):
        samples = probe.Samples("through", 48000, 256, 1)
        data = np.ones((256, 2), dtype=np.float32)
        out = np.empty_like(data)
        samples.process(data, out, 256)
        np.testing.assert_array_equal(out[0], (0, 0))
        samples.position = samples.total - 128
        self.assertFalse(samples.process(data, out, 256))
        np.testing.assert_array_equal(out[127:], 0)

    def test_capture_retains_no_audio(self):
        samples = probe.Samples("capture", 48000, 256, 1)
        data = np.full((256, 2), -0.5, dtype=np.float32)
        self.assertTrue(samples.process(data, None, 256))
        self.assertEqual(samples.input_peak, 0.5)
        self.assertEqual(samples.output_peak, 0)
        self.assertEqual(samples.position, 256)

    def test_nonfinite_and_frame_mismatch_fail(self):
        for data, frames in ((np.full((256, 2), np.nan), 256),
                             (np.zeros((128, 2)), 128)):
            with self.subTest(frames=frames):
                samples = probe.Samples("capture", 48000, 256, 1)
                self.assertFalse(samples.process(data, None, frames))
                self.assertIsNotNone(samples.fault)

    def test_tone_is_low_level_and_ends(self):
        samples = probe.Samples("tone", 48000, 256, 1)
        out = np.empty((256, 2), dtype=np.float32)
        samples.process(None, out, 256)
        self.assertLessEqual(float(np.abs(out).max()), 0.01)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        samples.position = samples.total - 128
        self.assertFalse(samples.process(None, out, 256))
        np.testing.assert_array_equal(out[128:], 0)

    @staticmethod
    def _run_stream(samples, data):
        block = samples.blocksize
        out = np.empty_like(data)
        for start in range(0, len(data), block):
            samples.process(data[start:start + block], out[start:start + block], block)
        return out[:samples.total]

    def test_markers_add_exactly_at_reported_positions(self):
        rate, block = 48000, 1024
        rng = np.random.default_rng(1)
        total = round(rate * 2)
        frames = -(-total // block) * block
        data = rng.uniform(-0.5, 0.5, (frames, 2)).astype(np.float32)
        plain = self._run_stream(probe.Samples("through", rate, block, 2), data)
        marked_samples = probe.Samples("through", rate, block, 2, markers=True)
        marked = self._run_stream(marked_samples, data)
        report = marked_samples.marker_report()
        self.assertEqual(report["start_marker_samples"], [12000])
        self.assertEqual(report["end_marker_samples"], [total - 24000, total - 12000])
        wave = probe.marker_waveform(rate)
        expected = np.zeros(total, dtype=np.float32)
        for start in report["start_marker_samples"] + report["end_marker_samples"]:
            expected[start:start + len(wave)] += wave
        for channel in (0, 1):
            np.testing.assert_allclose(marked[:, channel] - plain[:, channel], expected, atol=1e-6)
        self.assertLessEqual(float(np.abs(marked).max()), float(np.float32(0.1)))

    def test_marker_level_and_shape(self):
        wave = probe.marker_waveform(48000)
        self.assertEqual(len(wave), round(48000 * probe.MARKER_SECONDS))
        self.assertLessEqual(float(np.abs(wave).max()), probe.MARKER_AMPLITUDE)
        self.assertGreater(float(np.abs(wave).max()), 0.9 * probe.MARKER_AMPLITUDE)
        self.assertEqual(float(wave[0]), 0.0)
        self.assertAlmostEqual(float(wave[-1]), 0.0, places=6)

    def test_marker_separable_from_upward_test_chirp(self):
        rate = 48000
        marker = probe.marker_waveform(rate).astype(np.float64)
        t = np.arange(round(0.04 * rate)) / rate
        chirp = np.hanning(len(t)) * np.sin(2 * np.pi * (700 * t + 3300 * t * t / (2 * 0.04)))
        def peak(a, b):
            return np.abs(np.correlate(a, b, "full")).max() / (np.linalg.norm(a) * np.linalg.norm(b))
        self.assertGreater(peak(marker, marker), 0.999)
        self.assertLess(peak(chirp, marker), 0.25)

    def test_markers_only_for_through(self):
        args = probe.parser().parse_args(["--mode", "tone", "--markers", "--safety-confirmed"])
        with self.assertRaisesRegex(ValueError, "--markers"):
            probe.run(args, None, {})
        self.assertIsNone(probe.Samples("through", 48000, 256, 1).marker_report())
        self.assertFalse(probe.parser().parse_args([]).markers)

    def test_output_requires_explicit_safety_confirmation(self):
        args = probe.parser().parse_args(["--mode", "through"])
        with self.assertRaisesRegex(ValueError, "safety-confirmed"):
            probe.run(args, None, {})

    @patch.object(probe, "query", return_value="unexpected source")
    def test_route_change_rejected(self, query):
        with self.assertRaisesRegex(RuntimeError, "source changed"):
            probe.check_route(False)

    def test_default_does_not_open_stream(self):
        self.assertEqual(probe.parser().parse_args([]).mode, "list")

    @patch.object(probe, "query")
    def test_explicit_analog_port_no_silent_fallback(self, query):
        source = {"name": probe.SOURCE,
                  "active_port": "analog-input-headset-mic-es8389-m2",
                  **self._source_volume(49108, 49108)}
        sink = {"name": probe.SINK, "active_port": "analog-output-es8389-m2", "mute": False}
        import json
        responses = [probe.SOURCE, json.dumps([source]), probe.SINK, json.dumps([sink])]
        query.side_effect = responses
        with self.assertRaisesRegex(RuntimeError, "confirmed output port"):
            probe.check_route(True)
        query.side_effect = responses
        self.assertEqual(probe.check_route(True, sink["active_port"])["output_port"],
                         sink["active_port"])


if __name__ == "__main__":
    unittest.main()
