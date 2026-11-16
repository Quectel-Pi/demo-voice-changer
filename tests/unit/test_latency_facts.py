"""Latency facts are recorded, never judged; synthetic values only."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from voice_changer.audio import latency_facts


class StartupReserveTests(unittest.TestCase):
    def test_uses_the_documented_reserve_formula(self):
        self.assertEqual(latency_facts.startup_frames(512, 1000), 2048)
        self.assertEqual(latency_facts.startup_frames(512, 4000), 8000)

    def test_rejects_invalid_inputs(self):
        for blocksize, start_pad in ((0, 10), (-1, 10), (512, -1)):
            with self.subTest(blocksize=blocksize, start_pad=start_pad):
                with self.assertRaises(ValueError):
                    latency_facts.startup_frames(blocksize, start_pad)


class FactsTests(unittest.TestCase):
    def test_reports_scheduling_values_without_claiming_end_to_end(self):
        facts = latency_facts.measure_facts(
            samplerate=48000, blocksize=512, channels=1, start_pad=1000,
        )
        self.assertAlmostEqual(facts.buffer_ms, 512 / 48000 * 1000)
        self.assertEqual(facts.dsp_startup_frames, 2048)
        self.assertAlmostEqual(facts.dsp_startup_ms, 2048 / 48000 * 1000)
        documented = facts.documented()
        self.assertIsNone(documented["end_to_end_acoustic_ms"])
        self.assertIn("not measured", documented["note"])

    def test_unmeasured_values_stay_none_not_zero(self):
        facts = latency_facts.measure_facts(
            samplerate=48000, blocksize=512, channels=1, start_pad=0,
        )
        documented = facts.documented()
        self.assertIsNone(documented["stream_reported_input_ms"])
        self.assertIsNone(documented["stream_reported_output_ms"])
        self.assertIsNone(documented["callback_peak_seconds"])

    def test_rejects_bool_and_negative_measurements(self):
        with self.assertRaises(ValueError):
            latency_facts.LatencyFacts(48000, 512, 1, 10.0, 2048, 42.0, stream_input_ms=True)
        with self.assertRaises(ValueError):
            latency_facts.LatencyFacts(48000, 512, 1, 10.0, 2048, 42.0, stream_output_ms=-1.0)
        with self.assertRaises(ValueError):
            latency_facts.LatencyFacts(0, 512, 1, 10.0, 2048, 42.0)

    def test_report_round_trips_through_json(self):
        facts = latency_facts.measure_facts(
            samplerate=48000, blocksize=512, channels=1, start_pad=1000,
            stream_input_ms=10.7, stream_output_ms=10.7,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latency.json"
            latency_facts.write_report(facts, path)
            document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["blocksize"], 512)
        self.assertEqual(document["stream_reported_input_ms"], 10.7)
        self.assertIsNone(document["end_to_end_acoustic_ms"])


if __name__ == "__main__":
    unittest.main()
