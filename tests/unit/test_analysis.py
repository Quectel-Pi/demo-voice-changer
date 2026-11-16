"""Synthetic PCM only; no device, audio playback or personal recording."""

import math
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from voice_changer.audio.analysis import AudioAnalysis, SignalAnalyzer


RATE = 48000
WINDOW = RATE // 10


def tone(frequency, amplitude=0.1, *, frames=WINDOW, rate=RATE):
    return (amplitude * np.sin(2 * np.pi * frequency * np.arange(frames) / rate)).astype(np.float32)


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class SignalAnalyzerTests(unittest.TestCase):
    def test_frequency_bands_use_window_compensated_rms_and_common_scale(self):
        analyzer = SignalAnalyzer(RATE, WINDOW, 1)
        for index, frequency in enumerate((100, 1000, 4000)):
            with self.subTest(frequency=frequency):
                result = analyzer.analyze(tone(frequency)[:, None])
                self.assertEqual(max(range(3), key=lambda band: result.bands[band]), index)
                self.assertAlmostEqual(20 * math.log10(result.bands[index]), -23.0103, places=2)
        louder = analyzer.analyze(tone(100, 0.2)[:, None])
        self.assertAlmostEqual(20 * math.log10(louder.bands[0]), -16.9897, places=2)

    def test_pitch_silence_noise_range_and_stereo_conflicts(self):
        analyzer = SignalAnalyzer(RATE, WINDOW, 1)
        for frequency in (50, 80, 110, 220, 440, 800):
            with self.subTest(frequency=frequency):
                result = analyzer.analyze(tone(frequency)[:, None])
                self.assertIsNotNone(result.pitch_hz)
                self.assertAlmostEqual(result.pitch_hz, frequency, delta=max(0.5, frequency * 0.005))
        for samples in (
            tone(30), tone(1000), np.random.default_rng(42).normal(0, 0.1, WINDOW),
            tone(220, 0.00001),
        ):
            self.assertIsNone(analyzer.analyze(samples[:, None]).pitch_hz)
        silence = analyzer.analyze(np.zeros((WINDOW, 1), dtype=np.float32))
        self.assertEqual(silence.bands, (0.0, 0.0, 0.0))
        self.assertIsNone(silence.pitch_hz)
        self.assertTrue(silence.silent)

        stereo = SignalAnalyzer(RATE, WINDOW, 2)
        opposite = stereo.analyze(np.column_stack((tone(220), -tone(220))))
        self.assertAlmostEqual(opposite.pitch_hz, 220, delta=0.5)
        self.assertAlmostEqual(opposite.bands[0], analyzer.analyze(tone(220)[:, None]).bands[0])
        conflicting = stereo.analyze(np.column_stack((tone(220), tone(440))))
        self.assertIsNone(conflicting.pitch_hz)
        for first, second in ((220, 330), (220, 337), (180, 290), (200, 300), (200, 261.63)):
            with self.subTest(mixed=(first, second)):
                mixed = (tone(first, 0.05) + tone(second, 0.05))[:, None]
                self.assertIsNone(analyzer.analyze(mixed).pitch_hz)
        harmonic_voice = (tone(220, 0.05) + tone(440, 0.1))[:, None]
        self.assertAlmostEqual(analyzer.analyze(harmonic_voice).pitch_hz, 220, delta=0.5)

    def test_nyquist_limits_and_nonfinite_are_explicit(self):
        analyzer = SignalAnalyzer(8000, 800, 1)
        result = analyzer.analyze(tone(1000, frames=800, rate=8000)[:, None])
        self.assertTrue(all(band is not None for band in result.bands))
        narrow = SignalAnalyzer(2000, 200, 1)
        self.assertIsNone(narrow.analyze(tone(200, frames=200, rate=2000)[:, None]).bands[2])
        at_boundary = SignalAnalyzer(4000, 400, 1)
        nyquist = (0.1 * (-1.0) ** np.arange(400)).astype(np.float32)[:, None]
        self.assertGreater(at_boundary.analyze(nyquist).bands[2], 0)
        for value in (float("nan"), float("inf"), -float("inf")):
            samples = np.zeros((WINDOW, 1), dtype=np.float32)
            samples[0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "non-finite"):
                SignalAnalyzer(RATE, WINDOW, 1).analyze(samples)


class AudioAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.workers = []
        self.addCleanup(self.cleanup_workers)

    def cleanup_workers(self):
        for worker in self.workers:
            worker.close()
        self.assertFalse(any(thread.name == "voice-changer-analysis" for thread in threading.enumerate()))

    def worker(self, *, capture):
        worker = AudioAnalysis(RATE, 512, 1, capture=capture)
        self.workers.append(worker)
        return worker

    def test_capture_is_bounded_and_independent_and_playback_is_output_only(self):
        for capture in (True, False):
            with self.subTest(capture=capture):
                worker = self.worker(capture=capture)
                worker.start()
                first = tone(100, frames=WINDOW)[:, None]
                second = tone(440, frames=WINDOW)[:, None]
                for start in range(0, WINDOW, 480):
                    worker.push(first[start:start + 480], second[start:start + 480] if capture else None)
                self.assertTrue(wait_for(lambda: worker.snapshot is not None))
                snapshot = worker.snapshot
                self.assertEqual(snapshot.sequence, 1)
                self.assertEqual(snapshot.dropped_blocks, 0)
                self.assertEqual(worker.capacity, max(WINDOW * 3, 512 * 8))
                if capture:
                    self.assertAlmostEqual(snapshot.input.pitch_hz, 100, delta=0.5)
                    self.assertAlmostEqual(snapshot.output.pitch_hz, 440, delta=0.5)
                else:
                    self.assertIsNone(snapshot.input)
                    self.assertAlmostEqual(snapshot.output.pitch_hz, 100, delta=0.5)
                worker.close()

    def test_overflow_is_visible_and_incomplete_windows_are_discarded(self):
        worker = self.worker(capture=False)
        worker._accepting = True
        data = tone(100, frames=512)[:, None]
        for _ in range(worker.capacity // 512 + 1):
            worker.push(data)
        self.assertEqual(worker._dropped, 1)
        worker.start()
        self.assertTrue(wait_for(lambda: worker.snapshot is not None))
        self.assertEqual(worker.snapshot.dropped_blocks, 1)
        self.assertIsNone(worker.snapshot.output)
        first_sequence = worker.snapshot.sequence
        data = tone(440)[:, None]
        for start in range(0, WINDOW, 480):
            worker.push(data[start:start + 480])
        self.assertTrue(wait_for(lambda: worker.snapshot.sequence > first_sequence))
        self.assertAlmostEqual(worker.snapshot.output.pitch_hz, 440, delta=0.5)
        self.assertEqual(worker.snapshot.dropped_blocks, 1)

    def test_one_analysis_failure_is_reported_without_hiding_other_lane(self):
        worker = self.worker(capture=True)
        original = SignalAnalyzer.analyze
        calls = 0

        def sometimes_fail(analyzer, samples):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("synthetic analysis failure")
            return original(analyzer, samples)

        with patch.object(SignalAnalyzer, "analyze", sometimes_fail):
            worker.start()
            data = tone(220)[:, None]
            worker.push(data, data)
            self.assertTrue(wait_for(lambda: worker.snapshot is not None))
            self.assertIn("synthetic analysis failure", worker.snapshot.input.error)
            self.assertAlmostEqual(worker.snapshot.output.pitch_hz, 220, delta=0.5)
            worker.push(data, data)
            self.assertTrue(wait_for(lambda: worker.snapshot.sequence >= 2))
            self.assertIn("synthetic analysis failure", worker.snapshot.input.error)
            self.assertEqual(calls, 3)

    def test_overflow_does_not_hide_an_existing_lane_failure(self):
        worker = self.worker(capture=True)
        original = SignalAnalyzer.analyze
        entered, release = threading.Event(), threading.Event()
        calls = 0

        def block_after_failure(analyzer, samples):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("sticky lane failure")
            if calls == 3:
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("test failed to release analysis")
            return original(analyzer, samples)

        with patch.object(SignalAnalyzer, "analyze", block_after_failure):
            worker.start()
            data = tone(220)[:, None]
            worker.push(data, data)
            self.assertTrue(wait_for(lambda: worker.snapshot is not None))
            self.assertIn("sticky lane failure", worker.snapshot.input.error)
            worker.push(data, data)
            try:
                self.assertTrue(entered.wait(2))
                block = data[:512]
                for _ in range(worker.capacity // 512 + 2):
                    worker.push(block, block)
                self.assertGreater(worker._dropped, 0)
            finally:
                release.set()
            self.assertTrue(wait_for(lambda: (
                worker.snapshot.dropped_blocks > 0 and worker.snapshot.output is None
            )))
            self.assertIn("sticky lane failure", worker.snapshot.input.error)
            self.assertIsNone(worker.snapshot.output)


if __name__ == "__main__":
    unittest.main()
