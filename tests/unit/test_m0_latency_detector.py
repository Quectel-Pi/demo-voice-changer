import importlib.util
import math
import os
import random
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "m0_latency_detector",
    Path(__file__).resolve().parents[2] / "scripts/m0_latency_detector.py",
)
detector = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = detector
spec.loader.exec_module(detector)


RATE = 48_000
CHIP_FRAMES = RATE * detector.CHIP_MILLISECONDS // 1000
EVENT_FRAMES = CHIP_FRAMES * len(detector.EVENT_CODE)
NATIVE_LIBRARY = os.environ.get("VOICE_CHANGER_RUBBERBAND_LIBRARY")
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def _formant_weights(f0, centers):
    weights = []
    for harmonic in range(1, 45):
        frequency = f0 * harmonic
        emphasis = 0.025 + math.fsum(
            math.exp(-0.5 * ((frequency - center) / width) ** 2)
            for center, width in zip(centers, (100.0, 170.0, 240.0))
        )
        weights.append(emphasis / harmonic)
    scale = 0.5 / math.fsum(weights)
    return [(weight * scale, f0 * harmonic) for harmonic, weight in enumerate(weights, 1)]


def _synthetic_recording(event_start, f0, centers, duration_frames,
                         gain=1.0, noise=0.0, seed=1, duplicate=False):
    """Make a pitch/formant-shifted carrier with the same coded event envelope."""
    rng = random.Random(seed)
    partials = _formant_weights(f0, centers)
    samples = [0.0] * duration_frames
    starts = [event_start]
    if duplicate:
        starts.append(event_start + 2 * EVENT_FRAMES)
    for start in starts:
        for frame in range(start, min(start + EVENT_FRAMES, duration_frames)):
            chip = (frame - start) // CHIP_FRAMES
            level = 1.0 if detector.EVENT_CODE[chip] else 0.12
            seconds = frame / RATE
            carrier = math.fsum(
                weight * math.sin(2.0 * math.pi * frequency * seconds)
                for weight, frequency in partials
            )
            samples[frame] += gain * level * carrier
    if noise:
        samples = [sample + rng.uniform(-noise, noise) for sample in samples]
    return samples


class LatencyDetectorTests(unittest.TestCase):
    def test_detects_delay_on_both_sides_of_100ms_after_spectral_change(self):
        input_start = RATE // 10
        duration = RATE
        input_reference = _synthetic_recording(
            input_start, 110.0, (500.0, 1_500.0, 2_500.0), duration
        )
        for delay_ms in (96, 104):
            with self.subTest(delay_ms=delay_ms):
                output_reference = _synthetic_recording(
                    input_start + delay_ms * RATE // 1_000,
                    185.0,
                    (850.0, 2_100.0, 3_200.0),
                    duration,
                    gain=0.35,
                    noise=0.002,
                    seed=delay_ms,
                )
                measured = detector.estimate_latency(
                    input_reference, output_reference, RATE
                )
                expected = delay_ms * RATE // 1_000
                self.assertLessEqual(
                    abs(measured.delay_frames - expected), RATE // 1_000
                )
                self.assertGreater(
                    measured.input_event.score, detector.MIN_CORRELATION
                )
                self.assertGreater(
                    measured.output_event.score, detector.MIN_CORRELATION
                )
                self.assertEqual(measured.uncertainty_frames, 2 * RATE // 1_000)

    def test_handles_attenuation_and_noise_but_rejects_noise_only(self):
        recording = _synthetic_recording(
            RATE // 10,
            185.0,
            (850.0, 2_100.0, 3_200.0),
            RATE,
            gain=0.1,
            noise=0.02,
            seed=22,
        )
        self.assertAlmostEqual(
            detector.detect_event(recording, RATE).start_frame, RATE // 10,
            delta=RATE // 1_000,
        )

        for seed in range(5):
            rng = random.Random(seed)
            noise = [rng.uniform(-0.02, 0.02) for _ in range(RATE)]
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(detector.DetectionError, "not detected"):
                    detector.detect_event(noise, RATE)

    def test_detects_weak_copy_on_both_sides_across_noise_seeds(self):
        input_start = RATE // 10
        input_reference = _synthetic_recording(
            input_start, 110.0, (500.0, 1_500.0, 2_500.0), RATE
        )
        seeds = (0, 5, 14, 22, 29)
        detected_by_noise = {}
        rejected_by_noise = {}

        for noise in (0.002, 0.008, 0.01, 0.012, 0.015):
            detected_by_noise[noise] = 0
            rejected_by_noise[noise] = 0
            for delay_ms in (96, 104):
                expected_delay = delay_ms * RATE // 1_000
                expected_start = input_start + expected_delay
                for seed in seeds:
                    with self.subTest(noise=noise, delay_ms=delay_ms, seed=seed):
                        output_reference = _synthetic_recording(
                            expected_start,
                            185.0,
                            (850.0, 2_100.0, 3_200.0),
                            RATE,
                            gain=0.02,
                            noise=noise,
                            seed=seed,
                        )
                        try:
                            measured = detector.estimate_latency(
                                input_reference, output_reference, RATE
                            )
                        except detector.DetectionError:
                            rejected_by_noise[noise] += 1
                            continue

                        detected_by_noise[noise] += 1
                        self.assertGreaterEqual(
                            measured.output_event.score, detector.MIN_CORRELATION
                        )
                        self.assertLessEqual(
                            abs(measured.delay_frames - expected_delay),
                            RATE // 1_000,
                        )

        self.assertGreaterEqual(detected_by_noise[0.002], 8)
        self.assertGreaterEqual(detected_by_noise[0.008], 7)
        self.assertGreater(rejected_by_noise[0.015], 0)

    def test_rejects_near_threshold_weak_copy_with_false_localization(self):
        recording = _synthetic_recording(
            RATE // 10 + 96 * RATE // 1_000,
            185.0,
            (850.0, 2_100.0, 3_200.0),
            RATE,
            gain=0.02,
            noise=0.01,
            seed=5,
        )
        with self.assertRaisesRegex(detector.DetectionError, "not detected"):
            detector.detect_event(recording, RATE)

    def test_rejects_missing_event(self):
        with self.assertRaisesRegex(detector.DetectionError, "not detected"):
            detector.detect_event([0.0] * RATE, RATE)

    def test_rejects_ambiguous_duplicate_events(self):
        recording = _synthetic_recording(
            RATE // 10,
            185.0,
            (850.0, 2_100.0, 3_200.0),
            RATE,
            duplicate=True,
        )
        with self.assertRaisesRegex(detector.DetectionError, "ambiguous"):
            detector.detect_event(recording, RATE)

    def test_rejects_nonfinite_invalid_and_too_short_input(self):
        for samples, sample_rate in (
            ([0.0, math.nan], RATE),
            ([1e308], RATE),
            ([1e7], RATE),
            ([10**400], RATE),
            ([0.0], 0),
            ([0.0], detector.MAX_SAMPLE_RATE + 1),
            ([0.0] * (EVENT_FRAMES - 1), RATE),
        ):
            with self.subTest(sample_rate=sample_rate, length=len(samples)):
                with self.assertRaises(ValueError):
                    detector.detect_event(samples, sample_rate)


@unittest.skipUnless(
    NATIVE_LIBRARY and HAS_NUMPY,
    "Rubber Band and NumPy are required for native DSP calibration",
)
class NativeDSPDetectorTests(unittest.TestCase):
    def test_rejects_r3_output_with_insufficient_localization_precision(self):
        import numpy as np
        from test_live_processing import NativeDSPTests
        from voice_changer.dsp.parameters import EffectParameters

        event_start = RATE // 2
        native_test = NativeDSPTests(
            "test_real_native_both_pitch_directions_and_limits_keep_sample_rate"
        )
        try:
            default_source = _synthetic_recording(
                event_start,
                110.0,
                (500.0, 1_500.0, 2_500.0),
                event_start + EVENT_FRAMES + RATE,
                gain=0.5,
            )
            default_rendered = native_test.render(
                np.asarray(default_source, dtype=np.float32)[:, None],
                EffectParameters(12.0 * math.log2(1.25), 1.0, False),
                512,
            )
        finally:
            native_test.doCleanups()
        with self.assertRaisesRegex(detector.DetectionError, "not detected"):
            detector.detect_event(default_rendered[:, 0].tolist(), RATE)

        chip_milliseconds = 16
        chip_frames = RATE * chip_milliseconds // 1_000
        event_frames = chip_frames * len(detector.EVENT_CODE)
        with (
            patch.object(detector, "CHIP_MILLISECONDS", chip_milliseconds),
            patch.object(sys.modules[__name__], "CHIP_FRAMES", chip_frames),
            patch.object(sys.modules[__name__], "EVENT_FRAMES", event_frames),
        ):
            source = _synthetic_recording(
                event_start,
                110.0,
                (500.0, 1_500.0, 2_500.0),
                event_start + event_frames + RATE,
                gain=0.5,
            )
            native_test = NativeDSPTests(
                "test_real_native_both_pitch_directions_and_limits_keep_sample_rate"
            )
            try:
                rendered = native_test.render(
                    np.asarray(source, dtype=np.float32)[:, None],
                    EffectParameters(12.0 * math.log2(1.25), 1.0, False),
                    512,
                )
            finally:
                native_test.doCleanups()
            processed = (rendered[:, 0].astype(np.float64) * 0.02).tolist()
            for noise in (0.0, 0.002, 0.004, 0.008):
                for delay_ms in (96, 104):
                    delay_frames = delay_ms * RATE // 1_000
                    shifted = [0.0] * delay_frames + processed
                    for seed in range(5):
                        rng = random.Random(seed)
                        trace = [
                            sample + rng.uniform(-noise, noise)
                            for sample in shifted
                        ]
                        with self.subTest(
                            noise=noise, delay_ms=delay_ms, seed=seed
                        ), self.assertRaisesRegex(
                            detector.DetectionError,
                            "not detected|localization is ambiguous",
                        ):
                            detector.detect_event(trace, RATE)


if __name__ == "__main__":
    unittest.main()
