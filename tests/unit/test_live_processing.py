"""Synthetic samples only; native cases explicitly require a local Rubber Band library."""

from array import array
from ctypes.util import find_library
from dataclasses import replace
import importlib.util
import math
import os
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from voice_changer.audio.device_types import DeviceRef, StreamConfig
from voice_changer.audio.live_processing import FrameRing, LiveProcessor
from voice_changer.dsp.parameters import EffectParameters, LiveControl, PRESETS
from voice_changer.dsp.rubberband import RubberBand


CONFIG = StreamConfig(DeviceRef(0, "test input", 0), DeviceRef(1, "test output", 0),
                      48000, 512, 1, 0.1)
NATIVE = os.environ.get("VOICE_CHANGER_RUBBERBAND_LIBRARY") or find_library("rubberband")


class ParametersTests(unittest.TestCase):
    def test_candidates_and_validation(self):
        self.assertEqual(PRESETS["male_to_female"].pitch_semitones, 7.0)
        self.assertEqual(PRESETS["male_to_female"].formant_scale, 1.18)
        self.assertEqual(PRESETS["female_to_male"].pitch_semitones, -6.0)
        self.assertEqual(PRESETS["female_to_male"].formant_scale, 0.86)
        self.assertTrue(PRESETS["original"].bypass)
        for field, values in (
            ("pitch_semitones", (True, "1", math.nan, math.inf, -12.01, 12.01)),
            ("formant_scale", (True, "1", math.nan, math.inf, 0.79, 1.26)),
            ("input_gain_db", (True, "1", math.nan, math.inf, -24.01, 12.01)),
            ("output_gain_db", (True, "1", math.nan, math.inf, -24.01, 12.01)),
            ("robot_mix", (True, "1", math.nan, math.inf, -0.01, 1.01)),
            ("echo_mix", (True, "1", math.nan, math.inf, -0.01, 0.41)),
            ("reverb_mix", (True, "1", math.nan, math.inf, -0.01, 0.41)),
            ("denoise_strength", (True, "1", math.nan, math.inf, -0.01, 1.01)),
            ("bypass", (1, "yes", None)),
        ):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(EffectParameters(), **{field: value})
        with self.assertRaisesRegex(ValueError, "unity"):
            EffectParameters(1, 1, True)

    def test_gender_presets_land_in_typical_adult_ranges(self):
        # Typical adult F0: male ~85-180 Hz, female ~165-255 Hz.
        m2f, f2m = PRESETS["male_to_female"], PRESETS["female_to_male"]
        self.assertTrue(165 <= 120 * m2f.pitch_scale <= 255)
        self.assertTrue(85 <= 210 * f2m.pitch_scale <= 180)
        self.assertTrue(1.1 <= m2f.formant_scale <= 1.25)
        self.assertTrue(0.8 <= f2m.formant_scale <= 0.92)

    def test_all_presets_are_distinct_and_gains_do_not_enable_original_effects(self):
        self.assertEqual(len(PRESETS), 9)
        self.assertEqual(len(set(PRESETS.values())), 9)
        self.assertEqual([key for key, value in PRESETS.items() if value.bypass], ["original"])
        self.assertTrue(EffectParameters(input_gain_db=12, output_gain_db=-24).bypass)
        self.assertTrue(EffectParameters(denoise_strength=1.0).bypass)
        for name in ("robot_mix", "echo_mix", "reverb_mix"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "no effects"):
                replace(EffectParameters(), **{name: 0.1})

    def test_mailbox_keeps_only_latest_valid_snapshot(self):
        control = LiveControl(PRESETS["original"])
        for _ in range(10000):
            control.submit(PRESETS["male_to_female"])
        self.assertEqual(control.requested, (10000, PRESETS["male_to_female"]))
        with self.assertRaises(ValueError):
            control.submit(None)
        self.assertEqual(control.requested[0], 10000)

    def test_missing_native_library_fails_explicitly(self):
        with patch("voice_changer.dsp.rubberband.find_library", return_value=None):
            with patch.dict(os.environ, {"VOICE_CHANGER_RUBBERBAND_LIBRARY": ""}):
                with self.assertRaisesRegex(RuntimeError, "library is missing"):
                    RubberBand(48000, 1, 512)

    def test_demo_format_limit_is_visible_before_loading_native(self):
        for key, value in (("samplerate", 44100), ("blocksize", 128), ("channels", 3)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "D1 demo DSP"):
                LiveProcessor(replace(CONFIG, **{key: value}), LiveControl(EffectParameters()))


class RingTests(unittest.TestCase):
    def test_wrap_and_failed_operations_do_not_consume_or_overwrite(self):
        ring = FrameRing(5, 1)
        self.assertTrue(ring.push(np.arange(4, dtype=np.float32)[:, None]))
        untouched = np.full((5, 1), -1, dtype=np.float32)
        self.assertFalse(ring.pop(untouched))
        np.testing.assert_array_equal(untouched, -1)
        self.assertFalse(ring.push(untouched))
        output = np.empty((3, 1), dtype=np.float32)
        self.assertTrue(ring.pop(output))
        np.testing.assert_array_equal(output[:, 0], [0, 1, 2])
        self.assertTrue(ring.push(np.arange(4, 8, dtype=np.float32)[:, None]))
        self.assertTrue(ring.pop(untouched))
        np.testing.assert_array_equal(untouched[:, 0], [3, 4, 5, 6, 7])
        self.assertEqual(ring.available, 0)

    def test_spsc_threads_preserve_sequence_and_bound_capacity(self):
        ring = FrameRing(128, 2)
        failures = []
        deadline = time.monotonic() + 5
        def produce():
            for index in range(1000):
                samples = np.full((16, 2), index, dtype=np.float32)
                while not ring.push(samples):
                    if time.monotonic() > deadline:
                        failures.append("producer timeout")
                        return
                    time.sleep(0.0001)
        thread = threading.Thread(target=produce)
        thread.start()
        try:
            output = np.empty((16, 2), dtype=np.float32)
            for index in range(1000):
                while not ring.pop(output):
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.0001)
                np.testing.assert_array_equal(output, index)
                self.assertLessEqual(ring.available, ring.capacity)
        finally:
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])


@unittest.skipUnless(NATIVE, "Rubber Band native library unavailable; no DSP numerical verification")
class NativeDSPTests(unittest.TestCase):
    def processor(self, parameters=EffectParameters(), config=CONFIG):
        processor = LiveProcessor(config, LiveControl(parameters))
        self.addCleanup(processor.close)
        return processor

    def render(self, samples, parameters, blocksize=512):
        config = replace(CONFIG, channels=samples.shape[1], blocksize=blocksize)
        processor = self.processor(parameters, config)
        output = np.empty_like(processor._input_block)
        blocks = []
        for start in range(0, len(samples) - config.blocksize + 1, config.blocksize):
            processor._input_block[:] = samples[start:start + config.blocksize]
            processor._process_block()
            while processor.output.pop(output):
                blocks.append(output.copy())
        result = np.concatenate(blocks)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(processor.control.applied_revision, 0)
        return result

    @staticmethod
    def frequency(samples):
        window = samples[-48000:, 0]
        spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
        return float(np.argmax(spectrum) * 48000 / len(window))

    def test_real_native_original_is_delayed_bypass_without_sample_changes(self):
        rng = np.random.default_rng(42)
        samples = rng.uniform(-0.1, 0.1, (96000, 2)).astype(np.float32)
        for blocksize in (256, 512, 1024):
            with self.subTest(blocksize=blocksize):
                rendered = self.render(samples, PRESETS["original"], blocksize)
                hop = 256
                self.assertTrue(np.all(rendered[:hop] == 0))
                np.testing.assert_array_equal(rendered[hop:], samples[:len(rendered) - hop])

    def test_real_native_both_pitch_directions_and_limits_keep_sample_rate(self):
        time_axis = np.arange(144000) / 48000
        samples = (0.03 * np.sin(2 * np.pi * 220 * time_axis)).astype(np.float32)[:, None]
        for parameters in (
            PRESETS["male_to_female"], PRESETS["female_to_male"],
            PRESETS["loli"], PRESETS["uncle"], PRESETS["monster"],
            EffectParameters(-12, 1, False), EffectParameters(12, 1, False),
        ):
            with self.subTest(parameters=parameters):
                rendered = self.render(samples, parameters)
                measured = self.frequency(rendered)
                expected = 220 * parameters.pitch_scale
                self.assertLess(abs(measured / expected - 1), 0.01)
                self.assertGreater(float(np.abs(rendered).max()), 0.001)
                self.assertLess(len(rendered), len(samples))

    def test_original_input_and_output_gains_apply_without_mutating_source(self):
        phase = np.arange(96000) / 48000
        samples = (0.02 * np.sin(2 * np.pi * 220 * phase)).astype(np.float32)[:, None]
        before = samples.copy()
        for field in ("input_gain_db", "output_gain_db"):
            with self.subTest(field=field):
                parameters = replace(PRESETS["original"], **{field: 6.0})
                rendered = self.render(samples, parameters)
                hop = 256
                expected = samples[:len(rendered) - hop] * 10 ** (6 / 20)
                np.testing.assert_allclose(rendered[hop:][-24000:], expected[-24000:], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(samples, before)

    def test_input_gain_is_before_native_and_paired_raw_remains_pre_gain(self):
        processor = self.processor(EffectParameters(input_gain_db=6, output_gain_db=-6))
        native_process = processor._dsp.process
        observed = {}
        def observe(block):
            observed["last"] = block.copy()
            native_process(block)
        output = np.empty_like(processor._input_block)
        with patch.object(processor._dsp, "process", side_effect=observe):
            for _ in range(100):
                processor._input_block.fill(0.02)
                processor._process_block()
                while processor.output.pop(output):
                    pass
        np.testing.assert_allclose(observed["last"], 0.02 * 10 ** (6 / 20), rtol=0, atol=1e-8)
        np.testing.assert_array_equal(processor._dry_block, np.float32(0.02))
        np.testing.assert_allclose(output, 0.02, rtol=0, atol=1e-7)
        self.assertEqual(processor.control.applied_revision, 0)

    def test_gain_changes_ramp_between_adjacent_samples_and_acknowledge_latest(self):
        processor = self.processor()
        output = np.empty_like(processor._input_block)
        blocks = []
        for gain in (0.0, 12.0, -24.0, 6.0):
            processor.control.submit(EffectParameters(input_gain_db=gain, output_gain_db=gain))
            for _ in range(100):
                processor._input_block.fill(0.01)
                processor._process_block()
                while processor.output.pop(output):
                    blocks.append(output.copy())
            self.assertEqual(processor.control.applied_revision, processor.control.requested[0])
        rendered = np.concatenate(blocks)[:, 0]
        hop = 256
        self.assertLess(float(np.abs(np.diff(rendered[hop:])).max()), 0.001)
        self.assertLessEqual(processor._gained_dry.available, processor.capacity)

    def test_real_native_formant_scale_is_relative_to_the_original_voice(self):
        path = Path(__file__).resolve().parents[2] / "scripts/m0_02b_dsp_measure.py"
        spec = importlib.util.spec_from_file_location("d1_probe_analysis", path)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        samples = np.asarray(probe.make_formant_probe(), dtype=np.float32)[:, None]
        bands = probe.BASE_FORMANT_BANDS
        original = self.render(samples, EffectParameters(0, 1, False))
        base = probe.analyze_window(array("f", original[:, 0]), 12000, 120, bands)
        for parameters in (
            EffectParameters(7, 1, False), EffectParameters(-6, 1, False),
            PRESETS["male_to_female"], PRESETS["female_to_male"],
        ):
            with self.subTest(parameters=parameters):
                scale = parameters.formant_scale
                rendered = self.render(samples, parameters)
                f0 = 120 * parameters.pitch_scale
                scaled_bands = tuple((low * scale, high * scale) for low, high in bands)
                result = probe.analyze_window(array("f", rendered[:, 0]), 12000, f0, scaled_bands)
                self.assertLess(abs(result["f0_hz"] / f0 - 1), 0.02, result)
                matches = sum(
                    current is not None and baseline is not None
                    and abs(current / baseline / scale - 1) <= 0.06
                    for current, baseline in zip(result["formant_peaks_hz"], base["formant_peaks_hz"])
                )
                self.assertGreaterEqual(matches, 2, (base, result))

    def test_real_native_formants_move_without_changing_f0(self):
        path = Path(__file__).resolve().parents[2] / "scripts/m0_02b_dsp_measure.py"
        spec = importlib.util.spec_from_file_location("d1_probe_analysis", path)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        signal = probe.make_formant_probe()
        samples = np.asarray(signal, dtype=np.float32)[:, None]
        bands = probe.BASE_FORMANT_BANDS
        original = self.render(samples, EffectParameters(0, 1, False))
        base = probe.analyze_window(array("f", original[:, 0]), 12000, 120, bands)
        for scale in (0.8, 1.25):
            with self.subTest(scale=scale):
                rendered = self.render(samples, EffectParameters(0, scale, False))
                scaled_bands = tuple((low * scale, high * scale) for low, high in bands)
                result = probe.analyze_window(array("f", rendered[:, 0]), 12000, 120, scaled_bands)
                self.assertLess(abs(result["f0_hz"] / base["f0_hz"] - 1), 0.01)
                matches = sum(
                    current is not None and baseline is not None
                    and abs(current / baseline / scale - 1) <= 0.10
                    for current, baseline in zip(result["formant_peaks_hz"], base["formant_peaks_hz"])
                )
                self.assertGreaterEqual(matches, 2, (base, result))

    def test_buffers_fail_closed_and_never_grow(self):
        processor = self.processor()
        data = np.zeros_like(processor._input_block)
        output = np.ones_like(data)
        self.assertTrue(processor.exchange(data, output))
        np.testing.assert_array_equal(output, 0)
        for _ in range(64):
            if not processor.exchange(data, output):
                break
        self.assertIsNotNone(processor.error)
        self.assertLessEqual(processor.input.available, processor.capacity)
        self.assertFalse(processor.exchange(data, output))
        np.testing.assert_array_equal(output, 0)
        with self.assertRaisesRegex(RuntimeError, "DSP worker failed"):
            processor.check_health()

    def test_startup_waits_for_native_window_reserve(self):
        processor = self.processor()
        block = processor.config.blocksize
        self.assertEqual(processor._startup_frames, 4096)
        samples = np.full_like(processor._input_block, 0.02)
        output = np.empty_like(samples)
        self.assertTrue(processor.output.push(
            np.full((processor._startup_frames - block, 1), 0.02, dtype=np.float32),
        ))
        self.assertTrue(processor.exchange(samples, output))
        self.assertFalse(processor._playing)
        np.testing.assert_array_equal(output, 0)
        self.assertTrue(processor.output.push(samples))
        self.assertTrue(processor.exchange(samples, output))
        self.assertTrue(processor._playing)
        np.testing.assert_array_equal(output, samples)

    def test_native_switches_and_four_callback_worker_pause_preserve_output(self):
        processor = self.processor()
        samples = np.empty_like(processor._input_block)
        output = np.empty_like(samples)
        started = False
        delayed_callbacks = 0
        for index in range(1500):
            phase = (np.arange(CONFIG.blocksize) + index * CONFIG.blocksize) / 48000
            samples[:, 0] = 0.02 * np.sin(2 * np.pi * 220 * phase)
            if index % 300 == 0:
                processor.control.submit(PRESETS[
                    ("original", "male_to_female", "female_to_male")[index // 300 % 3]
                ])
            self.assertTrue(processor.exchange(samples, output), (index, processor.error))
            self.assertTrue(np.isfinite(output).all())
            if started:
                self.assertTrue(processor._playing)
            started = processor._playing
            if index >= 200 and index % 200 < 4:
                delayed_callbacks += 1
                continue
            while processor.input.pop(processor._input_block):
                processor._process_block()
        self.assertTrue(started)
        self.assertEqual(delayed_callbacks, 28)
        self.assertIsNone(processor.error)

    def test_native_worker_has_clean_shutdown_and_explicit_underrun(self):
        processor = self.processor()
        processor.start()
        processor.check_health()
        data = np.zeros_like(processor._input_block)
        output = np.ones_like(data)
        processor._playing = True
        self.assertFalse(processor.exchange(data, output))
        self.assertIn("underrun", str(processor.error))
        np.testing.assert_array_equal(output, 0)
        processor.close()
        self.assertIsNone(processor._thread)

    def test_worker_start_failure_can_release_native_processor(self):
        processor = self.processor()
        with patch("threading.Thread.start", side_effect=RuntimeError("test thread start failure")):
            with self.assertRaisesRegex(RuntimeError, "test thread start failure"):
                processor.start()
        processor.close()
        self.assertIsNone(processor._dsp._state)
        with self.assertRaisesRegex(RuntimeError, "new DSP processor"):
            processor.start()

    def test_latest_parameters_settle_and_bypass_switch_is_finite(self):
        processor = self.processor()
        output = np.empty_like(processor._input_block)
        processor._input_block.fill(0.02)
        for index in range(200):
            if index < 100:
                processor.control.submit(PRESETS["male_to_female"])
            elif index == 100:
                processor.control.submit(PRESETS["original"])
            processor._process_block()
            while processor.output.pop(output):
                self.assertTrue(np.isfinite(output).all())
                self.assertLess(float(np.max(np.abs(output))), 0.2)
        self.assertEqual(processor.control.applied_revision, 101)
        self.assertEqual(processor._mix, 0)
        self.assertEqual(processor._pitch, 0)
        np.testing.assert_allclose(output, 0.02, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
