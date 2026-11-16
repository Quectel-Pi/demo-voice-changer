"""Unit tests for the offline HEVC preview helper (no model, no audio device)."""

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from voice_changer.ai import hevc_preview


class PreviewDirectoryTests(unittest.TestCase):
    def test_directory_follows_xdg_data_home(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"XDG_DATA_HOME": temporary}):
                self.assertEqual(
                    hevc_preview.preview_directory(),
                    Path(temporary) / "voice-changer" / "previews",
                )

    def test_file_name_uses_model_stem_and_timestamp(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"XDG_DATA_HOME": temporary},
        ):
            named = hevc_preview.preview_file_name(Path("/models/female_voice_old.onnx"))
            self.assertEqual(named.parent, Path(temporary) / "voice-changer" / "previews")
            self.assertTrue(named.name.startswith("hevc-female_voice_old-"))
            self.assertTrue(named.name.endswith(".wav"))


class ResampleTests(unittest.TestCase):
    def test_identity_resample_returns_a_copy(self):
        samples = np.linspace(-0.5, 0.5, 480, dtype=np.float32)
        output = hevc_preview.resample(samples, 48000, 48000)
        np.testing.assert_array_equal(output, samples)
        self.assertIsNot(output, samples)

    def test_length_matches_the_rate_ratio(self):
        samples = np.linspace(-0.25, 0.25, 24000, dtype=np.float32)
        down = hevc_preview.resample(samples, 48000, 16000)
        self.assertEqual(down.size, 8000)
        back = hevc_preview.resample(down, 16000, 48000)
        self.assertEqual(back.size, 24000)

    def test_rejects_unexpected_audio(self):
        with self.assertRaises(ValueError):
            hevc_preview.resample(np.zeros((10, 2), dtype=np.float32), 48000, 16000)
        with self.assertRaises(ValueError):
            hevc_preview.resample(np.zeros(10, dtype=np.float64), 48000, 16000)


class _FakeSession:
    def __init__(self, output: np.ndarray):
        self.output = output
        self.inputs = []

    def run(self, output_names, feeds):
        self.inputs.append(feeds[hevc_preview.INPUT_NAME].shape)
        return [self.output]


class ConvertTests(unittest.TestCase):
    def test_convert_resamples_around_one_batch_inference(self):
        audio = np.zeros(48000, dtype=np.float32)
        fake = _FakeSession(np.full((1, 1, 16000), 0.1, dtype=np.float32))
        with patch.object(hevc_preview, "_session", return_value=fake), patch(
            "voice_changer.ai.hevc.validate_hevc_session", return_value=None,
        ):
            output = hevc_preview.convert(audio, 48000, Path("/models/female_voice_hfg_new.onnx"))
        self.assertEqual(fake.inputs, [(1, 1, 16000)])
        self.assertEqual(output.size, 48000)
        self.assertTrue(np.isfinite(output).all())
        self.assertLessEqual(float(np.abs(output).max()), hevc_preview.MAX_PEAK)

    def test_convert_rejects_empty_short_and_invalid_audio(self):
        fake = _FakeSession(np.zeros((1, 1, 16000), dtype=np.float32))
        patches = (
            patch.object(hevc_preview, "_session", return_value=fake),
            patch("voice_changer.ai.hevc.validate_hevc_session", return_value=None),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        with self.assertRaises(hevc_preview.PreviewError):
            hevc_preview.convert(np.zeros(0, dtype=np.float32), 48000, Path("/m.onnx"))
        with self.assertRaises(hevc_preview.PreviewError):
            hevc_preview.convert(np.zeros(100, dtype=np.float32), 48000, Path("/m.onnx"))
        fake.output = np.full((1, 1, 16000), np.nan, dtype=np.float32)
        with self.assertRaises(hevc_preview.PreviewError):
            hevc_preview.convert(np.zeros(48000, dtype=np.float32), 48000, Path("/m.onnx"))


class SaveWavTests(unittest.TestCase):
    def test_writes_pcm16_mono_and_never_overwrites(self):
        samples = np.linspace(-0.5, 0.5, 4800, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "nested" / "preview.wav"
            saved = hevc_preview.save_wav(target, samples)
            self.assertEqual(saved, target)
            with wave.open(str(saved), "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getframerate(), hevc_preview.HOST_RATE)
                self.assertEqual(handle.getnframes(), samples.size)
            again = hevc_preview.save_wav(target, samples)
            self.assertNotEqual(again, saved)
            self.assertEqual(again.name, "preview-2.wav")
            self.assertTrue(again.is_file())


class RecordPlayTests(unittest.TestCase):
    def test_record_uses_the_selected_device_and_returns_mono_float32(self):
        calls = {}

        def rec(frames, **options):
            calls["rec"] = (frames, options)
            return np.zeros((frames, 1), dtype=np.float32)

        def wait():
            calls["wait"] = True

        fake = SimpleNamespace(rec=rec, wait=wait, play=lambda *a, **k: None)
        with patch.dict(sys.modules, {"sounddevice": fake}):
            samples = hevc_preview.record(1.0, device=7)
        frames, options = calls["rec"]
        self.assertEqual(frames, hevc_preview.HOST_RATE)
        self.assertEqual(options["samplerate"], hevc_preview.HOST_RATE)
        self.assertEqual(options["channels"], 1)
        self.assertEqual(options["device"], 7)
        self.assertTrue(calls["wait"])
        self.assertEqual(samples.shape, (hevc_preview.HOST_RATE,))
        self.assertEqual(samples.dtype, np.float32)

    def test_play_uses_the_selected_device_and_wraps_errors(self):
        calls = {}

        def play(samples, **options):
            calls["play"] = (samples.size, options)

        def failing_play(samples, **options):
            raise RuntimeError("no such device")

        with patch.dict(sys.modules, {"sounddevice": SimpleNamespace(play=play, rec=None, wait=None)}):
            hevc_preview.play(np.zeros(480, dtype=np.float32), device=3)
        self.assertEqual(calls["play"][0], 480)
        self.assertEqual(calls["play"][1]["device"], 3)
        self.assertTrue(calls["play"][1]["blocking"])
        with patch.dict(
            sys.modules,
            {"sounddevice": SimpleNamespace(play=failing_play, rec=None, wait=None)},
        ), self.assertRaises(hevc_preview.PreviewError):
            hevc_preview.play(np.zeros(480, dtype=np.float32), device=3)

    def test_record_rejects_out_of_range_lengths(self):
        with self.assertRaises(ValueError):
            hevc_preview.record(0.1)
        with self.assertRaises(ValueError):
            hevc_preview.record(20)


class LimitTests(unittest.TestCase):
    def test_scales_loud_audio_and_fades_the_edges(self):
        samples = np.full(48000, 1.5, dtype=np.float32)
        output = hevc_preview._limit(samples)
        self.assertAlmostEqual(float(np.abs(output).max()), hevc_preview.MAX_PEAK, places=5)
        self.assertAlmostEqual(float(output[0]), 0.0, places=5)
        self.assertAlmostEqual(float(output[-1]), 0.0, places=5)
        self.assertEqual(output.size, samples.size)


if __name__ == "__main__":
    unittest.main()
