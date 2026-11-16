"""Unit tests for the experimental Beatrice v2 engine wrapper (no model needed)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from voice_changer.ai import beatrice
from voice_changer.ai.beatrice import (
    HOP_IN,
    HOP_OUT,
    SPEAKER_COUNT,
    BeatriceConfig,
    BeatriceConverter,
    BeatriceProcessorFactory,
    clamp_external_formant,
    clamp_speaker,
    configured,
    unavailable_reason,
)
from voice_changer.ai.rvc_worker import RvcWorkerProcessor
from voice_changer.audio.device_types import CaptureConfig, DeviceRef


def device(index: int, name: str = "Device") -> DeviceRef:
    return DeviceRef(index, name, 0)


def capture_config(
    *, samplerate: int = 48000, blocksize: int = 1024, channels: int = 1,
) -> CaptureConfig:
    return CaptureConfig(device(0), samplerate, blocksize, channels, 0.5)


def rubberband_available() -> bool:
    """Whether the host can create a Rubber Band stage (the board has 2.3.0)."""
    try:
        from voice_changer.dsp.rubberband import RubberBand

        RubberBand(48000, 1, 1024).close()
    except Exception:
        return False
    return True


def build_tree(root: Path, *, model: bool = True, speaker: int = 3) -> BeatriceConfig:
    api_dir = root / "api"
    package = api_dir / "beatrice"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    if model:
        for name in beatrice.PARAMETER_FILES:
            (model_dir / name).write_bytes(b"\0")
    return BeatriceConfig(api_dir, model_dir, speaker)


class FakeEngine:
    def __init__(self, *paths: str) -> None:
        self.paths = paths
        self.speaker: int | None = None
        self.pitch_shift: float | None = None
        self.formant_shift: float | None = None
        self.error = 0
        self.conversions = 0
        self.ready = True

    def is_ready(self) -> bool:
        return self.ready

    def set_target_speaker(self, speaker: int) -> None:
        self.speaker = speaker

    def set_pitch_shift_semitone(self, value: float) -> None:
        self.pitch_shift = value

    def set_formant_shift(self, value: float) -> None:
        self.formant_shift = value

    def convert(self, block: np.ndarray):
        if block.shape != (HOP_IN,):
            raise AssertionError(f"unexpected hop shape {block.shape}")
        self.conversions += 1
        return np.full(HOP_OUT, 0.2, dtype=np.float32), np.float32(0.0)

    def last_error(self) -> int:
        return self.error


class FakeRuntime:
    def __init__(self) -> None:
        self.engines: list[FakeEngine] = []

    def SimpleBeatrice(self, *paths: str) -> FakeEngine:  # noqa: N802 - mirrors the API
        engine = FakeEngine(*paths)
        self.engines.append(engine)
        return engine


class ConfigTests(unittest.TestCase):
    def test_configured_requires_both_environment_variables(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(beatrice.API_ENV, None)
            os.environ.pop(beatrice.MODEL_ENV, None)
            self.assertIsNone(configured())
            with mock.patch.dict(os.environ, {beatrice.API_ENV: "/api/path"}):
                self.assertIsNone(configured())
            with mock.patch.dict(
                os.environ,
                {beatrice.API_ENV: "/api/path", beatrice.MODEL_ENV: "/model/path"},
            ):
                config = configured()
        self.assertIsNotNone(config)
        self.assertEqual(config.api_dir, Path("/api/path"))
        self.assertEqual(config.model_dir, Path("/model/path"))
        self.assertEqual(config.speaker, 0)

    def test_clamp_speaker_bounds(self) -> None:
        self.assertEqual(clamp_speaker(-5), 0)
        self.assertEqual(clamp_speaker(0), 0)
        self.assertEqual(clamp_speaker(7), 7)
        self.assertEqual(clamp_speaker(500), SPEAKER_COUNT - 1)
        with self.assertRaises(ValueError):
            clamp_speaker(1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            clamp_speaker(True)  # type: ignore[arg-type]

    def test_with_speaker_keeps_directories(self) -> None:
        config = BeatriceConfig(Path("/api"), Path("/model"), 1)
        changed = config.with_speaker(42)
        self.assertEqual(changed.speaker, 42)
        self.assertEqual(changed.api_dir, Path("/api"))
        self.assertEqual(changed.model_dir, Path("/model"))
        self.assertEqual(config.speaker, 1)

    def test_with_voice_keeps_defaults_and_clamps_range(self) -> None:
        config = BeatriceConfig(Path("/api"), Path("/model"), 1)
        female = config.with_voice(10, 12.0, 1.5, 1.18)
        self.assertEqual(female.speaker, 10)
        self.assertEqual(female.pitch_shift_semitone, 12.0)
        self.assertEqual(female.formant_shift, 1.5)
        self.assertEqual(female.external_formant, 1.18)
        self.assertEqual(config.speaker, 1)
        self.assertEqual(config.pitch_shift_semitone, 0.0)
        self.assertEqual(config.external_formant, 1.0)
        clamped = config.with_voice(SPEAKER_COUNT + 5, 99.0, 9.0, 99.0)
        self.assertEqual(clamped.speaker, SPEAKER_COUNT - 1)
        self.assertEqual(clamped.pitch_shift_semitone, beatrice.MAX_PITCH_SHIFT)
        self.assertEqual(clamped.formant_shift, beatrice.MAX_FORMANT_SHIFT)
        self.assertEqual(clamped.external_formant, beatrice.MAX_EXTERNAL_FORMANT)
        with self.assertRaises(ValueError):
            config.with_voice(1, "12", 1.5, 1.18)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            config.with_voice(1, 12.0, None, 1.18)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            config.with_voice(1, 12.0, 1.0, "1.18")  # type: ignore[arg-type]

    def test_clamp_external_formant_bounds(self) -> None:
        self.assertEqual(clamp_external_formant(0.5), beatrice.MIN_EXTERNAL_FORMANT)
        self.assertEqual(clamp_external_formant(1.18), 1.18)
        self.assertEqual(clamp_external_formant(9.0), beatrice.MAX_EXTERNAL_FORMANT)
        with self.assertRaises(ValueError):
            clamp_external_formant(True)  # type: ignore[arg-type]

    def test_unavailable_reason_explains_missing_setup(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(beatrice.API_ENV, None)
            os.environ.pop(beatrice.MODEL_ENV, None)
            reason = unavailable_reason()
        self.assertIn(beatrice.API_ENV, reason)
        self.assertIn(beatrice.MODEL_ENV, reason)

    def test_unavailable_reason_reports_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = build_tree(root, model=False)
            reason = unavailable_reason(config)
            self.assertIn("missing", reason)
            self.assertIn("phone_extractor.bin", reason)

    def test_unavailable_reason_accepts_complete_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = build_tree(Path(temp))
            self.assertIsNone(unavailable_reason(config))

    def test_unavailable_reason_rejects_invalid_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = build_tree(Path(temp), speaker=SPEAKER_COUNT + 5)
            self.assertIsNotNone(unavailable_reason(config))


class ConverterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine_config = build_tree(self.root)
        self.runtime = FakeRuntime()

    def test_streams_48k_device_blocks_through_the_api(self) -> None:
        converter = BeatriceConverter(
            capture_config(), self.engine_config, runtime=self.runtime,
        )
        output = converter(np.zeros((1024, 1), dtype=np.float32))
        self.assertEqual(output.shape, (1024, 1))
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.isfinite(output).all())
        self.assertLessEqual(float(np.abs(output).max()), 1.0)
        self.assertEqual(len(self.runtime.engines), 1)
        self.assertEqual(self.runtime.engines[0].speaker, 3)
        self.assertEqual(self.runtime.engines[0].conversions, 2)

    def test_applies_the_voice_pitch_and_formant_defaults(self) -> None:
        config = BeatriceConfig(
            self.engine_config.api_dir, self.engine_config.model_dir, 9,
            pitch_shift_semitone=12.0, formant_shift=1.5,
        )
        BeatriceConverter(capture_config(), config, runtime=self.runtime)
        engine = self.runtime.engines[0]
        self.assertEqual(engine.speaker, 9)
        self.assertEqual(engine.pitch_shift, 12.0)
        self.assertEqual(engine.formant_shift, 1.5)

    def test_default_config_does_not_touch_pitch_or_formant(self) -> None:
        converter = BeatriceConverter(
            capture_config(), self.engine_config, runtime=self.runtime,
        )
        engine = self.runtime.engines[0]
        self.assertIsNone(engine.pitch_shift)
        self.assertIsNone(engine.formant_shift)
        self.assertEqual(converter.flush().shape, (0, 1))

    @unittest.skipUnless(rubberband_available(), "Rubber Band library is not available")
    def test_external_formant_buffers_a_flushed_tail(self) -> None:
        config = self.engine_config.with_voice(10, 12.0, 0.0, 1.18)
        converter = BeatriceConverter(capture_config(), config, runtime=self.runtime)
        self.addCleanup(converter.close)
        output = converter(np.zeros((1024, 1), dtype=np.float32))
        self.assertEqual(output.shape, (1024, 1))
        self.assertTrue(np.isfinite(output).all())
        tail = converter.flush()
        self.assertGreater(tail.shape[0], 0)
        self.assertEqual(tail.shape[1], 1)

    def test_second_block_stays_normalized(self) -> None:
        converter = BeatriceConverter(
            capture_config(), self.engine_config, runtime=self.runtime,
        )
        samples = np.full((1024, 1), 0.1, dtype=np.float32)
        converter(samples)
        output = converter(samples)
        self.assertEqual(output.shape, (1024, 1))
        self.assertTrue(np.isfinite(output).all())

    def test_stereo_output_is_duplicated(self) -> None:
        converter = BeatriceConverter(
            capture_config(channels=2), self.engine_config, runtime=self.runtime,
        )
        output = converter(np.zeros((1024, 2), dtype=np.float32))
        self.assertEqual(output.shape, (1024, 2))
        np.testing.assert_array_equal(output[:, 0], output[:, 1])

    def test_rejects_unsupported_device_format(self) -> None:
        for config in (
            capture_config(samplerate=44100),
            capture_config(blocksize=512),
            capture_config(channels=4),
        ):
            with self.assertRaises(ValueError):
                BeatriceConverter(config, self.engine_config, runtime=self.runtime)

    def test_unavailable_model_raises_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = build_tree(Path(temp), model=False)
            with self.assertRaises(RuntimeError) as raised:
                BeatriceConverter(capture_config(), config, runtime=self.runtime)
        self.assertIn("missing", str(raised.exception))
        self.assertEqual(self.runtime.engines, [])

    def test_invalid_device_audio_is_rejected(self) -> None:
        converter = BeatriceConverter(
            capture_config(), self.engine_config, runtime=self.runtime,
        )
        for samples in (
            np.zeros((1024,), dtype=np.float32),
            np.zeros((1024, 1), dtype=np.float64),
            np.full((1024, 1), np.nan, dtype=np.float32),
        ):
            with self.assertRaises(ValueError):
                converter(samples)
        self.assertEqual(self.runtime.engines[0].conversions, 0)

    def test_engine_error_is_surfaced(self) -> None:
        class FailingRuntime(FakeRuntime):
            def SimpleBeatrice(self, *paths: str) -> FakeEngine:  # noqa: N802
                engine = super().SimpleBeatrice(*paths)
                engine.error = 7
                return engine

        with self.assertRaises(RuntimeError) as raised:
            converter = BeatriceConverter(
                capture_config(), self.engine_config, runtime=FailingRuntime(),
            )
            converter(np.zeros((1024, 1), dtype=np.float32))
        self.assertIn("7", str(raised.exception))

    def test_not_ready_engine_raises(self) -> None:
        class NotReadyRuntime(FakeRuntime):
            def SimpleBeatrice(self, *paths: str) -> FakeEngine:  # noqa: N802
                engine = super().SimpleBeatrice(*paths)
                engine.ready = False
                return engine

        with self.assertRaises(RuntimeError):
            BeatriceConverter(capture_config(), self.engine_config, runtime=NotReadyRuntime())

    def test_load_runtime_imports_extracted_package(self) -> None:
        api_dir = self.root / "runtime_api"
        package = api_dir / "beatrice"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            "def load_beatrice(version):\n    return {'version': version}\n",
            encoding="utf-8",
        )
        try:
            runtime = beatrice._load_runtime(api_dir)
            self.assertEqual(runtime, {"version": beatrice.VERSION})
        finally:
            sys.modules.pop("beatrice", None)
            if str(api_dir) in sys.path:
                sys.path.remove(str(api_dir))


class FactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.engine_config = build_tree(Path(self.temp.name))

    def test_rejects_unsupported_device_format(self) -> None:
        factory = BeatriceProcessorFactory(self.engine_config)
        with self.assertRaises(ValueError):
            factory(capture_config(samplerate=44100), None)
        with self.assertRaises(ValueError):
            factory(capture_config(blocksize=512), None)

    def test_builds_worker_with_device_block_conversion(self) -> None:
        config = capture_config()
        fake_converter = mock.Mock()
        fake_converter.config = config
        with mock.patch.object(beatrice, "BeatriceConverter", return_value=fake_converter):
            worker = BeatriceProcessorFactory(self.engine_config)(config, None)
        self.assertIsInstance(worker, RvcWorkerProcessor)
        self.assertIs(worker.convert, fake_converter)
        self.assertEqual(worker.conversion_frames, config.blocksize)


class PreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.engine_config = build_tree(Path(self.temp.name))

    def test_block_budget_follows_device_block_size(self) -> None:
        self.assertAlmostEqual(beatrice.block_budget_ms(capture_config()), 21.333, places=3)
        self.assertAlmostEqual(
            beatrice.block_budget_ms(capture_config(blocksize=2496)), 52.0, places=3
        )

    def test_measure_block_ms_uses_one_converter(self) -> None:
        runtime = FakeRuntime()
        measured = beatrice.measure_block_ms(
            capture_config(), self.engine_config, blocks=2, runtime=runtime
        )
        self.assertGreater(measured, 0.0)
        self.assertEqual(len(runtime.engines), 1)
        self.assertEqual(runtime.engines[0].speaker, self.engine_config.speaker)

    def test_measure_block_ms_rejects_bad_device_format(self) -> None:
        with self.assertRaises(ValueError):
            beatrice.measure_block_ms(
                capture_config(samplerate=44100), self.engine_config, blocks=1,
                runtime=FakeRuntime(),
            )
