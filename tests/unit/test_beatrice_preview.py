"""Tests for the offline Beatrice v2 preview (no model is loaded)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from voice_changer.ai import beatrice, beatrice_preview


class _FakeEngine:
    """A deterministic stand-in for ``SimpleBeatrice``; no native code involved."""

    instances: list["_FakeEngine"] = []

    def __init__(self, *paths: str) -> None:
        self.paths = paths
        self.speaker: int | None = None
        self.hops = 0
        _FakeEngine.instances.append(self)

    def set_target_speaker(self, speaker: int) -> None:
        self.speaker = speaker

    def is_ready(self) -> bool:
        return True

    def convert(self, hop: np.ndarray) -> tuple[np.ndarray]:
        self.hops += 1
        assert hop.shape == (beatrice.HOP_IN,)
        return (np.full(beatrice.HOP_OUT, 0.05, dtype=np.float32),)

    def last_error(self) -> int:
        return 0


class _FakeRuntime:
    SimpleBeatrice = _FakeEngine


def _make_engine_package(root: Path) -> tuple[Path, Path]:
    api_dir = root / "api"
    package_dir = api_dir / "beatrice"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    model_dir = root / "model"
    model_dir.mkdir()
    for name in beatrice.PARAMETER_FILES:
        (model_dir / name).write_bytes(b"")
    return api_dir, model_dir


class BeatricePreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeEngine.instances.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        api_dir, model_dir = _make_engine_package(root)
        self.config = beatrice.BeatriceConfig(
            api_dir=api_dir, model_dir=model_dir, speaker=5,
        )
        self._env = mock.patch.dict(
            os.environ, {"XDG_DATA_HOME": str(root / "data")}, clear=False,
        )
        self._env.start()
        self.addCleanup(self._env.stop)
        self._load = mock.patch.object(
            beatrice, "_load_runtime", return_value=_FakeRuntime(),
        )
        self._load.start()
        self.addCleanup(self._load.stop)

    def test_convert_returns_a_finite_clip_of_the_same_duration(self) -> None:
        samples = (np.sin(np.linspace(0.0, 40.0, 48000)) * 0.2).astype(np.float32)
        converted = beatrice_preview.convert(samples, 48000, self.config)
        self.assertEqual(converted.shape, (48000,))
        self.assertEqual(converted.dtype, np.float32)
        self.assertTrue(np.isfinite(converted).all())
        self.assertLessEqual(float(np.max(np.abs(converted))), beatrice_preview.MAX_PEAK)
        self.assertEqual(_FakeEngine.instances[-1].speaker, 5)

    def test_convert_resamples_a_16k_recording(self) -> None:
        samples = np.zeros(16000, dtype=np.float32)
        converted = beatrice_preview.convert(samples, 16000, self.config)
        self.assertEqual(converted.shape, (48000,))
        self.assertTrue(np.isfinite(converted).all())

    def test_convert_rejects_empty_short_and_non_mono_audio(self) -> None:
        with self.assertRaises(beatrice_preview.PreviewError):
            beatrice_preview.convert(np.zeros(0, dtype=np.float32), 48000, self.config)
        with self.assertRaises(beatrice_preview.PreviewError):
            beatrice_preview.convert(np.zeros(100, dtype=np.float32), 48000, self.config)
        with self.assertRaises(beatrice_preview.PreviewError):
            beatrice_preview.convert(
                np.zeros((48000, 2), dtype=np.float32), 48000, self.config,
            )

    def test_convert_requires_a_configured_engine(self) -> None:
        with self.assertRaises(beatrice_preview.PreviewError):
            beatrice_preview.convert(np.zeros(48000, dtype=np.float32), 48000, None)

    def test_convert_rejects_missing_engine_files(self) -> None:
        config = beatrice.BeatriceConfig(
            api_dir=self.config.api_dir.parent / "missing",
            model_dir=self.config.model_dir,
            speaker=0,
        )
        with self.assertRaises(RuntimeError):
            beatrice_preview.convert(np.zeros(48000, dtype=np.float32), 48000, config)

    def test_preview_file_name_uses_the_one_based_speaker(self) -> None:
        path = beatrice_preview.preview_file_name(self.config)
        self.assertEqual(path.parent, beatrice_preview.preview_directory())
        self.assertTrue(path.name.startswith("beatrice-jvs006-"))
        self.assertEqual(path.suffix, ".wav")
        self.assertRegex(path.name, r"^beatrice-jvs006-\d{8}-\d{6}\.wav$")

    def test_preview_file_name_clamps_out_of_range_speakers(self) -> None:
        high = beatrice.BeatriceConfig(
            api_dir=self.config.api_dir,
            model_dir=self.config.model_dir,
            speaker=999,
        )
        self.assertTrue(
            beatrice_preview.preview_file_name(high).name.startswith("beatrice-jvs100-")
        )

    def test_convert_drives_the_engine_hop_by_hop(self) -> None:
        samples = np.zeros(48000, dtype=np.float32)
        beatrice_preview.convert(samples, 48000, self.config)
        engine = _FakeEngine.instances[-1]
        self.assertGreater(engine.hops, 140)  # ~71 blocks * ~2 hops


if __name__ == "__main__":
    unittest.main()
