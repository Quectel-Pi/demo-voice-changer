import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

from voice_changer.ai.availability import availability, preset_voices
from voice_changer.ai.profiles import (
    CONSENT_STATEMENT, NAME_LIMIT, ProfileError, ProfileStore, capability_text, default_root,
    validate_name,
)


def write_wav(path: Path, frames: int = 4800, rate: int = 48000) -> Path:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b"\x01\x00" * frames)
    return path


class ProfileStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.root = self.base / "voices"
        self.store = ProfileStore(self.root)
        self.source = write_wav(self.base / "take.wav")
        self.source_bytes = self.source.read_bytes()

    def tearDown(self):
        self._tmp.cleanup()

    def create(self, **overrides):
        options = dict(name="  我的声音  ", relation="self", consent=True, prompt_text="你好",
                       source=self.source)
        options.update(overrides)
        return self.store.create(options.pop("name"), **options)

    def test_create_copies_reference_and_writes_reference_only_metadata(self):
        profile = self.create()
        self.assertEqual(profile.name, "我的声音")
        self.assertFalse(profile.damaged)
        self.assertAlmostEqual(profile.duration, 0.1)
        self.assertEqual(profile.capabilities, {"tts": False, "vc": False})
        self.assertFalse(profile.supports("tts"))
        self.assertFalse(profile.supports("vc"))
        self.assertIn("未接入模型", capability_text(profile))
        self.assertEqual(profile.directory.parent, self.root)
        self.assertEqual(profile.reference_path.read_bytes(), self.source_bytes)
        self.assertEqual(self.source.read_bytes(), self.source_bytes)
        metadata = json.loads((profile.directory / "profile.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "reference_only")
        self.assertEqual(metadata["consent"]["relation"], "self")
        self.assertTrue(metadata["consent"]["confirmed"])
        self.assertEqual(metadata["consent"]["statement"], CONSENT_STATEMENT)
        self.assertEqual(metadata["prompt_text"], "你好")
        self.assertEqual(metadata["reference"]["frames"], 4800)
        self.assertEqual(len(metadata["reference"]["sha256"]), 64)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), [profile.directory.name])
        self.assertEqual(self.store.list(), (profile,))

    def test_create_rejects_invalid_input_without_writing_or_touching_source(self):
        empty = write_wav(self.base / "empty.wav", frames=0)
        garbage = self.base / "garbage.wav"
        garbage.write_bytes(b"not a wave file")
        cases = (
            dict(consent=False),
            dict(relation="someone"),
            dict(name="   "),
            dict(name="x" * (NAME_LIMIT + 1)),
            dict(name="bad\nname"),
            dict(source=empty),
            dict(source=garbage),
            dict(source=self.base / "missing.wav"),
            dict(source=self.base),
        )
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises((ProfileError, OSError)):
                    self.create(**case)
        self.assertFalse(self.root.exists() and any(self.root.iterdir()))
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_failed_copy_removes_only_partial_directory(self):
        self.root.mkdir()
        unrelated = self.root / "notes.txt"
        unrelated.write_text("keep", encoding="utf-8")
        original = os.fsync

        def broken(fd):
            raise OSError("disk full")

        os.fsync = broken
        try:
            with self.assertRaisesRegex(ProfileError, "disk full"):
                self.create()
        finally:
            os.fsync = original
        self.assertEqual([p.name for p in self.root.iterdir()], ["notes.txt"])
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_damaged_profiles_are_listed_not_hidden(self):
        good = self.create()
        broken = self.create(name="坏档案")
        (broken.directory / "profile.json").write_text("{", encoding="utf-8")
        (self.root / "unrelated").mkdir()
        listed = {profile.profile_id: profile for profile in self.store.list()}
        self.assertEqual(set(listed), {good.profile_id, broken.profile_id})
        self.assertFalse(listed[good.profile_id].damaged)
        self.assertTrue(listed[broken.profile_id].damaged)
        self.assertFalse(listed[broken.profile_id].supports("tts"))
        self.assertIn("档案损坏", capability_text(listed[broken.profile_id]))

    def test_delete_removes_owned_files_only_and_reports_residuals(self):
        clean = self.create()
        result = self.store.delete(clean)
        self.assertEqual(set(result.removed), {"reference.wav", "profile.json", ".owner"})
        self.assertEqual(result.residual, ())
        self.assertFalse(clean.directory.exists())

        kept = self.create(name="留存")
        extra = kept.directory / "user-note.txt"
        extra.write_text("mine", encoding="utf-8")
        result = self.store.delete(kept)
        self.assertEqual(result.residual, ("user-note.txt",))
        self.assertEqual(extra.read_text(encoding="utf-8"), "mine")
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_delete_refuses_unowned_directories(self):
        profile = self.create()
        (profile.directory / ".owner").unlink()
        with self.assertRaises(ProfileError):
            self.store.delete(profile)
        self.assertTrue(profile.reference_path.exists())

        outside = self.base / f"voice-{'0' * 32}"
        outside.mkdir()
        (outside / ".owner").write_text("voice-changer voice profile v1\n", encoding="utf-8")
        foreign = type(profile)(profile_id="0" * 32, name="x", directory=outside)
        with self.assertRaises(ProfileError):
            self.store.delete(foreign)
        self.assertTrue((outside / ".owner").exists())

    def test_validate_name_and_default_root(self):
        self.assertEqual(validate_name(" 甲 "), "甲")
        with self.assertRaises(ProfileError):
            validate_name("a\x7fb")
        original = os.environ.get("XDG_DATA_HOME")
        os.environ["XDG_DATA_HOME"] = str(self.base)
        try:
            self.assertEqual(default_root(), self.base / "voice-changer" / "voices")
        finally:
            if original is None:
                os.environ.pop("XDG_DATA_HOME")
            else:
                os.environ["XDG_DATA_HOME"] = original


class AvailabilityTest(unittest.TestCase):
    def test_features_are_explicitly_unavailable_without_models(self):
        for feature in ("tts", "vc"):
            with self.subTest(feature=feature):
                state = availability(feature)
                self.assertFalse(state.available)
                self.assertIn("不可用", state.reason)
                self.assertIn("unavailable", state.reason)
        self.assertIn("直通", availability("vc").reason)
        self.assertEqual(preset_voices(), ())

    def test_ai_core_does_not_import_model_runtimes(self):
        code = (
            "import sys, voice_changer.ai.availability, voice_changer.ai.profiles\n"
            "bad = [m for m in ('numpy', 'onnxruntime', 'torch', 'PySide6') if m in sys.modules]\n"
            "print(','.join(bad))\n"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        )
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
