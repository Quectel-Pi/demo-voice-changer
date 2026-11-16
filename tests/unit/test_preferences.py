import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from voice_changer.app.preferences import AudioSettings, DeviceIdentity, Preferences, PreferencesStore
from voice_changer.audio.device_discovery import DeviceInfo
from voice_changer.audio.device_types import DeviceRef


class PreferencesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = PreferencesStore(self.root / "config/preferences.json")

    def test_missing_file_does_not_create_directories(self):
        self.assertEqual(self.store.load(), Preferences())
        self.assertFalse(self.store.path.parent.exists())

    def test_round_trip_has_no_indices_effects_gains_or_activity(self):
        original = Preferences(
            DeviceIdentity("test input", "test API"), DeviceIdentity("test output", "test API"),
            AudioSettings(48000, 512, 1, 0.05), str(self.root),
        )
        self.store.save(original)
        self.assertEqual(self.store.load(), original)
        payload = json.loads(self.store.path.read_text())
        self.assertEqual(set(payload), {"version", "input_device", "output_device", "audio", "save_directory"})
        self.assertEqual(set(payload["input_device"]), {"name", "hostapi_name"})
        self.assertEqual(self.store.path.stat().st_mode & 0o777, 0o600)

    def test_invalid_schema_types_and_numbers_are_rejected(self):
        self.store.save(Preferences(audio=AudioSettings(48000, 512, 1, 0.05)))
        valid = json.loads(self.store.path.read_text())
        invalid = [
            {}, {**valid, "version": True}, {**valid, "extra": 1},
            {**valid, "audio": {**valid["audio"], "channels": True}},
            {**valid, "audio": {**valid["audio"], "output_ceiling": float("nan")}},
            {**valid, "input_device": {"name": "test", "hostapi_name": "API", "index": 4}},
            {**valid, "save_directory": "relative"},
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                self.store.path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    self.store.load()
        self.store.path.write_bytes(b"x" * 16385)
        with self.assertRaisesRegex(ValueError, "16KiB"):
            self.store.load()
        self.store.path.write_text('{"version": 1, "version": 2}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.store.load()

    def test_failed_atomic_replace_keeps_previous_file_and_cleans_temporary(self):
        self.store.save(Preferences())
        old = self.store.path.read_bytes()
        with patch("voice_changer.app.preferences.os.replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                self.store.save(Preferences(save_directory=str(self.root)))
        self.assertEqual(self.store.path.read_bytes(), old)
        self.assertEqual([p.name for p in self.store.path.parent.iterdir()], ["preferences.json"])

    def test_identity_matches_new_indices_but_rejects_missing_duplicate_or_capability(self):
        identity = DeviceIdentity("test", "API")
        info = DeviceInfo(DeviceRef(99, "test", 7), "API", 1, 2, 48000)
        self.assertEqual(identity.match((info,), "input", 1), info.ref)
        for devices, kind, channels in (((), "input", 1), ((info, info), "input", 1), ((info,), "input", 2)):
            with self.assertRaisesRegex(ValueError, "unavailable or ambiguous"):
                identity.match(devices, kind, channels)
