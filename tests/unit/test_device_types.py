"""Shared type contracts and legacy exports; no NumPy or actual device library."""

from dataclasses import FrozenInstanceError, fields, replace
import os
from pathlib import Path
import subprocess
import sys
import unittest

from voice_changer.audio.device_types import DeviceRef, StreamConfig
from voice_changer.audio import sounddevice_backend


ROOT = Path(__file__).resolve().parents[2]


class DeviceTypeTests(unittest.TestCase):
    def setUp(self):
        self.config = StreamConfig(
            DeviceRef(3, "explicit input", 1), DeviceRef(7, "explicit output", 1),
            samplerate=48000, blocksize=512, channels=2, output_ceiling=0.1,
        )

    def test_types_import_and_construct_without_backend_or_heavy_libraries(self):
        script = """
import importlib.abc
import sys
class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("sounddevice", "numpy", "PySide6",
                                "voice_changer.audio.sounddevice_backend",
                                "voice_changer.dsp", "voice_changer.ai")):
            raise AssertionError("Unexpected import: " + fullname)
sys.meta_path.insert(0, NoBackend())
from voice_changer.audio.device_types import DeviceRef, StreamConfig
ref = DeviceRef(0, "explicit", 0)
config = StreamConfig(ref, ref, 48000, 512, 1, 0.1)
assert config.input_device is ref
print("types constructed without backend")
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("types constructed without backend", result.stdout)

    def test_legacy_backend_exports_are_the_same_classes(self):
        self.assertIs(sounddevice_backend.DeviceRef, DeviceRef)
        self.assertIs(sounddevice_backend.StreamConfig, StreamConfig)
        legacy = sounddevice_backend.StreamConfig(
            self.config.input_device, self.config.output_device, 48000, 512, 2, 0.1,
        )
        self.assertEqual(legacy, self.config)
        self.assertIsInstance(legacy.input_device, sounddevice_backend.DeviceRef)

    def test_field_annotation_and_pattern_matching_shapes_are_preserved(self):
        self.assertEqual(
            [(field.name, field.type) for field in fields(DeviceRef)],
            [("index", "int"), ("name", "str"), ("hostapi", "int")],
        )
        names = ("input_device", "output_device", "samplerate", "blocksize", "channels",
                 "output_ceiling")
        self.assertEqual(StreamConfig.__match_args__, names)
        self.assertEqual(DeviceRef.__match_args__, ("index", "name", "hostapi"))
        self.assertEqual(
            [field.type for field in fields(StreamConfig)],
            ["DeviceRef", "DeviceRef", "int", "int", "int", "float"],
        )

    def test_original_device_identity_guards_and_messages_are_preserved(self):
        for values, message in (
            ((-1, "input", 1), "Device index"),
            ((True, "input", 1), "Device index"),
            ((None, "input", 1), "Device index"),
            ((3.0, "input", 1), "Device index"),
            ((3, "", 1), "Device name"),
            ((3, " ", 1), "Device name"),
            ((3, "input", -1), "Host API index"),
            ((3, "input", True), "Host API index"),
            ((3, "input", 1.0), "Host API index"),
        ):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                DeviceRef(*values)

    def test_original_explicit_format_and_ceiling_guards_are_preserved(self):
        invalid = {
            "input_device": (None, "default"),
            "output_device": (None,),
            "samplerate": (0, -1, True, 48000.0),
            "blocksize": (0, -1, False, 512.0),
            "channels": (0, -1, True, 2.0),
            "output_ceiling": (0, -0.1, 1.01, float("nan"), float("inf"), True, "0.1"),
        }
        for key, values in invalid.items():
            for value in values:
                with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                    replace(self.config, **{key: value})
        self.assertEqual(replace(self.config, output_ceiling=1).output_ceiling, 1)

    def test_refs_and_configs_remain_frozen_and_replace_does_not_mutate(self):
        with self.assertRaises(FrozenInstanceError):
            self.config.channels = 1
        with self.assertRaises(FrozenInstanceError):
            self.config.input_device.index = 0
        changed = replace(self.config, blocksize=256)
        self.assertEqual(changed.blocksize, 256)
        self.assertEqual(self.config.blocksize, 512)
        self.assertEqual(changed.input_device, self.config.input_device)


if __name__ == "__main__":
    unittest.main()
