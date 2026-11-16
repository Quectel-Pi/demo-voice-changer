"""Discovery uses only an in-test device library; no NumPy, Qt or PortAudio."""

from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from voice_changer.audio.device_discovery import DeviceInfo, discover_devices
from voice_changer.audio.sounddevice_backend import DeviceRef, SoundDeviceBackend, StreamConfig


ROOT = Path(__file__).resolve().parents[2]


class FakeDeviceLibrary(ModuleType):
    def __init__(self):
        super().__init__("sounddevice")
        self.calls = []
        self.devices = [
            dict(name="duplex", hostapi=1, max_input_channels=2,
                 max_output_channels=2, default_samplerate=48000.0),
            dict(name="same name", hostapi=0, max_input_channels=1,
                 max_output_channels=0, default_samplerate=44100),
            dict(name="same name", hostapi=1, max_input_channels=0,
                 max_output_channels=2, default_samplerate=48000.0),
            dict(name="no channels", hostapi=0, max_input_channels=0,
                 max_output_channels=0, default_samplerate=48000.0),
        ]
        self.hostapis = [
            dict(name="API A", devices=[1, 3], default_input_device=1, default_output_device=-1),
            dict(name="API B", devices=[2, 0], default_input_device=0, default_output_device=2),
        ]
        self.device_error = None
        self.host_error = None

    def query_devices(self):
        self.calls.append("devices")
        if self.device_error is not None:
            raise self.device_error
        return self.devices

    def query_hostapis(self):
        self.calls.append("hostapis")
        if self.host_error is not None:
            raise self.host_error
        return self.hostapis

    @property
    def default(self):
        raise AssertionError("Discovery must not choose default devices")

    def Stream(self, **options):
        raise AssertionError("Discovery must not open streams")

    def check_input_settings(self, **options):
        raise AssertionError("Discovery must not probe stream formats")

    def check_output_settings(self, **options):
        raise AssertionError("Discovery must not probe stream formats")


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.library = FakeDeviceLibrary()
        self.patch = patch.dict(sys.modules, {"sounddevice": self.library})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_import_does_not_load_audio_numeric_gui_or_ai_libraries(self):
        script = """
import importlib.abc
import sys
class NoHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("sounddevice", "numpy", "PySide6",
                                "voice_changer.audio.sounddevice_backend",
                                "voice_changer.dsp", "voice_changer.ai")):
            raise AssertionError("Unexpected import: " + fullname)
sys.meta_path.insert(0, NoHeavyImports())
from voice_changer.audio.device_discovery import discover_devices
print("imported without devices")
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("imported without devices", result.stdout)

    def test_complete_snapshot_keeps_global_indices_and_separate_capabilities(self):
        devices = discover_devices()
        self.assertIsInstance(devices, tuple)
        self.assertEqual(devices, (
            DeviceInfo(DeviceRef(0, "duplex", 1), "API B", 2, 2, 48000.0),
            DeviceInfo(DeviceRef(1, "same name", 0), "API A", 1, 0, 44100.0),
            DeviceInfo(DeviceRef(2, "same name", 1), "API B", 0, 2, 48000.0),
            DeviceInfo(DeviceRef(3, "no channels", 0), "API A", 0, 0, 48000.0),
        ))
        self.assertEqual(self.library.calls, ["devices", "hostapis"])

    def test_duplicate_names_are_not_deduplicated_or_selected_as_defaults(self):
        devices = discover_devices()
        self.assertEqual([d.ref.index for d in devices if d.ref.name == "same name"], [1, 2])
        self.assertEqual([d.ref.index for d in devices if d.max_input_channels > 0], [0, 1])
        self.assertEqual([d.ref.index for d in devices if d.max_output_channels > 0], [0, 2])
        self.assertEqual(self.library.calls, ["devices", "hostapis"])

    def test_empty_device_list_is_not_replaced_with_fake_devices(self):
        self.library.devices = []
        self.library.hostapis = []
        self.assertEqual(discover_devices(), ())
        self.assertEqual(self.library.calls, ["devices", "hostapis"])

    def test_empty_host_api_can_coexist_with_an_empty_device_list(self):
        self.library.devices = []
        self.library.hostapis = [dict(name="API A", devices=[])]
        self.assertEqual(discover_devices(), ())

    def test_snapshot_is_frozen_and_detached_from_mutable_library_records(self):
        devices = discover_devices()
        self.library.devices[0]["name"] = "changed"
        self.library.devices[0]["max_input_channels"] = 8
        self.library.hostapis[1]["name"] = "changed API"
        self.library.hostapis[1]["devices"].clear()
        self.assertEqual(devices[0].ref, DeviceRef(0, "duplex", 1))
        self.assertEqual(devices[0].hostapi_name, "API B")
        self.assertEqual(devices[0].max_input_channels, 2)
        with self.assertRaises(FrozenInstanceError):
            devices[0].max_input_channels = 8
        with self.assertRaises(FrozenInstanceError):
            devices[0].ref.index = 1

    def test_refresh_reads_new_results_without_mutating_previous_snapshot(self):
        first = discover_devices()
        self.library.devices[0]["name"] = "replacement"
        second = discover_devices()
        self.assertEqual(first[0].ref.name, "duplex")
        self.assertEqual(second[0].ref.name, "replacement")
        self.assertNotEqual(first[0].ref, second[0].ref)
        self.assertEqual(self.library.calls, ["devices", "hostapis"] * 2)

    def test_missing_dependency_propagates_without_empty_success_fallback(self):
        with patch.dict(sys.modules, {"sounddevice": None}):
            with self.assertRaises(ModuleNotFoundError):
                discover_devices()
        self.assertEqual(self.library.calls, [])

    def test_query_errors_preserve_original_cause_and_allow_explicit_retry(self):
        for attribute in ("device_error", "host_error"):
            with self.subTest(query=attribute):
                cause = OSError("synthetic query failure")
                setattr(self.library, attribute, cause)
                with self.assertRaises(OSError) as raised:
                    discover_devices()
                self.assertIs(raised.exception, cause)
                setattr(self.library, attribute, None)
                self.assertEqual(len(discover_devices()), 4)

    def test_invalid_list_and_record_shapes_are_explicit_errors(self):
        for attribute in ("devices", "hostapis"):
            original = getattr(self.library, attribute)
            for value in (None, {}, "wrong", b"wrong", {1}, [None], ["wrong"]):
                with self.subTest(attribute=attribute, value=value):
                    setattr(self.library, attribute, value)
                    with self.assertRaises(ValueError):
                        discover_devices()
            setattr(self.library, attribute, original)

    def test_missing_or_invalid_device_fields_are_not_defaulted(self):
        invalid = {
            "name": (None, "", " ", True),
            "hostapi": (None, -1, True, 1.0, 20),
            "max_input_channels": (None, -1, True, 2.0),
            "max_output_channels": (None, -1, False, "2"),
            "default_samplerate": (
                None, True, "48000", 0, -1, float("nan"), float("inf"), 10 ** 1000,
            ),
        }
        original = self.library.devices[0].copy()
        for key, values in invalid.items():
            for value in values:
                with self.subTest(field=key, value=value):
                    self.library.devices[0] = dict(original, **{key: value})
                    with self.assertRaisesRegex(ValueError, "Device 0"):
                        discover_devices()
            self.library.devices[0] = {k: v for k, v in original.items() if k != key}
            with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "Device 0"):
                discover_devices()
        self.library.devices[0] = original

    def test_invalid_host_fields_members_and_duplicate_indices_fail(self):
        original = self.library.hostapis[0].copy()
        for key, values in {
            "name": (None, "", " ", True),
            "devices": (None, {}, "wrong", [-1], [True], [1.0], [1, 1], [99]),
        }.items():
            for value in values:
                with self.subTest(field=key, value=value):
                    self.library.hostapis[0] = dict(original, **{key: value})
                    with self.assertRaisesRegex(ValueError, "Host API 0"):
                        discover_devices()
        self.library.hostapis[0] = original

    def test_inconsistent_host_membership_is_not_a_partial_success(self):
        for members in ([1], [0, 1, 3], []):
            with self.subTest(members=members):
                self.library.hostapis[0]["devices"] = members
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    discover_devices()

    def test_refs_feed_existing_explicit_config_without_loading_or_opening_audio(self):
        devices = discover_devices()
        config = StreamConfig(
            devices[1].ref, devices[2].ref, samplerate=48000,
            blocksize=512, channels=1, output_ceiling=0.1,
        )
        backend = SoundDeviceBackend(config)
        with self.assertRaisesRegex(RuntimeError, "confirmation"):
            backend.open_stream()
        self.assertEqual(self.library.calls, ["devices", "hostapis"])

    def test_device_info_rejects_invalid_manually_constructed_metadata(self):
        info = discover_devices()[0]
        for field, values in {
            "ref": (None, 0, "default"),
            "hostapi_name": (None, "", True),
            "max_input_channels": (-1, True, 1.0),
            "max_output_channels": (-1, False, "2"),
            "default_samplerate": (0, True, float("nan"), float("inf"), 10 ** 1000),
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(info, **{field: value})


if __name__ == "__main__":
    unittest.main()
