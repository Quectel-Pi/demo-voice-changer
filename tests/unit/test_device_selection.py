"""In-memory selection snapshots only; no discovery, Qt or physical devices."""

from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path
import subprocess
import sys
import unittest

from voice_changer.app.device_selection import (
    DeviceSelection, DeviceSelectionError, SelectionInvalidatedError,
)
from voice_changer.audio.device_discovery import DeviceInfo
from voice_changer.audio.device_types import DeviceRef, StreamConfig


ROOT = Path(__file__).resolve().parents[2]


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.devices = (
            DeviceInfo(DeviceRef(0, "duplex", 0), "API A", 2, 2, 48000.0),
            DeviceInfo(DeviceRef(1, "same name", 0), "API A", 1, 0, 48000.0),
            DeviceInfo(DeviceRef(2, "same name", 1), "API B", 0, 2, 48000.0),
        )
        self.model = DeviceSelection()
        self.model.refresh(self.devices)
        self.config_fields = dict(samplerate=48000, blocksize=512, channels=1, output_ceiling=0.1)

    def select_pair(self):
        self.model.select("input", self.devices[1].ref)
        self.model.select("output", self.devices[2].ref)

    def assert_pair(self):
        self.assertEqual(self.model.input_device, self.devices[1].ref)
        self.assertEqual(self.model.output_device, self.devices[2].ref)

    def test_import_and_creation_do_not_load_or_query_device_libraries(self):
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
from voice_changer.app.device_selection import DeviceSelection
from voice_changer.audio.device_discovery import DeviceInfo
from voice_changer.audio.device_types import DeviceRef
model = DeviceSelection()
assert model.devices == ()
assert model.input_device is None and model.output_device is None
ref = DeviceRef(0, "synthetic duplex", 0)
model.refresh((DeviceInfo(ref, "synthetic API", 2, 2, 48000.0),))
model.select("input", ref)
model.select("output", ref)
config = model.build_stream_config(samplerate=44100, blocksize=256, channels=2, output_ceiling=0.1)
assert config.input_device == ref and config.output_device == ref
assert config.samplerate == 44100 and config.channels == 2
print("created without devices")
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("created without devices", result.stdout)

    def test_default_and_refresh_never_choose_devices(self):
        self.assertEqual(self.model.devices, self.devices)
        self.assertIsNone(self.model.input_device)
        self.assertIsNone(self.model.output_device)
        self.assertFalse(self.model.locked)
        self.model.refresh(self.devices)
        self.assertIsNone(self.model.input_device)
        self.assertIsNone(self.model.output_device)

    def test_config_uses_explicit_fields_and_current_independent_refs_without_mutation(self):
        self.select_pair()
        config = self.model.build_stream_config(**self.config_fields)
        self.assertIsInstance(config, StreamConfig)
        self.assertIs(config.input_device, self.devices[1].ref)
        self.assertIs(config.output_device, self.devices[2].ref)
        for field, value in self.config_fields.items():
            self.assertEqual(getattr(config, field), value)
        self.assert_pair()
        self.assertEqual(self.model.devices, self.devices)
        self.assertFalse(self.model.locked)

    def test_config_requires_both_explicit_selections_without_default_fallback(self):
        for selected in ((), ("input",), ("output",)):
            with self.subTest(selected=selected):
                model = DeviceSelection()
                model.refresh(self.devices)
                for kind in selected:
                    model.select(kind, self.devices[0].ref)
                before = (model.devices, model.input_device, model.output_device)
                with self.assertRaisesRegex(DeviceSelectionError, "Both input and output"):
                    model.build_stream_config(**self.config_fields)
                self.assertEqual((model.devices, model.input_device, model.output_device), before)
        with self.assertRaises(DeviceSelectionError):
            DeviceSelection().build_stream_config(**self.config_fields)

    def test_config_all_fields_are_required_without_implicit_format_or_ceiling(self):
        self.select_pair()
        for field in self.config_fields:
            fields = self.config_fields.copy()
            del fields[field]
            with self.subTest(missing=field), self.assertRaises(TypeError):
                self.model.build_stream_config(**fields)
            self.assert_pair()

    def test_config_reuses_strict_field_validation_without_changing_selection(self):
        self.select_pair()
        invalid = {
            "samplerate": (None, 0, -1, True, 48000.0, "48000"),
            "blocksize": (None, 0, -1, False, 512.0),
            "channels": (None, 0, -1, True, 1.0),
            "output_ceiling": (
                None, 0, -0.1, 1.01, float("nan"), float("inf"), -float("inf"),
                True, False, "0.1",
            ),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    fields = {**self.config_fields, field: value}
                    with self.assertRaisesRegex(ValueError, field):
                        self.model.build_stream_config(**fields)
                    self.assert_pair()
                    self.assertEqual(self.model.devices, self.devices)
                    self.assertFalse(self.model.locked)

    def test_config_accepts_exact_duplex_channel_capacity_and_ceiling_upper_bound(self):
        for kind in ("input", "output"):
            self.model.select(kind, self.devices[0].ref)
        fields = {**self.config_fields, "channels": 2, "output_ceiling": 1}
        config = self.model.build_stream_config(**fields)
        self.assertEqual(config.input_device, config.output_device)
        self.assertEqual(config.channels, 2)
        self.assertEqual(config.output_ceiling, 1)
        with self.assertRaisesRegex(DeviceSelectionError, "input.*too few channels"):
            self.model.build_stream_config(**{**fields, "channels": 3})

    def test_config_checks_both_direction_capacities_not_total_or_opposite_channels(self):
        for kind in ("input", "output"):
            with self.subTest(limited_direction=kind):
                info = replace(self.devices[0], **{f"max_{kind}_channels": 1})
                self.model.refresh((info,))
                for selected_kind in ("input", "output"):
                    self.model.select(selected_kind, info.ref)
                self.assertEqual(
                    self.model.build_stream_config(**self.config_fields).channels, 1,
                )
                with self.assertRaisesRegex(DeviceSelectionError, f"{kind}.*too few channels"):
                    self.model.build_stream_config(**{**self.config_fields, "channels": 2})
                self.assertEqual(self.model.input_device, info.ref)
                self.assertEqual(self.model.output_device, info.ref)

    def test_config_busy_or_owned_devices_reject_until_explicit_release(self):
        self.select_pair()
        for busy, owns in ((True, False), (False, True), (True, True)):
            with self.subTest(busy=busy, owns_devices=owns):
                self.model.set_activity(busy=busy, owns_devices=owns)
                with self.assertRaisesRegex(DeviceSelectionError, "locked"):
                    self.model.build_stream_config(**self.config_fields)
                self.assert_pair()
                self.assertTrue(self.model.locked)
        self.model.set_activity(busy=False, owns_devices=False)
        self.assertEqual(self.model.build_stream_config(**self.config_fields).channels, 1)

    def test_config_explicit_samplerate_does_not_negotiate_reported_defaults(self):
        self.select_pair()
        changed = replace(self.devices[1], default_samplerate=44100.0)
        self.model.refresh((self.devices[0], changed, self.devices[2]))
        config = self.model.build_stream_config(**{**self.config_fields, "samplerate": 32000})
        self.assertEqual(config.samplerate, 32000)
        self.assert_pair()

    def test_config_uses_latest_nonzero_capacity_and_old_config_remains_frozen(self):
        for kind in ("input", "output"):
            self.model.select(kind, self.devices[0].ref)
        config = self.model.build_stream_config(**{**self.config_fields, "channels": 2})
        self.model.refresh((replace(self.devices[0], max_output_channels=1),))
        with self.assertRaisesRegex(DeviceSelectionError, "output.*too few channels"):
            self.model.build_stream_config(**{**self.config_fields, "channels": 2})
        self.assertEqual(self.model.build_stream_config(**self.config_fields).channels, 1)
        self.assertEqual(config.channels, 2)
        with self.assertRaises(FrozenInstanceError):
            config.channels = 1
        self.model.select("input", None)
        self.assertEqual(config.input_device, self.devices[0].ref)
        with self.assertRaises(DeviceSelectionError):
            self.model.build_stream_config(**self.config_fields)

    def test_config_after_identity_change_requires_explicit_reselection(self):
        for changed_ref in (
            DeviceRef(3, "duplex", 0), DeviceRef(0, "renamed duplex", 0),
            DeviceRef(0, "duplex", 1),
        ):
            with self.subTest(ref=changed_ref):
                model = DeviceSelection()
                model.refresh((self.devices[0],))
                for kind in ("input", "output"):
                    model.select(kind, self.devices[0].ref)
                config = model.build_stream_config(**self.config_fields)
                changed = replace(self.devices[0], ref=changed_ref)
                with self.assertRaises(SelectionInvalidatedError):
                    model.refresh((changed,))
                with self.assertRaises(DeviceSelectionError):
                    model.build_stream_config(**self.config_fields)
                with self.assertRaises(DeviceSelectionError):
                    model.select("input", config.input_device)
                for kind in ("input", "output"):
                    model.select(kind, changed_ref)
                self.assertEqual(
                    model.build_stream_config(**self.config_fields).input_device, changed_ref,
                )
                self.assertEqual(config.input_device, self.devices[0].ref)

    def test_input_and_output_are_explicit_and_independent(self):
        self.model.select("input", self.devices[1].ref)
        self.assertEqual(self.model.input_device, self.devices[1].ref)
        self.assertIsNone(self.model.output_device)
        self.model.select("output", self.devices[2].ref)
        self.model.select("input", self.devices[0].ref)
        self.assertEqual(self.model.input_device, self.devices[0].ref)
        self.assertEqual(self.model.output_device, self.devices[2].ref)

    def test_duplex_device_can_be_explicitly_selected_for_both_directions(self):
        self.model.select("input", self.devices[0].ref)
        self.model.select("output", self.devices[0].ref)
        self.assertEqual(self.model.input_device, self.model.output_device)

    def test_explicit_clear_changes_only_selected_direction(self):
        self.select_pair()
        self.model.select("output", None)
        self.assertEqual(self.model.input_device, self.devices[1].ref)
        self.assertIsNone(self.model.output_device)
        self.model.select("input", None)
        self.assertIsNone(self.model.input_device)

    def test_invalid_reference_or_direction_preserves_last_valid_selection(self):
        self.select_pair()
        for kind, ref in (
            ("input", self.devices[2].ref), ("output", self.devices[1].ref),
            ("input", DeviceRef(99, "same name", 0)),
            ("input", DeviceRef(1, "renamed", 0)),
            ("output", DeviceRef(2, "same name", 0)),
        ):
            with self.subTest(kind=kind, ref=ref), self.assertRaises(DeviceSelectionError):
                self.model.select(kind, ref)
            self.assert_pair()

    def test_invalid_arguments_are_explicit_errors_without_state_changes(self):
        self.select_pair()
        for kind, ref in (
            ("unknown", None), (None, None), ("input", 1), ("output", "default"),
            ("input", True),
        ):
            with self.subTest(kind=kind, ref=ref), self.assertRaises(ValueError):
                self.model.select(kind, ref)
            self.assert_pair()

    def test_invalid_refresh_shape_or_records_preserves_entire_previous_state(self):
        self.select_pair()
        for devices in (None, [], {}, "devices", (None,), (self.devices[0], "bad")):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                self.model.refresh(devices)
            self.assertEqual(self.model.devices, self.devices)
            self.assert_pair()

    def test_duplicate_global_indices_are_rejected_atomically(self):
        self.select_pair()
        duplicate = replace(self.devices[0], ref=DeviceRef(1, "different", 1))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.model.refresh((duplicate, self.devices[1]))
        self.assertEqual(self.model.devices, self.devices)
        self.assert_pair()

    def test_valid_refresh_keeps_supported_refs_and_updates_metadata(self):
        self.select_pair()
        changed = replace(self.devices[1], max_input_channels=2, default_samplerate=44100.0)
        self.model.refresh((self.devices[0], changed, self.devices[2]))
        self.assert_pair()
        self.assertEqual(self.model.devices[1].max_input_channels, 2)
        self.assertEqual(self.model.devices[1].default_samplerate, 44100.0)

    def test_missing_input_refresh_commits_and_reports_only_input_invalidation(self):
        self.select_pair()
        refreshed = (self.devices[0], self.devices[2])
        with self.assertRaises(SelectionInvalidatedError) as raised:
            self.model.refresh(refreshed)
        self.assertEqual(raised.exception.invalidated, ("input",))
        self.assertEqual(self.model.devices, refreshed)
        self.assertIsNone(self.model.input_device)
        self.assertEqual(self.model.output_device, self.devices[2].ref)
        self.assertIn("No automatic fallback", str(raised.exception))
        self.model.select("input", self.devices[0].ref)
        self.assertEqual(self.model.input_device, self.devices[0].ref)

    def test_output_capability_loss_is_visible_without_changing_input(self):
        self.select_pair()
        removed_output = replace(self.devices[2], max_output_channels=0)
        with self.assertRaises(SelectionInvalidatedError) as raised:
            self.model.refresh((self.devices[0], self.devices[1], removed_output))
        self.assertEqual(raised.exception.invalidated, ("output",))
        self.assertIsNone(self.model.output_device)
        self.assertEqual(self.model.input_device, self.devices[1].ref)

    def test_empty_refresh_reports_both_losses_and_does_not_auto_restore(self):
        self.select_pair()
        with self.assertRaises(SelectionInvalidatedError) as raised:
            self.model.refresh(())
        self.assertEqual(raised.exception.invalidated, ("input", "output"))
        self.assertEqual(self.model.devices, ())
        self.assertIsNone(self.model.input_device)
        self.assertIsNone(self.model.output_device)
        self.model.refresh(self.devices)
        self.assertIsNone(self.model.input_device)
        self.assertIsNone(self.model.output_device)

    def test_reordered_same_names_never_migrate_selection_to_new_indices(self):
        self.select_pair()
        input_replacement = replace(self.devices[1], ref=DeviceRef(3, "same name", 0))
        output_replacement = replace(self.devices[2], ref=DeviceRef(4, "same name", 1))
        with self.assertRaises(SelectionInvalidatedError):
            self.model.refresh((self.devices[0], input_replacement, output_replacement))
        self.assertIsNone(self.model.input_device)
        self.assertIsNone(self.model.output_device)

    def test_busy_or_owned_devices_lock_selection_clear_and_manual_refresh(self):
        self.select_pair()
        for busy, owns in ((True, False), (False, True), (True, True)):
            with self.subTest(busy=busy, owns_devices=owns):
                self.model.set_activity(busy=busy, owns_devices=owns)
                self.assertTrue(self.model.locked)
                for kind in ("input", "output"):
                    with self.assertRaises(DeviceSelectionError):
                        self.model.select(kind, None)
                    with self.assertRaises(DeviceSelectionError):
                        self.model.select(kind, self.devices[0].ref)
                with self.assertRaises(DeviceSelectionError):
                    self.model.refresh(())
                self.assert_pair()
                self.assertEqual(self.model.devices, self.devices)
        self.model.set_activity(busy=False, owns_devices=False)
        self.model.select("input", self.devices[0].ref)
        self.assertEqual(self.model.input_device, self.devices[0].ref)

    def test_cleanup_failure_remains_locked_after_operation_finishes(self):
        self.model.set_activity(busy=True, owns_devices=True)
        self.model.set_activity(busy=False, owns_devices=True)
        with self.assertRaises(DeviceSelectionError):
            self.model.refresh(self.devices)
        self.model.set_activity(busy=False, owns_devices=False)
        self.model.refresh(self.devices)
        self.assertFalse(self.model.locked)

    def test_activity_facts_reject_truthy_defaults_without_partial_unlock(self):
        self.model.set_activity(busy=True, owns_devices=True)
        for busy, owns in ((None, False), (False, None), (False, 1), (1, False), ("yes", False)):
            with self.subTest(busy=busy, owns_devices=owns), self.assertRaises(ValueError):
                self.model.set_activity(busy=busy, owns_devices=owns)
            self.assertTrue(self.model.locked)


if __name__ == "__main__":
    unittest.main()
