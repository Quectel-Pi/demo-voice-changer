"""D2 desktop lifecycle using real DSP and temporary synthetic recordings only."""

from ctypes.util import find_library
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import wave

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from voice_changer.audio.engine import EngineState
from voice_changer.audio.media_status import RecordingState
from voice_changer.audio.recording import RecordingWriter
from voice_changer.dsp.parameters import PRESETS
from voice_changer.ui.main_window import MainWindow

from media_fakes import MediaLibrary
from qt_wait import wait_until


NATIVE = os.environ.get("VOICE_CHANGER_RUBBERBAND_LIBRARY") or find_library("rubberband")


@unittest.skipUnless(NATIVE, "Rubber Band unavailable; D2 DSP integration is not verified")
class RecordingDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.environment = patch.dict(os.environ, {
            "XDG_DATA_HOME": str(self.root / "xdg"), "XDG_CONFIG_HOME": str(self.root / "config"),
        })
        self.environment.start()
        self.library = MediaLibrary()
        self.patcher = patch.dict(sys.modules, {"sounddevice": self.library})
        self.patcher.start()
        self.window = MainWindow()
        self.window.show()
        self.addCleanup(self.cleanup)
        self.wait_for(lambda: self.window.controller.is_ready
                      and not self.window.controller.busy and self.window.controller.devices_loaded)

    def wait_for(self, condition, timeout=4):
        self.assertTrue(wait_until(condition, timeout), "Timed out waiting for D2")

    def test_successful_capture_preferences_write_after_release_and_restore_without_audio(self):
        from voice_changer.app.preferences import PreferencesStore, preferences_path
        group = self.record()
        path = preferences_path()
        self.assertFalse(path.exists())
        self.finish()
        self.assertTrue(path.exists())
        saved = PreferencesStore(path).load()
        self.assertIsNotNone(saved.input_device)
        self.assertIsNone(saved.output_device)
        self.assertEqual((saved.audio.samplerate, saved.audio.blocksize, saved.audio.channels), (48000, 512, 1))
        self.close_window()
        streams = len(self.library.streams)
        self.window = MainWindow()
        self.window.show()
        self.wait_for(lambda: self.window.controller.is_ready
                      and not self.window.controller.busy and self.window.controller.devices_loaded)
        self.assertEqual(len(self.library.streams), streams)
        self.assertIsNotNone(self.window.controller.device_selection.input_device)
        self.assertEqual(self.window.preset_combo.currentData(), "original")
        self.assertEqual(self.window.input_gain_input.value(), 0)
        self.assertEqual(self.window.output_gain_input.value(), 0)
        self.window.refresh_button.click()
        self.idle()
        self.assertIsNotNone(self.window.controller.device_selection.input_device)
        self.assertEqual(len(self.library.streams), streams)

    def test_preferences_write_failure_is_visible_after_audio_release_and_retry_preserves_recording(self):
        group = self.record()
        with patch(
            "voice_changer.app.preferences.PreferencesStore.save",
            side_effect=OSError("synthetic preferences write failure"),
        ):
            self.finish()
        self.assertFalse(self.window.controller.owns_devices)
        self.assertTrue(group.complete)
        self.assertIn("synthetic preferences write failure", self.window.error_notice.text())
        self.assertTrue(group.original_path.exists())
        self.assertTrue(self.window.controller.request_stop_all())
        self.idle()
        from voice_changer.app.preferences import preferences_path
        self.assertTrue(preferences_path().exists())
        self.assertTrue(group.original_path.exists())

    def idle(self):
        self.wait_for(lambda: not self.window.controller.busy)

    def close_window(self):
        controller = self.window.controller
        if not controller.is_closed:
            self.idle()
            self.assertTrue(controller.request_stop_all())
            self.idle()
            if controller.media_status.group is not None:
                self.assertTrue(controller.request_discard_recording())
                self.idle()
            self.window._begin_shutdown_now()
            self.wait_for(lambda: controller.is_closed)
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def cleanup(self):
        for raw in self.library.streams:
            raw.close_error = None
        self.close_window()
        self.patcher.stop()
        self.environment.stop()
        self.temp.cleanup()
        self.assertFalse(any(
            thread.name in (
                "voice-changer-dsp", "voice-changer-analysis",
                "synthetic-media-device", "recording-disk-writer",
            )
            for thread in threading.enumerate()
        ))

    def configure(self, *, output=False):
        self.window.refresh_button.click()
        self.idle()
        self.window.input_combo.setFocus()
        QTest.keyClick(self.window.input_combo, Qt.Key.Key_Down)
        if output:
            self.window.output_combo.setFocus()
            QTest.keyClick(self.window.output_combo, Qt.Key.Key_Down)
        for field, value in zip(self.window.config_inputs, ("48000", "512", "1", "0.05")):
            field.setText(value)

    def record(self, *, output=False):
        self.configure(output=output)
        self.window.record_button.click()
        self.wait_for(lambda: self.window.controller.media_status.recording and not self.window.controller.busy)
        self.wait_for(lambda: self.window.controller.media_status.duration >= 0.15)
        return self.window.controller.media_status.group

    def finish(self):
        self.window.record_button.click()
        self.wait_for(lambda: not self.window.controller.media_status.recording and not self.window.controller.busy)
        self.assertEqual(self.window.controller.media_status.recording_state, RecordingState.UNSAVED)
        return self.window.controller.media_status.group

    def monitor(self):
        with patch.object(QMessageBox, "exec", side_effect=AssertionError("unexpected dialog")):
            self.window.listen_button.click()
        self.wait_for(lambda: self.window.controller.media_status.monitoring and not self.window.controller.busy)

    def select_preset(self, key):
        button = self.window.preset_buttons[key]
        button.setFocus()
        QTest.keyClick(button, Qt.Key.Key_Space)
        self.assertEqual(self.window.preset_combo.currentData(), key)

    def test_fun_presets_custom_reset_preserve_gains_and_startup_defaults(self):
        self.assertEqual(self.window._effect_parameters(), PRESETS["original"])
        self.assertEqual(self.window.input_gain_input.value(), 0)
        self.assertEqual(self.window.output_gain_input.value(), 0)
        self.select_preset("robot")
        self.select_preset("custom")
        self.window.input_gain_input.setValue(6)
        self.window.output_gain_input.setValue(-6)
        self.assertEqual(self.window.preset_combo.currentData(), "custom")
        self.window.echo_input.setValue(15)
        self.assertEqual(self.window.preset_combo.currentData(), "custom")
        self.assertIn("机器人", self.window.preset_combo.currentText())
        self.assertEqual(self.window._effect_parameters().echo_mix, 0.15)
        self.window.reset_effect_button.setFocus()
        QTest.keyClick(self.window.reset_effect_button, Qt.Key.Key_Space)
        parameters = self.window._effect_parameters()
        self.assertEqual(self.window.preset_combo.currentData(), "robot")
        self.assertEqual(parameters.robot_mix, 0.8)
        self.assertEqual(parameters.echo_mix, 0)
        self.assertEqual((parameters.input_gain_db, parameters.output_gain_db), (6, -6))
        self.select_preset("original")
        parameters = self.window._effect_parameters()
        self.assertTrue(parameters.bypass)
        self.assertEqual((parameters.input_gain_db, parameters.output_gain_db), (6, -6))
        self.select_preset("custom")
        self.window.resize(480, 400)
        self.app.processEvents()
        self.assertEqual(self.window.scroll_area.horizontalScrollBar().maximum(), 0)
        for control in (*self.window._sound_inputs, *self.window._gain_inputs):
            self.assertTrue(control.accessibleName())
        controls = set((*self.window._sound_inputs, *self.window._gain_inputs))
        current = self.window.preset_buttons["custom"]
        current.setFocus()
        visited = set()
        for _ in range(60):
            QTest.keyClick(self.app.focusWidget(), Qt.Key.Key_Tab)
            self.app.processEvents()
            focused = self.app.focusWidget()
            if focused in controls:
                viewport = self.window.scroll_area.viewport()
                bounds = focused.rect().translated(
                    focused.mapTo(viewport, QPoint(0, 0)),
                )
                self.assertTrue(
                    viewport.rect().contains(bounds),
                    f"{focused.accessibleName()} is focused but not visible: "
                    f"viewport={viewport.rect()}, bounds={bounds}, "
                    f"scrollbar={self.window.scroll_area.verticalScrollBar().value()}",
                )
            if focused is current or focused in visited:
                break
            visited.add(focused)
        self.assertTrue(controls.issubset(visited))

    def test_fun_effects_gains_record_monitor_save_and_play_without_reprocessing(self):
        self.configure(output=True)
        self.select_preset("robot")
        self.window.input_gain_input.setValue(12)
        self.window.output_gain_input.setValue(12)
        self.window.record_button.click()
        self.wait_for(lambda: self.window.controller.media_status.recording and not self.window.controller.busy)
        self.monitor()
        for index, key in enumerate(("robot", "echo", "reverb", "loli", "uncle", "monster")):
            self.select_preset(key)
            control = self.window.controller._live_control
            self.wait_for(lambda: control.applied_revision == control.requested[0])
            self.assertEqual(control.requested[1].input_gain_db, 12)
            self.assertEqual(control.requested[1].output_gain_db, 12)
            if index == 2:
                self.window.stop_button.click()
                self.idle()
                self.assertTrue(self.window.controller.media_status.recording)
                self.assertFalse(self.window.controller.media_status.monitoring)
        monitor = self.library.streams[1]
        self.assertGreater(monitor.output_peak, 0.04)
        self.assertLessEqual(monitor.output_peak, 0.05)
        self.assertFalse(monitor.output_nonfinite)
        group = self.finish()
        metadata = json.loads(group.metadata_path.read_text())
        journal_bytes = (group.directory / "parameters.jsonl").read_bytes()
        events = [json.loads(line) for line in journal_bytes.decode().splitlines()]
        for preset in ("robot", "echo", "reverb", "loli", "uncle", "monster"):
            expected = PRESETS[preset]
            self.assertTrue(any(
                all(event["parameters"][name] == getattr(expected, name) for name in (
                    "pitch_semitones", "formant_scale", "robot_mix", "echo_mix", "reverb_mix",
                )) for event in events
            ), preset)
        self.assertTrue(all(
            event["parameters"]["input_gain_db"] == 12 and event["parameters"]["output_gain_db"] == 12
            for event in events
        ))
        with wave.open(str(group.original_path), "rb") as original, wave.open(str(group.effect_path), "rb") as effect:
            raw = np.frombuffer(original.readframes(group.frames), dtype="<i2").astype(np.float32) / 32768
            wet = np.frombuffer(effect.readframes(group.frames), dtype="<i2").astype(np.float32) / 32768
        positions = np.arange(group.frames) + metadata["source_interval"]["start"]
        expected = 0.02 * np.sin(2 * np.pi * 220 * positions / 48000)
        np.testing.assert_allclose(raw, expected, rtol=0, atol=1 / 32768)
        self.assertLessEqual(float(np.abs(wet).max()), 0.05)
        self.assertGreater(float(np.abs(wet).max()), float(np.abs(raw).max()) * 2)
        self.window.input_gain_input.setValue(-24)
        self.window.output_gain_input.setValue(-24)
        with self.answer(QMessageBox.StandardButton.Yes):
            self.window.play_effect_button.click()
        self.wait_for(lambda: self.window.controller.media_status.playback == "effect" and not self.window.controller.busy)
        playback = self.library.streams[-1]
        self.wait_for(lambda: playback.output_peak > 0.04)
        self.assertLessEqual(playback.output_peak, 0.05)
        self.assertFalse(playback.output_nonfinite)
        self.assertFalse(self.window.input_gain_input.isEnabled())
        self.assertFalse(self.window.reset_effect_button.isEnabled())
        self.window.stop_button.click()
        self.idle()
        destination = self.root / "d3-saved"
        with patch("voice_changer.ui.main_window.QFileDialog.getSaveFileName", return_value=(str(destination), "")):
            self.window.save_recording_button.click()
            self.idle()
        self.assertTrue(self.window.controller.media_status.group.saved)
        self.assertEqual(self.window.controller.preferences.save_directory, str(destination.parent))
        self.assertEqual(
            (destination / "parameters.jsonl").read_bytes(),
            journal_bytes,
        )
        self.window.close()
        self.wait_for(lambda: self.window.controller.is_closed)

    def answer(self, button):
        def respond(dialog):
            self.assertEqual(dialog.defaultButton(), dialog.button(QMessageBox.StandardButton.Cancel))
            return button
        return patch.object(QMessageBox, "exec", respond)

    def test_input_only_recording_and_compact_keyboard_stop(self):
        self.configure()
        self.window.resize(480, 400)
        self.app.processEvents()
        for button in (self.window.record_button, self.window.stop_button, self.window.close_button):
            top = button.mapTo(self.window, QPoint())
            bottom = button.mapTo(self.window, button.rect().bottomRight())
            self.assertTrue(self.window.rect().contains(top))
            self.assertTrue(self.window.rect().contains(bottom))
        self.window.record_button.setFocus()
        QTest.keyClick(self.window.record_button, Qt.Key.Key_Space)
        self.wait_for(lambda: self.window.controller.media_status.duration >= 0.15)
        self.assertIn("monitoring off", self.window.audio_status.text())
        self.assertIn("monitoring off", self.window.activity_status.text())
        self.assertFalse(self.window.play_original_button.isEnabled())
        self.assertEqual([raw.kind for raw in self.library.streams], ["input"])
        QTest.keyClick(self.window.record_button, Qt.Key.Key_Space)
        self.wait_for(lambda: not self.window.controller.busy and self.window.controller.state is EngineState.STOPPED)
        self.assertTrue(self.window.controller.media_status.group.complete)
        self.assertTrue(self.library.streams[0].closed)

    def test_monitor_toggle_preserves_single_input_and_recording(self):
        self.record(output=True)
        source = self.library.streams[0]
        self.wait_for(lambda: bool(self.window.input_level.history))
        self.wait_for(lambda: all(
            meter.pitch_history for meter in (self.window.input_level, self.window.output_level)
        ))
        source_id = self.window.controller.media_status.meter_source_id
        with patch.object(QMessageBox, "exec", side_effect=AssertionError("unexpected dialog")):
            self.window.listen_button.click()
        self.wait_for(lambda: self.window.controller.media_status.monitoring and not self.window.controller.busy)
        self.assertIn("Listening and recording", self.window.activity_status.text())
        self.assertEqual(len(self.library.streams), 2)
        history_before_stop = self.window.input_level.history
        pitches_before_stop = tuple(
            meter.pitch_history for meter in (self.window.input_level, self.window.output_level)
        )
        self.window.stop_button.click()
        self.idle()
        self.assertTrue(self.window.controller.media_status.recording)
        self.assertFalse(self.window.controller.media_status.monitoring)
        self.assertEqual(self.window.controller.media_status.meter_source_id, source_id)
        self.assertTrue(set(history_before_stop) & set(self.window.input_level.history))
        for before, meter in zip(
            pitches_before_stop, (self.window.input_level, self.window.output_level),
        ):
            self.assertTrue(set(before) & set(meter.pitch_history))
        self.assertIn("monitoring off", self.window.activity_status.text())
        self.assertTrue(source.active)
        self.monitor()
        self.finish()
        self.assertIn("Listening", self.window.activity_status.text())
        self.assertTrue(self.window.controller.media_status.monitoring)
        self.assertTrue(source.active)
        self.assertEqual(sum(raw.kind == "input" for raw in self.library.streams), 1)

    def test_listen_then_record_releases_duplex_before_input_owned_session(self):
        self.configure(output=True)
        self.window.validate_button.click()
        self.monitor()
        duplex = self.library.streams[0]
        self.window.record_button.click()
        self.wait_for(lambda: self.window.controller.media_status.recording and not self.window.controller.busy)
        self.assertTrue(duplex.closed)
        self.assertEqual([raw.kind for raw in self.library.streams], ["duplex", "input", "output"])
        events = [(operation, kind) for operation, kind, ident in self.library.operations]
        self.assertLess(events.index(("close", "duplex")), events.index(("create", "input")))
        self.wait_for(lambda: self.window.controller.media_status.duration >= 0.15)
        self.finish()
        self.assertTrue(self.window.controller.media_status.monitoring)

    def test_recorded_parameters_and_source_interval_follow_effect_changes(self):
        group = self.record()
        self.window.preset_buttons["male_to_female"].setFocus()
        QTest.keyClick(self.window.preset_buttons["male_to_female"], Qt.Key.Key_Space)
        self.wait_for(lambda: self.window.controller.media_status.duration >= 0.35)
        self.finish()
        metadata = json.loads(group.metadata_path.read_text())
        interval = metadata["source_interval"]
        self.assertEqual(interval["end_exclusive"] - interval["start"], group.frames)
        events = [json.loads(line) for line in (group.directory / "parameters.jsonl").read_text().splitlines()]
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[0]["source_frame"], interval["start"])
        self.assertEqual([event["revision"] for event in events], sorted(event["revision"] for event in events))
        self.assertTrue(all(interval["start"] <= event["source_frame"] < interval["end_exclusive"] for event in events))
        self.assertTrue(any(not event["parameters"]["bypass"] for event in events))
        with wave.open(str(group.original_path), "rb") as original, wave.open(str(group.effect_path), "rb") as effect:
            raw = original.readframes(group.frames)
            processed = effect.readframes(group.frames)
            self.assertEqual(len(raw), len(processed))
            self.assertNotEqual(raw, processed)
        self.assertFalse(self.window.error_notice.isVisible())

    def test_playback_releases_monitor_switches_tracks_and_never_resumes_capture(self):
        self.record(output=True)
        self.wait_for(lambda: self.window.controller.media_status.duration >= 0.65)
        self.wait_for(lambda: all(
            meter._analysis is not None and meter._analysis.pitch_hz is not None
            for meter in (self.window.input_level, self.window.output_level)
        ))
        self.assertIn("monitoring off", self.window.audio_status.text())
        self.assertAlmostEqual(self.window.input_level._analysis.pitch_hz, 220, delta=1)
        self.assertAlmostEqual(self.window.output_level._analysis.pitch_hz, 220, delta=1)
        self.monitor()
        self.finish()
        with patch.object(QMessageBox, "exec", side_effect=AssertionError("unexpected dialog")):
            self.window.play_original_button.click()
        self.wait_for(lambda: self.window.controller.media_status.playback == "original" and not self.window.controller.busy)
        self.assertTrue(all(raw.closed for raw in self.library.streams if raw.kind == "input"))
        first_source_id = self.window.controller.media_status.meter_source_id
        self.wait_for(lambda: self.window.output_level.peak is not None)
        self.wait_for(lambda: (
            self.window.output_level._analysis is not None
            and self.window.output_level._analysis.pitch_hz is not None
        ))
        self.assertAlmostEqual(self.window.output_level._analysis.pitch_hz, 220, delta=1)
        self.assertIn("No capture", self.window.input_level.analysis_status.text())
        self.assertIsNone(self.window.input_level.peak)
        self.assertIn("Playback", self.window.output_level.accessibleName())
        first = self.library.streams[-1]
        self.assertFalse(self.window.record_button.isEnabled())
        with patch.object(QMessageBox, "exec", side_effect=AssertionError("unexpected dialog")):
            self.window.play_effect_button.click()
        self.wait_for(lambda: self.window.controller.media_status.playback == "effect" and not self.window.controller.busy)
        self.assertGreater(self.window.controller.media_status.meter_source_id, first_source_id)
        self.assertTrue(first.closed)
        self.assertIsNot(self.library.streams[-1], first)
        self.wait_for(lambda: (
            self.window.output_level._analysis is not None
            and self.window.output_level._analysis.pitch_hz is not None
        ))
        self.assertAlmostEqual(self.window.output_level._analysis.pitch_hz, 220, delta=1)
        self.assertIn("Effect", self.window.output_level.analysis_canvas.accessibleName())
        self.wait_for(lambda: self.window.controller.state is EngineState.STOPPED)
        self.assertIn("No capture", self.window.input_level.display_text)
        self.assertIn("No capture", self.window.output_level.display_text)
        self.assertEqual(self.window.input_level.history, ())
        self.assertEqual(self.window.output_level.history, ())
        self.assertEqual(self.window.input_level.pitch_history, ())
        self.assertEqual(self.window.output_level.pitch_history, ())
        self.assertFalse(self.window.controller.media_status.monitoring)
        self.window.input_combo.setFocus()
        QTest.keyClick(self.window.input_combo, Qt.Key.Key_Up)
        self.window.config_inputs[0].setText("44100")
        with patch.object(QMessageBox, "exec", side_effect=AssertionError("unexpected dialog")):
            self.window.play_original_button.click()
        self.wait_for(lambda: self.window.controller.media_status.playback == "original" and not self.window.controller.busy)
        self.assertEqual(self.library.streams[-1].samplerate, 48000)
        self.assertIn("No capture", self.window.input_level.display_text)
        self.window.stop_button.click()
        self.idle()
        self.assertFalse(self.window.controller.owns_devices)
        self.assertFalse(self.window.controller.media_status.monitoring)

    def test_playback_uses_current_settings_without_confirmation_dialog(self):
        self.record(output=True)
        self.finish()
        self.window.blocksize_input.setText("1024")
        with patch.object(self.window.controller, "request_play", return_value=True) as request:
            with patch.object(QMessageBox, "exec", side_effect=AssertionError("unexpected dialog")):
                self.window.play_original_button.click()
        request.assert_called_once()
        self.assertEqual(request.call_args.kwargs["blocksize"], 1024)
        self.assertIs(request.call_args.kwargs["output_confirmed"], True)

    def test_save_collision_retains_temporary_group_then_retry_saves_all_files(self):
        group = self.record()
        self.finish()
        destination = self.root / "existing"
        destination.mkdir()
        with patch("voice_changer.ui.main_window.QFileDialog.getSaveFileName", return_value=(str(destination), "")):
            self.window.save_recording_button.click()
            self.idle()
        self.assertEqual(self.window.controller.media_status.recording_state, RecordingState.SAVE_FAILED)
        self.assertTrue(group.original_path.exists())
        self.assertTrue(self.window.error_notice.isVisible())
        destination = self.root / "saved"
        with patch("voice_changer.ui.main_window.QFileDialog.getSaveFileName", return_value=(str(destination), "")):
            self.window.save_recording_button.click()
            self.idle()
        saved = self.window.controller.media_status.group
        self.assertTrue(saved.saved)
        self.assertFalse(self.window.error_notice.isVisible())
        self.assertEqual(saved.directory, destination)
        self.assertFalse(group.directory.exists())
        for name in ("original.wav", "effect.wav", "parameters.jsonl", "metadata.json"):
            self.assertTrue((destination / name).is_file())
        with patch.object(QMessageBox, "exec", side_effect=AssertionError("Saved data must not prompt")):
            self.window.close()
            self.wait_for(lambda: self.window.controller.is_closed)
        self.assertTrue(saved.original_path.exists())

    def test_unsaved_new_record_cancel_and_cancel_save_do_not_replace(self):
        group = self.record()
        self.finish()
        for answer in (QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save):
            with self.answer(answer), patch(
                "voice_changer.ui.main_window.QFileDialog.getSaveFileName", return_value=("", ""),
            ):
                self.window.record_button.click()
                self.idle()
            self.assertIs(self.window.controller.media_status.group, group)
            self.assertFalse(self.window.controller.media_status.recording)
            self.assertIsNone(self.window._pending_media_action)
        self.assertTrue(group.original_path.exists())

    def test_unsaved_new_record_save_failure_cancels_then_success_starts_new(self):
        group = self.record()
        self.finish()
        destination = self.root / "existing"
        destination.mkdir()
        with self.answer(QMessageBox.StandardButton.Save), patch(
            "voice_changer.ui.main_window.QFileDialog.getSaveFileName", return_value=(str(destination), ""),
        ):
            self.window.record_button.click()
            self.idle()
        self.assertFalse(self.window.controller.media_status.recording)
        self.assertIsNone(self.window._pending_media_action)
        self.assertTrue(group.directory.exists())
        destination = self.root / "saved"
        with self.answer(QMessageBox.StandardButton.Save), patch(
            "voice_changer.ui.main_window.QFileDialog.getSaveFileName", return_value=(str(destination), ""),
        ):
            self.window.record_button.click()
            self.wait_for(lambda: self.window.controller.media_status.recording and not self.window.controller.busy)
        self.assertIsNot(self.window.controller.media_status.group, group)
        self.assertTrue((destination / "original.wav").exists())

    def test_cancel_exit_finalizes_but_does_not_restore_monitor(self):
        group = self.record(output=True)
        self.monitor()
        with self.answer(QMessageBox.StandardButton.Cancel):
            self.window.close()
            self.wait_for(lambda: (
                self.window.controller.state is EngineState.STOPPED
                and not self.window.controller.busy and self.window._pending_media_action is None
            ))
        self.assertTrue(self.window.isVisible())
        self.assertFalse(self.window.controller.closing)
        self.assertTrue(group.complete)
        self.assertTrue(group.original_path.exists())
        self.assertFalse(self.window.controller.media_status.monitoring)

    def test_exit_discards_only_after_explicit_confirmation(self):
        group = self.record()
        with self.answer(QMessageBox.StandardButton.Discard):
            self.window.close()
            self.wait_for(lambda: self.window.controller.is_closed)
        self.assertFalse(group.directory.exists())

    def test_exit_saves_complete_group_before_closing(self):
        self.record()
        destination = self.root / "saved-on-exit"
        with self.answer(QMessageBox.StandardButton.Save), patch(
            "voice_changer.ui.main_window.QFileDialog.getSaveFileName", return_value=(str(destination), ""),
        ):
            self.window.close()
            self.wait_for(lambda: self.window.controller.is_closed)
        self.assertTrue((destination / "original.wav").is_file())
        self.assertTrue((destination / "effect.wav").is_file())
        self.assertTrue(json.loads((destination / "metadata.json").read_text())["complete"])

    def test_exit_during_save_waits_without_blocking_gui_then_closes(self):
        group = self.record()
        self.finish()
        destination = self.root / "slow-save"
        entered, release = threading.Event(), threading.Event()
        original_save = type(group).save

        def delayed_save(current, parent, name):
            entered.set()
            if not release.wait(timeout=3):
                raise RuntimeError("Synthetic save was not released")
            return original_save(current, parent, name)

        with patch.object(type(group), "save", delayed_save), patch(
            "voice_changer.ui.main_window.QFileDialog.getSaveFileName", return_value=(str(destination), ""),
        ):
            try:
                self.window.save_recording_button.click()
                self.wait_for(entered.is_set)
                self.window.close()
                self.assertIn("Waiting for the current operation", self.window.activity_status.text())
                self.assertFalse(self.window.controller.closing)
                self.app.processEvents()
                self.assertTrue(self.window.isVisible())
            finally:
                release.set()
            self.wait_for(lambda: self.window.controller.is_closed)
        self.assertTrue((destination / "metadata.json").is_file())

    def test_writer_fault_stops_audio_and_retains_incomplete_group(self):
        group = self.record(output=True)
        self.monitor()
        with patch.object(RecordingWriter, "check_health", side_effect=RuntimeError("synthetic disk failure")):
            # stateChanged and mediaStatus are separate queued signals; wait for both
            # so the assertion does not depend on delivery order.
            self.wait_for(
                lambda: self.window.controller.state is EngineState.FAULT
                and not self.window.controller.media_status.recording
            )
        self.assertFalse(self.window.controller.owns_devices)
        self.assertFalse(self.window.controller.media_status.recording)
        self.assertEqual(self.window.controller.media_status.recording_state, RecordingState.INCOMPLETE)
        self.assertIn("synthetic disk failure", self.window.error_notice.text())
        self.assertFalse(group.complete)
        self.assertTrue(group.directory.exists())

    def test_verified_saved_copy_cleanup_error_cancels_pending_exit(self):
        group = self.record()
        self.finish()
        destination = self.root / "saved"
        with self.answer(QMessageBox.StandardButton.Save), patch(
            "voice_changer.ui.main_window.QFileDialog.getSaveFileName", return_value=(str(destination), ""),
        ), patch.object(type(group), "_discard_owned", side_effect=OSError("synthetic temporary cleanup failure")):
            self.window.close()
            self.idle()
        self.assertFalse(self.window.controller.is_closed)
        self.assertTrue(self.window.controller.media_status.group.saved)
        self.assertEqual(self.window.controller.media_status.group.directory, destination)
        self.assertIn("cleanup failure", self.window.error_notice.text())
        self.assertIsNone(self.window._pending_media_action)
        self.assertTrue(group.directory.exists())

    def leftover(self, *, corrupt=False, complete=False):
        self.close_window()
        writer = RecordingWriter(48000, 1, 512)
        writer.start()
        samples = np.zeros((512, 1), dtype=np.float32)
        writer.submit(samples, samples, source_frame=0, parameters=PRESETS["original"], revision=0)
        group = writer.finish(complete=complete, reason=None if complete else "synthetic interrupted recording")
        writer.check_health()
        if corrupt:
            group.metadata_path.write_text("{}")
        self.window = MainWindow()
        self.window.show()
        self.wait_for(lambda: self.window.controller.is_ready
                      and not self.window.controller.busy and self.window.controller.devices_loaded)
        self.assertIn(str(group.directory), self.window.controller.media_status.recoverable)
        return group

    def test_startup_offers_recovery_without_capture_or_automatic_playback(self):
        group = self.leftover()
        self.assertEqual(self.library.streams, [])
        with patch("voice_changer.ui.main_window.QInputDialog.getItem", return_value=(str(group.directory), True)):
            self.window.recover_recording_button.click()
        self.idle()
        self.assertEqual(self.window.controller.media_status.group.directory, group.directory)
        self.assertEqual(self.window.controller.media_status.recording_state, RecordingState.INCOMPLETE)
        self.assertEqual(self.library.streams, [])

    def test_recovered_finalized_group_preserves_actual_complete_state(self):
        group = self.leftover(complete=True)
        with patch("voice_changer.ui.main_window.QInputDialog.getItem", return_value=(str(group.directory), True)):
            self.window.recover_recording_button.click()
        self.idle()
        self.assertTrue(self.window.controller.media_status.group.complete)
        self.assertEqual(self.window.controller.media_status.recording_state, RecordingState.UNSAVED)

    def test_corrupt_leftover_reports_recovery_error_and_requires_explicit_delete(self):
        group = self.leftover(corrupt=True)
        with patch("voice_changer.ui.main_window.QInputDialog.getItem", return_value=(str(group.directory), True)):
            self.window.recover_recording_button.click()
        self.idle()
        self.assertIsNone(self.window.controller.media_status.group)
        self.assertTrue(self.window.error_notice.isVisible())
        with patch("voice_changer.ui.main_window.QInputDialog.getItem", return_value=(str(group.directory), True)):
            with self.answer(QMessageBox.StandardButton.Cancel):
                self.window.delete_recovery_button.click()
            self.assertTrue(group.directory.exists())
            with self.answer(QMessageBox.StandardButton.Discard):
                self.window.delete_recovery_button.click()
            self.idle()
        self.assertFalse(group.directory.exists())
        self.assertEqual(self.window.controller.media_status.recoverable, ())
