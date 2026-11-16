"""Offscreen window -> controller -> real DSP, with synthetic device callbacks only."""

from ctypes.util import find_library
from dataclasses import replace
import os
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from voice_changer.audio.engine import EngineState
from voice_changer.audio import live_processing
from voice_changer.dsp.parameters import EffectParameters, PRESETS
from voice_changer.ui.main_window import MainWindow

from device_fakes import FakeDeviceLibrary
from qt_wait import wait_until


NATIVE = os.environ.get("VOICE_CHANGER_RUBBERBAND_LIBRARY") or find_library("rubberband")


class CallbackAbort(Exception):
    pass


class DuplexStream:
    def __init__(self, library, options):
        self.library = library
        self.options = options
        self.device = options["device"]
        self.samplerate = options["samplerate"]
        self.blocksize = options["blocksize"]
        self.channels = options["channels"]
        self.dtype = ("float32", "float32")
        self.active = False
        self.closed = False
        self.close_error = None

    def start(self):
        self.library.operations.append(("start", threading.get_ident()))
        self.active = True

    def stop(self):
        self.library.operations.append(("stop", threading.get_ident()))
        self.active = False

    def close(self):
        self.library.operations.append(("close", threading.get_ident()))
        if self.close_error:
            raise self.close_error
        self.closed = True
        self.active = False


class DuplexLibrary(FakeDeviceLibrary):
    CallbackAbort = CallbackAbort

    def __init__(self):
        super().__init__()
        self.streams = []
        self.operations = []

    def query_devices(self, index=None, kind=None):
        if index is None:
            return super().query_devices()
        self.operations.append(("query", threading.get_ident()))
        return self.devices[index]

    def check_input_settings(self, **options):
        self.operations.append(("input_settings", threading.get_ident()))

    def check_output_settings(self, **options):
        self.operations.append(("output_settings", threading.get_ident()))

    def Stream(self, **options):
        self.operations.append(("create", threading.get_ident()))
        stream = DuplexStream(self, options)
        self.streams.append(stream)
        return stream


class LiveDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        environment = patch.dict(os.environ, {"XDG_CONFIG_HOME": temporary.name})
        environment.start()
        self.addCleanup(environment.stop)
        self.library = DuplexLibrary()
        self.patcher = patch.dict(sys.modules, {"sounddevice": self.library})
        self.patcher.start()
        self.window = MainWindow()
        self.window.show()
        self.addCleanup(self.cleanup)
        self.wait_for(lambda: self.window.controller.is_ready
                      and not self.window.controller.busy and self.window.controller.devices_loaded)

    def wait_for(self, condition, timeout=3):
        self.assertTrue(wait_until(condition, timeout), "Timed out waiting for live demo")

    def cleanup(self):
        for raw in self.library.streams:
            raw.close_error = None
        self.window.close()
        self.wait_for(lambda: self.window.controller.is_closed)
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.patcher.stop()
        self.assertFalse(any(
            thread.name in ("voice-changer-dsp", "voice-changer-analysis")
            for thread in threading.enumerate()
        ))

    def configure(self):
        self.window.refresh_button.click()
        self.wait_for(lambda: not self.window.controller.busy)
        for combo in (self.window.input_combo, self.window.output_combo):
            combo.setFocus()
            QTest.keyClick(combo, Qt.Key.Key_Down)
        for editor, value in zip(self.window.config_inputs, ("48000", "512", "1", "0.1")):
            editor.setText(value)
        self.window.validate_button.click()
        self.assertIsNotNone(self.window._previewed_config)
        return self.window._previewed_config

    def start(self):
        self.configure()
        self.assertTrue(self.window.listen_button.isEnabled())
        self.window.listen_button.click()
        self.wait_for(
            lambda: self.window.controller.state is EngineState.RUNNING
            and not self.window.controller.busy,
        )
        return self.library.streams[-1]

    def feed(self, raw, count=48):
        results = []
        for index in range(count):
            positions = np.arange(raw.blocksize) + index * raw.blocksize
            samples = (0.25 * np.sin(2 * np.pi * 220 * positions / 48000)).astype(np.float32)[:, None]
            output = np.full_like(samples, 9)
            raw.options["callback"](samples, output, raw.blocksize, None, False)
            results.append(output.copy())
            self.assertLessEqual(float(np.abs(output).max()), 0.1)
            self.app.processEvents()
            time.sleep(raw.blocksize / 48000)
        return np.concatenate(results)

    @unittest.skipUnless(NATIVE, "Rubber Band unavailable; live native DSP integration not verified")
    def test_first_run_original_starts_with_default_format_without_opening_custom(self):
        self.assertEqual(self.window.preset_combo.currentData(), "original")
        self.assertTrue(self.window.tuning_panel.isHidden())
        for combo in (self.window.input_combo, self.window.output_combo):
            combo.setFocus()
            QTest.keyClick(combo, Qt.Key.Key_Down)
        self.window.listen_button.click()
        self.wait_for(lambda: self.window.controller.state is EngineState.RUNNING
                      and not self.window.controller.busy)
        self.assertEqual(len(self.library.streams), 1)
        stream = self.library.streams[0]
        self.assertEqual((stream.samplerate, stream.blocksize), (48000, 1024))
        self.assertEqual(self.window._configuration_values(), (48000, 1024, 1, 0.05))
        self.assertTrue(self.window.tuning_panel.isHidden())
        self.window.stop_button.click()
        self.wait_for(lambda: self.window.controller.state is EngineState.STOPPED
                      and not self.window.controller.busy)
        self.assertTrue(stream.closed)

    def test_controller_rejects_unconfirmed_or_stale_configuration(self):
        config = self.configure()
        errors = []
        self.window.controller.errorOccurred.connect(errors.append)
        for confirmation in (False, None, 1, "yes"):
            self.assertFalse(self.window.controller.request_listen(
                config, PRESETS["original"], output_confirmed=confirmation,
            ))
        self.assertIn("Confirm", str(errors[-1]))
        self.assertFalse(self.window.controller.request_listen(
            replace(config, input_device=replace(config.input_device, name="stale")),
            PRESETS["original"], output_confirmed=True,
        ))
        self.assertEqual(self.library.streams, [])
        self.window.blocksize_input.setText("1024")
        self.assertTrue(self.window.listen_button.isEnabled())
        self.assertIsNone(self.window._previewed_config)

    def test_setup_controls_and_disabled_reason_remain_visible_in_fixed_footer(self):
        self.assertIn("microphone", self.window.listen_button.accessibleDescription())
        self.configure()
        self.assertIn("Devices ready", self.window.listen_button.accessibleDescription())
        self.window.resize(480, 400)
        self.app.processEvents()
        scrollbar = self.window.scroll_area.verticalScrollBar()
        for position in (scrollbar.minimum(), scrollbar.maximum()):
            scrollbar.setValue(position)
            self.app.processEvents()
            for control in (
                self.window.listen_button, self.window.record_button,
                self.window.stop_button, self.window.close_button,
            ):
                bounds = control.rect().translated(
                    control.mapTo(self.window.centralWidget(), control.rect().topLeft()),
                )
                self.assertTrue(control.isVisibleTo(self.window))
                self.assertTrue(self.window.centralWidget().rect().contains(bounds))
        self.assertFalse(self.window.validate_button.isVisibleTo(self.window))
        self.window.preset_buttons["custom"].click()
        QTest.qWait(20)
        self.app.processEvents()
        self.window.scroll_area.ensureWidgetVisible(self.window.validate_button)
        self.app.processEvents()
        self.assertTrue(self.window.validate_button.isVisibleTo(self.window))
        validate_bounds = self.window.validate_button.rect().translated(
            self.window.validate_button.mapTo(
                self.window.scroll_area.viewport(), self.window.validate_button.rect().topLeft(),
            ),
        )
        self.assertTrue(
            self.window.scroll_area.viewport().rect().contains(validate_bounds),
            f"viewport={self.window.scroll_area.viewport().rect()}, "
            f"button={validate_bounds}, scrollbar={scrollbar.value()}/{scrollbar.maximum()}",
        )
        self.window.blocksize_input.setText("1024")
        self.assertIn("checked before listening", self.window.listen_button.accessibleDescription())
        self.assertTrue(self.window.listen_button.isEnabled())
        self.window.validate_button.setFocus()
        QTest.keyClick(self.window.validate_button, Qt.Key.Key_Space)
        self.assertIn("Devices ready", self.window.listen_button.accessibleDescription())
        self.assertEqual(self.library.streams, [])

    def test_listen_validates_automatically_without_confirmation_dialog(self):
        self.configure()
        self.window.blocksize_input.setText("1024")
        with patch.object(QMessageBox, "exec", side_effect=AssertionError("unexpected dialog")):
            self.window.listen_button.click()
        self.assertEqual(self.window._previewed_config.blocksize, 1024)
        self.assertEqual(self.library.streams, [])

    def test_invalid_input_is_visible_without_confirmation_dialog_or_audio(self):
        self.configure()
        self.window.output_ceiling_input.clear()
        with patch.object(QMessageBox, "exec", side_effect=AssertionError("unexpected dialog")):
            self.window.listen_button.click()
        self.assertIn("output_ceiling", self.window.error_notice.text())
        self.assertEqual(self.library.streams, [])

    def test_updated_configuration_is_validated_before_listen_without_dialog(self):
        self.configure()
        self.window.blocksize_input.setText("1024")
        with patch.object(self.window.controller, "request_listen", return_value=True) as request:
            with patch.object(QMessageBox, "exec", side_effect=AssertionError("unexpected dialog")):
                self.window.listen_button.click()
        request.assert_called_once()
        self.assertEqual(request.call_args.args[0].blocksize, 1024)
        self.assertIs(request.call_args.kwargs["output_confirmed"], True)
        self.assertEqual(self.library.streams, [])

    def test_recording_monitor_starts_on_click_without_confirmation_dialog(self):
        self.configure()
        group = SimpleNamespace(samplerate=48000, channels=1, frames=1, saved=True)
        original_status = self.window.controller.media_status
        self.window.controller._media_status = replace(
            original_status, recording=True, group=group,
        )
        try:
            with patch.object(self.window.controller, "request_monitor", return_value=True) as request:
                with patch.object(QMessageBox, "exec", side_effect=AssertionError("unexpected dialog")):
                    self.window._request_listen()
            request.assert_called_once_with(output_confirmed=True)
        finally:
            self.window.controller._media_status = original_status

    def test_playback_starts_on_click_without_confirmation_dialog(self):
        self.configure()
        group = SimpleNamespace(samplerate=48000, channels=1, frames=1, saved=True)
        original_status = self.window.controller.media_status
        self.window.controller._media_status = replace(
            original_status, group=group,
        )
        try:
            with patch.object(self.window.controller, "request_play", return_value=True) as request:
                with patch.object(QMessageBox, "exec", side_effect=AssertionError("unexpected dialog")):
                    self.window._request_play("original")
            request.assert_called_once_with(
                "original", blocksize=512, output_ceiling=0.1, output_confirmed=True,
            )
        finally:
            self.window.controller._media_status = original_status

    @unittest.skipUnless(NATIVE, "Rubber Band unavailable; live native DSP integration not verified")
    def test_listen_starts_without_confirmation_dialog_or_separate_validation_click(self):
        self.configure()
        self.window.blocksize_input.setText("1024")
        self.assertIsNone(self.window._previewed_config)
        with patch.object(QMessageBox, "exec", side_effect=AssertionError("unexpected dialog")):
            self.window.listen_button.click()
        self.wait_for(lambda: self.window.controller.state is EngineState.RUNNING
                      and not self.window.controller.busy)
        self.assertEqual(len(self.library.streams), 1)
        self.assertEqual(self.library.streams[0].blocksize, 1024)

    def test_missing_dsp_failure_is_visible_without_starting_native_device(self):
        self.configure()
        with patch.object(live_processing, "RubberBand",
                          side_effect=RuntimeError("test missing DSP library")):
            self.window.listen_button.click()
            self.wait_for(
                lambda: self.window.controller.state is EngineState.FAULT
                and not self.window.controller.busy,
            )
        self.assertIn("test missing DSP library", self.window.error_notice.text())
        self.assertEqual(self.library.streams, [])
        self.assertFalse(self.window.controller.owns_devices)
        self.assertFalse(self.window.listen_button.isEnabled())
        self.assertTrue(self.window.stop_button.isEnabled())
        self.window.stop_button.click()
        self.wait_for(lambda: not self.window.controller.busy)
        self.assertIs(self.window.controller.state, EngineState.STOPPED)

    @unittest.skipUnless(NATIVE, "Rubber Band unavailable; live native DSP integration not verified")
    def test_window_native_dsp_callback_levels_parameters_stop_and_restart(self):
        raw = self.start()
        processor = raw.options["callback"].__self__._processor
        self.assertTrue(self.window.controller.owns_devices)
        self.assertFalse(self.window.refresh_button.isEnabled())
        self.assertFalse(self.window.input_combo.isEnabled())
        self.assertTrue(self.window.stop_button.isEnabled())
        self.assertTrue(self.window.pitch_input.isEnabled())
        output = self.feed(raw)
        self.assertGreater(float(np.abs(output).max()), 0.05)
        self.wait_for(lambda: "dBFS" in self.window.input_level.display_text)
        self.assertGreater(self.window.input_level.peak, 0.2)
        self.assertLessEqual(self.window.output_level.peak, 0.1)
        self.wait_for(lambda: bool(self.window.input_level.history))
        self.wait_for(lambda: all(
            meter._analysis is not None and meter._analysis.pitch_hz is not None
            for meter in (self.window.input_level, self.window.output_level)
        ))
        self.assertAlmostEqual(self.window.input_level._analysis.pitch_hz, 220, delta=1)
        self.assertAlmostEqual(self.window.output_level._analysis.pitch_hz, 220, delta=1)
        self.assertGreater(self.window.input_level._analysis.bands[0], 0)
        self.assertGreater(self.window.output_level._analysis.bands[0], 0)
        source_id = self.window.controller.media_status.meter_source_id
        self.window.preset_buttons["male_to_female"].click()
        for value in (1, 2, 3, 4):
            self.window.pitch_input.setValue(value)
        self.assertEqual(
            processor.control.requested[1],
            EffectParameters(4, PRESETS["male_to_female"].formant_scale, False),
        )
        self.feed(raw, 24)
        self.wait_for(
            lambda: processor.control.applied_revision == processor.control.requested[0],
        )
        self.wait_for(lambda: "applied" in self.window.effect_status.text())
        self.assertTrue(all(
            ident != threading.get_ident() for _, ident in self.library.operations
        ))
        self.window.resize(480, 400)
        self.app.processEvents()
        for button in (self.window.listen_button, self.window.stop_button, self.window.close_button):
            bounds = button.rect().translated(
                button.mapTo(self.window.centralWidget(), button.rect().topLeft()),
            )
            self.assertTrue(self.window.centralWidget().rect().contains(bounds))
        self.window.stop_button.setFocus()
        QTest.keyClick(self.window.stop_button, Qt.Key.Key_Space)
        self.wait_for(lambda: not self.window.controller.busy)
        self.assertIs(self.window.controller.state, EngineState.STOPPED)
        self.assertTrue(raw.closed)
        self.assertIsNone(processor._thread)
        self.assertFalse(self.window.controller.owns_devices)
        self.assertTrue(self.window.listen_button.isEnabled())
        self.assertIn("checked before listening", self.window.listen_guidance.text())
        self.assertIn("No capture", self.window.input_level.display_text)
        self.assertEqual(self.window.input_level.history, ())
        self.assertEqual(self.window.output_level.history, ())
        self.assertEqual(self.window.input_level.pitch_history, ())
        self.assertEqual(self.window.output_level.pitch_history, ())
        self.assertTrue(self.window.controller.request_start())
        self.wait_for(
            lambda: self.window.controller.state is EngineState.FAULT
            and not self.window.controller.busy,
        )
        self.assertIn("not configured", self.window.error_notice.text())
        self.assertEqual(len(self.library.streams), 1)
        self.window.stop_button.click()
        self.wait_for(lambda: not self.window.controller.busy)
        restarted = self.start()
        self.assertIsNot(restarted, raw)
        self.assertEqual(len(self.library.streams), 2)
        self.assertGreater(self.window.controller.media_status.meter_source_id, source_id)
        self.assertIsNone(self.window.input_level.peak)
        self.assertIsNone(self.window.output_level.peak)
        self.assertIn("Waiting", self.window.input_level.display_text)

    @unittest.skipUnless(NATIVE, "Rubber Band unavailable; live native DSP integration not verified")
    def test_worker_failure_stops_and_surfaces_cause_without_automatic_restart(self):
        raw = self.start()
        processor = raw.options["callback"].__self__._processor
        processor.error = RuntimeError("synthetic native processing failure")
        self.wait_for(lambda: self.window.controller.state is EngineState.FAULT)
        self.assertIn("synthetic native processing failure", self.window.error_notice.text())
        self.assertTrue(raw.closed)
        self.assertIsNone(processor._thread)

    @unittest.skipUnless(NATIVE, "Rubber Band unavailable; live native DSP integration not verified")
    def test_stale_status_cannot_acknowledge_newer_parameters(self):
        self.start()
        controller = self.window.controller
        self.assertTrue(controller.request_parameters(PRESETS["male_to_female"]))
        controller._on_live_status((None, 0, 0, controller.media_status.meter_source_id))
        self.assertIn("pending", self.window.effect_status.text())
        self.assertFalse(self.window.listen_button.isEnabled())
        self.assertEqual(len(self.library.streams), 1)

    @unittest.skipUnless(NATIVE, "Rubber Band unavailable; live native DSP integration not verified")
    def test_callback_fault_silences_live_output_and_releases_dsp(self):
        raw = self.start()
        processor = raw.options["callback"].__self__._processor
        samples = np.full((512, 1), np.nan, dtype=np.float32)
        output = np.ones_like(samples)
        with self.assertRaises(CallbackAbort):
            raw.options["callback"](samples, output, 512, None, False)
        np.testing.assert_array_equal(output, 0)
        self.wait_for(lambda: self.window.controller.state is EngineState.FAULT)
        self.assertIn("Non-finite", self.window.error_notice.text())
        self.assertIn("No capture", self.window.output_level.display_text)
        self.assertTrue(raw.closed)
        self.assertIsNone(processor._thread)

    @unittest.skipUnless(NATIVE, "Rubber Band unavailable; live native DSP integration not verified")
    def test_failed_release_keeps_ownership_and_retry_closes_worker(self):
        raw = self.start()
        processor = raw.options["callback"].__self__._processor
        raw.close_error = OSError("synthetic close failure")
        self.window.stop_button.click()
        self.wait_for(
            lambda: self.window.controller.state is EngineState.FAULT
            and not self.window.controller.busy,
        )
        self.assertIn("synthetic close failure", self.window.error_notice.text())
        self.assertTrue(self.window.controller.owns_devices)
        self.assertFalse(self.window.refresh_button.isEnabled())
        raw.close_error = None
        self.window.stop_button.click()
        self.wait_for(lambda: not self.window.controller.busy)
        self.assertFalse(self.window.controller.owns_devices)
        self.assertTrue(raw.closed)
        self.assertIsNone(processor._thread)


if __name__ == "__main__":
    unittest.main()
