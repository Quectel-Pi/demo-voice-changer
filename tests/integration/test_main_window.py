"""Offscreen desktop entry tests; no devices, network or audio required."""

import math
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPoint, QPointF, QThread, QTimer, Qt
from PySide6.QtGui import QFontMetrics, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QStyle, QStyleOption

from voice_changer.ui.main_window import MainWindow
from voice_changer.ui.audio_wave import _ANALYSIS_TEXT
from voice_changer.app.engine_controller import EngineController, _PolledStatus
from voice_changer.audio import sounddevice_backend
from voice_changer.audio.analysis_status import AnalysisReading, AnalysisSnapshot
from voice_changer.audio.engine import EngineError, EngineState
from voice_changer.audio.media_status import MediaStatus, RecordingState
from voice_changer.dsp.parameters import LiveControl, PRESETS
from voice_changer.ui.language_preferences import load_language

from controller_fakes import FakeBackend
from device_fakes import FakeDeviceLibrary
from qt_wait import ensure_visible_after_layout, wait_until


ROOT = Path(__file__).resolve().parents[2]


class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        environment = patch.dict(os.environ, {"XDG_CONFIG_HOME": temporary.name})
        environment.start()
        self.addCleanup(environment.stop)
        default_library = patch.dict(sys.modules, {"sounddevice": FakeDeviceLibrary()})
        default_library.start()
        self.addCleanup(default_library.stop)
        self.modules_before = set(sys.modules)
        self.window = MainWindow()
        self.assertFalse(self.window.diagnostics_panel.isVisibleTo(self.window))
        self.assertTrue(self.window.tuning_panel.isHidden())
        self.assertFalse(self.window.tuning_controls.isHidden())
        self.assertTrue(self.window.preset_combo.isHidden())
        self.assertEqual(len(self.window.preset_buttons), 10)
        self.backend = None
        self.library = None
        self.library_patch = None
        self.addCleanup(self.cleanup_window)

    def wait_for(self, condition, timeout=2):
        self.assertTrue(wait_until(condition, timeout), "Timed out waiting for desktop cleanup")

    def cleanup_window(self):
        if self.library is not None:
            self.library.release.set()
        if self.backend is not None:
            self.backend.start_release.set()
            self.backend.close_error = None
            self.backend.stop_error = None
        self.window.close()
        self.wait_for(lambda: self.window.controller.is_closed)
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        if self.library_patch is not None:
            self.library_patch.stop()
            self.library_patch = None
            self.library = None

    def device_library(self):
        self.library = FakeDeviceLibrary()
        self.library_patch = patch.dict(sys.modules, {"sounddevice": self.library})
        self.library_patch.start()
        return self.library

    def refresh_devices(self):
        already_shown = self.window.isVisible()
        self.window.show()
        self.wait_for(lambda: self.window.controller.is_ready and not self.window._startup_refresh_due)
        if already_shown:
            self.window.refresh_button.click()
        self.wait_for(lambda: not self.window.controller.busy)
        self.app.processEvents()

    def use_test_backend(self):
        self.cleanup_window()
        self.backend = FakeBackend()
        controller = EngineController(self.backend)
        self.window = MainWindow(controller)
        self.window.show()
        self.wait_for(lambda: controller.is_ready and not controller.busy
                      and not self.window._startup_refresh_due)
        return controller

    def select_config_pair(self):
        self.refresh_devices()
        self.window.preset_buttons["custom"].click()
        for combo in (self.window.input_combo, self.window.output_combo):
            combo.setFocus()
            QTest.keyClick(combo, Qt.Key.Key_Down)

    def select_custom_voice(self):
        self.window.show()
        self.wait_for(lambda: self.window.preset_buttons["custom"].isEnabled())
        self.window.preset_buttons["custom"].click()
        self.app.processEvents()
        self.assertEqual(self.window.preset_combo.currentData(), "custom")

    def fill_config_fields(self, values=("48000", "512", "1", "0.1")):
        for editor, value in zip(self.window.config_inputs, values):
            editor.setText(value)

    def validate_config(self):
        self.window.validate_button.click()
        self.wait_for(lambda: not self.window._feedback_scroll_timer.isActive())
        self.app.processEvents()

    def test_unavailable_audio_is_explicit_without_fake_controls(self):
        self.wait_for(lambda: self.window.controller.is_ready)
        self.assertEqual(self.window.audio_status.text(), "未采集")
        self.assertIn("默认原声", self.window.notice.text())
        self.assertFalse(self.window.listen_button.isEnabled())
        self.assertFalse(self.window.stop_button.isEnabled())
        self.assertIn("未采集", self.window.input_level.display_text)
        self.assertIn("未采集", self.window.output_level.display_text)
        for meter in (self.window.input_level, self.window.output_level):
            self.assertIsNone(meter.peak)
            self.assertEqual(meter.history, ())
            self.assertFalse(meter._timer.isActive())
        visible_buttons = {
            button for button in self.window.findChildren(QPushButton)
            if button.isVisibleTo(self.window)
        }
        self.assertEqual(
            visible_buttons,
            {
                self.window.refresh_button, self.window.listen_button,
                self.window.stop_button, self.window.record_button,
                self.window.close_button, self.window.language_button,
                self.window.wave_layout_button,
                *self.window.preset_buttons.values(), *self.window.mode_buttons.values(),
            },
        )
        self.assertEqual([editor.text() for editor in self.window.config_inputs],
                         ["48000", "1024", "1", "0.05"])
        self.assertFalse(self.window.validate_button.isEnabled())
        self.assertFalse(self.window.prepare_backend_button.isEnabled())
        self.assertFalse(self.window.config_preview.isVisible())
        self.assertFalse(self.window.controller.devices_loaded)
        self.assertIn("Refresh the device list", self.window.device_status.text())
        self.assertTrue(self.window.tuning_panel.isHidden())
        self.assertFalse(self.window.diagnostics_panel.isVisibleTo(self.window))
        self.assertFalse(hasattr(self.window, "tuning_toggle"))
        self.assertFalse(hasattr(self.window, "diagnostics_toggle"))
        self.assertFalse(self.window.play_original_button.isVisibleTo(self.window))
        for combo in (self.window.input_combo, self.window.output_combo):
            self.assertEqual(combo.count(), 1)
            self.assertIsNone(combo.currentData())
            self.assertFalse(combo.isEnabled())
        self.assertFalse(any(
            name.startswith(("sounddevice", "numpy", "voice_changer.audio.sounddevice_backend", "voice_changer.dsp",
                             "voice_changer.ai"))
            for name in set(sys.modules) - self.modules_before
        ))

    def test_language_switch_preserves_mode_text_devices_preview_and_window_size(self):
        library = self.device_library()
        self.select_config_pair()
        self.fill_config_fields()
        self.validate_config()
        self.assertIn("输入：", self.window.config_preview.toPlainText())

        self.window.mode_buttons["tts"].click()
        page = self.window.tts_page
        page.text_input.setPlainText("User text stays exactly as entered.")
        calls_before = list(library.calls)
        self.assertFalse(self.window.controller.owns_devices)

        for width, height in ((480, 400), (640, 440), (1366, 768)):
            self.window.resize(width, height)
            self.app.processEvents()
            self.window.language_button.click()
            self.assertEqual((self.window.width(), self.window.height()), (width, height))
            self.assertEqual(self.window.mode, "tts")
            self.assertIs(self.window.mode_stack.currentWidget(), page)
            self.assertEqual(page.text_input.toPlainText(), "User text stays exactly as entered.")
            self.assertEqual(self.window.controller.state, EngineState.STOPPED)
            self.assertFalse(self.window.controller.owns_devices)
            self.assertEqual(library.calls, calls_before)
            self.window.mode_buttons["vc"].click()
            self.assertEqual(self.window.mode, "vc")
            self.assertEqual(self.window.footer_stack.currentWidget().objectName(), "fixedFooter")
            self.assertEqual((self.window.width(), self.window.height()), (width, height))
            self.window.mode_buttons["tts"].click()
            self.assertEqual(self.window.mode, "tts")
            self.assertIs(self.window.tts_page, page)

        self.window.showMaximized()
        self.app.processEvents()
        maximized_geometry = self.window.geometry()
        self.window.language_button.click()
        self.assertTrue(self.window.isMaximized())
        self.assertEqual(self.window.geometry(), maximized_geometry)
        self.window.language_button.click()

        self.window.mode_buttons["dsp"].click()
        self.assertIn("Input:", self.window.config_preview.toPlainText())
        self.assertIn("test input", self.window.config_preview.toPlainText())
        self.window.language_button.click()
        self.assertIn("输入：", self.window.config_preview.toPlainText())
        self.assertIn("test input", self.window.config_preview.toPlainText())
        self.assertEqual(library.calls, calls_before)
        self.assertEqual(load_language()[0], "zh")

    def test_language_setting_write_failure_is_visible_after_immediate_switch(self):
        with patch(
            "voice_changer.ui.main_window.save_language",
            side_effect=OSError("read-only"),
        ):
            self.window.language_button.click()
        self.assertEqual(self.window._language, "en")
        self.assertFalse(self.window.language_notice.isHidden())
        self.assertIn("could not be saved", self.window.language_notice.text())

    def test_chart_language_switch_preserves_real_readings_and_histories(self):
        readings = []
        for meter in (self.window.input_level, self.window.output_level):
            meter.set_peak(0.25)
            meter._sample()
            meter.set_analysis(AnalysisReading((0.1, 0.02, 0.01), 220.0))
            meter._sample_analysis()
            readings.append((meter.history, meter.pitch_history, meter.analysis_canvas.reading))
        for language in ("en", "zh"):
            self.window.language_button.click()
            self.assertEqual(self.window._language, language)
            for meter, snapshot in zip(
                (self.window.input_level, self.window.output_level), readings,
            ):
                self.assertEqual(meter.analysis_canvas._language, language)
                self.assertEqual(
                    (meter.history, meter.pitch_history, meter.analysis_canvas.reading), snapshot,
                )
        self.assertFalse(self.window.controller.owns_devices)

    def test_preset_panel_stays_content_sized_when_custom_is_selected(self):
        self.window.resize(900, 600)
        self.window.show()
        self.wait_for(lambda: self.window.preset_buttons["custom"].isEnabled())
        self.app.processEvents()
        preset_height = self.window.preset_panel.height()
        self.assertLessEqual(
            preset_height, self.window.preset_panel.sizeHint().height() + 2,
        )
        self.window.preset_buttons["custom"].click()
        self.app.processEvents()
        self.assertEqual(self.window.preset_panel.height(), preset_height)
        self.window.preset_buttons["original"].click()
        self.app.processEvents()
        self.assertEqual(self.window.preset_panel.height(), preset_height)

    def test_only_custom_exposes_parameters_and_preserves_current_voice_as_base(self):
        self.refresh_devices()
        self.assertEqual(self.window._configuration_values(), (48000, 1024, 1, 0.05))
        for key in self.window.preset_buttons:
            if key == "custom":
                continue
            with self.subTest(preset=key):
                self.window.preset_buttons[key].click()
                self.assertTrue(self.window.tuning_panel.isHidden())
                current = tuple(control.value() for control in self.window._sound_inputs)
                self.window.preset_buttons["custom"].click()
                self.app.processEvents()
                self.assertFalse(self.window.tuning_panel.isHidden())
                self.assertTrue(all(editor.isVisibleTo(self.window) for editor in self.window.config_inputs))
                self.assertEqual(tuple(control.value() for control in self.window._sound_inputs), current)
                self.assertEqual(self.window.pitch_slider.value(), round(current[0] * 10))
                self.assertEqual(self.window._base_preset, key)
                self.window.pitch_input.setValue(current[0] + 0.125)
                self.window.reset_effect_button.click()
                self.assertEqual(self.window.preset_combo.currentData(), key)
                self.assertTrue(self.window.tuning_panel.isHidden())
                self.assertEqual(self.window.pitch_input.value(), current[0])
                self.assertEqual(self.window.pitch_slider.value(), round(current[0] * 10))
        self.assertFalse(self.window.controller.owns_devices)

    def test_sound_sliders_preserve_precision_and_single_parameter_submission(self):
        self.select_custom_voice()
        self.assertEqual(len(self.window.parameter_sliders), 8)
        expected = (
            (self.window.pitch_input, 10, 0.125),
            (self.window.formant_input, 100, 1.13),
            (self.window.robot_input, 1, 23),
            (self.window.echo_input, 1, 17),
            (self.window.reverb_input, 1, 19),
            (self.window.input_gain_input, 10, -3.7),
            (self.window.output_gain_input, 10, 4.3),
            (self.window.denoise_input, 1, 37),
        )
        for editor, scale, precise in expected:
            with self.subTest(parameter=editor.accessibleName()):
                slider = self.window.parameter_sliders[editor]
                self.assertEqual(slider.minimum(), round(editor.minimum() * scale))
                self.assertEqual(slider.maximum(), round(editor.maximum() * scale))
                self.assertEqual(slider.accessibleName(), editor.accessibleName())
                editor.setValue(precise)
                self.assertEqual(editor.value(), precise)
                self.assertEqual(slider.value(), round(precise * scale))
                slider.setValue(slider.minimum())
                self.assertEqual(editor.value(), editor.minimum())
                slider.setValue(slider.maximum())
                self.assertEqual(editor.value(), editor.maximum())
                editor.setValue(precise)
                before = editor.value()
                with patch.object(self.window.controller, "request_parameters", return_value=True) as request:
                    self.window.controller._state = EngineState.RUNNING
                    slider.setValue(slider.minimum())
                    request.assert_called_once()
                    self.window.controller._state = EngineState.STOPPED
                self.assertNotEqual(editor.value(), before)
        self.window._update_live_controls()
        self.assertTrue(all(slider.isEnabled() for slider in self.window.parameter_sliders.values()))
        self.window._closing_requested = True
        self.window._update_live_controls()
        self.assertTrue(all(not slider.isEnabled() for slider in self.window.parameter_sliders.values()))
        self.window._closing_requested = False

    def test_denoise_control_is_visible_in_every_preset_and_maps_to_parameters(self):
        self.refresh_devices()
        self.assertTrue(self.window.denoise_panel.isVisibleTo(self.window))
        self.assertTrue(self.window.denoise_input.isVisibleTo(self.window))
        self.assertEqual(self.window.denoise_input.value(), 0)
        self.assertEqual(self.window._effect_parameters().denoise_strength, 0.0)
        self.window.denoise_input.setValue(50)
        self.assertEqual(self.window._effect_parameters().denoise_strength, 0.5)
        self.assertTrue(self.window._effect_parameters().bypass)
        self.window.preset_buttons["robot"].click()
        self.app.processEvents()
        self.assertTrue(self.window.tuning_panel.isHidden())
        self.assertEqual(self.window.preset_combo.currentData(), "robot")
        parameters = self.window._effect_parameters()
        self.assertEqual(parameters.denoise_strength, 0.5)
        self.assertFalse(parameters.bypass)
        self.window.denoise_input.setValue(0)
        self.assertEqual(self.window._effect_parameters().denoise_strength, 0.0)

    def test_listen_rejects_a_format_the_dsp_demo_cannot_run(self):
        self.select_config_pair()
        self.fill_config_fields(("44100", "512", "1", "0.05"))
        self.validate_config()
        self.window.listen_button.click()
        self.app.processEvents()
        self.assertTrue(self.window.error_notice.isVisible())
        message = self.window.error_notice.text()
        self.assertIn("48 kHz", message)
        self.assertIn("44100", message)
        self.assertFalse(self.window.controller.owns_devices)

    def test_invalid_hidden_configuration_remains_visible_and_points_to_custom(self):
        self.select_config_pair()
        self.window.blocksize_input.setText("invalid")
        self.window.preset_buttons["original"].click()
        self.window.listen_button.click()
        self.app.processEvents()
        self.assertTrue(self.window.tuning_panel.isHidden())
        self.assertTrue(self.window.error_notice.isVisible())
        self.assertIn("自定义声音", self.window.error_notice.text())
        self.assertFalse(self.window.controller.owns_devices)
        self.window.preset_buttons["custom"].click()
        self.assertEqual(self.window.blocksize_input.text(), "invalid")

    def test_custom_parameters_are_dense_on_desktop_and_stacked_on_small_windows(self):
        self.select_custom_voice()
        for width, height in ((1280, 731), (480, 400), (1280, 731)):
            self.window.resize(width, height)
            self.app.processEvents()
            for form in self.window.parameter_forms:
                for index in range(form.rowCount()):
                    label = form.itemAt(index, form.ItemRole.LabelRole).widget()
                    field = form.itemAt(index, form.ItemRole.FieldRole).widget()
                    if width >= 900:
                        self.assertLess(label.geometry().right(), field.geometry().left())
                        self.assertLessEqual(abs(label.geometry().center().y() - field.geometry().center().y()), 2)
                    else:
                        self.assertLess(label.geometry().bottom(), field.geometry().top())
            if width >= 900:
                self.assertLessEqual(self.window.tuning_controls.height(), 370)
            for slider in self.window.parameter_sliders.values():
                self.assertGreaterEqual(slider.height(), 32)
                self.assertGreater(slider.width(), 100)

    def test_slider_updates_do_not_resize_or_move_status_or_shift_scroll_position(self):
        self.select_custom_voice()
        window = self.window
        area = window.scroll_area

        class GeometryTrace(QObject):
            def __init__(self):
                super().__init__()
                self.changes = []

            def eventFilter(self, watched, event):
                if event.type() in (QEvent.Type.Move, QEvent.Type.Resize):
                    self.changes.append((watched, event.type()))
                return False

        trace = GeometryTrace()
        for widget in (window.effect_status, window.tuning_controls, window.tuning_panel):
            widget.installEventFilter(trace)
        for width, height in ((1280, 731), (480, 400)):
            window.resize(width, height)
            for language in ("zh", "en"):
                if window._language != language:
                    window.language_button.click()
                for running in (False, True):
                    with patch.object(window.controller, "request_parameters", return_value=True):
                        window.controller._state = EngineState.RUNNING if running else EngineState.STOPPED
                        try:
                            window._submit_effects()
                            self.app.processEvents()
                            area.ensureWidgetVisible(window.pitch_slider)
                            QTest.qWait(30)
                            trace.changes.clear()
                            position = area.verticalScrollBar().value()
                            for editor, slider in window.parameter_sliders.items():
                                for value in (slider.minimum(), slider.maximum(), 0):
                                    slider.setValue(max(slider.minimum(), value))
                                    QTest.qWait(5)
                                    self.assertEqual(area.verticalScrollBar().value(), position)
                                    self.assertNotIn("\n", window.effect_status.text())
                            self.assertEqual(trace.changes, [], (width, language, running))
                        finally:
                            window.controller._state = EngineState.STOPPED
        for widget in (window.effect_status, window.tuning_controls, window.tuning_panel):
            widget.removeEventFilter(trace)

    def test_parameter_wheel_scrolls_page_without_changing_values(self):
        self.select_custom_voice()
        self.window.resize(480, 400)
        self.window.show()
        self.wait_for(lambda: self.window.controller.is_ready)
        self.app.processEvents()
        scroll = self.window.scroll_area.verticalScrollBar()
        self.window.scroll_area.ensureWidgetVisible(self.window.pitch_input)
        self.app.processEvents()
        old_scroll = scroll.value()
        old_value = self.window.pitch_input.value()
        bounds = self.window.pitch_input.rect()
        local = QPointF(bounds.center())
        global_position = QPointF(self.window.pitch_input.mapToGlobal(bounds.center()))
        event = QWheelEvent(
            local, global_position, QPoint(), QPoint(0, 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate, False,
        )
        QApplication.sendEvent(self.window.pitch_input, event)
        self.app.processEvents()
        self.assertEqual(self.window.pitch_input.value(), old_value)
        self.assertLess(scroll.value(), old_scroll)
        self.window.scroll_area.ensureWidgetVisible(self.window.pitch_slider)
        self.app.processEvents()
        old_scroll = scroll.value()
        old_value = self.window.pitch_slider.value()
        bounds = self.window.pitch_slider.rect()
        local = QPointF(bounds.center())
        global_position = QPointF(self.window.pitch_slider.mapToGlobal(bounds.center()))
        event = QWheelEvent(
            local, global_position, QPoint(), QPoint(0, 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate, False,
        )
        QApplication.sendEvent(self.window.pitch_slider, event)
        self.app.processEvents()
        self.assertEqual(self.window.pitch_slider.value(), old_value)
        self.assertLess(scroll.value(), old_scroll)

    def wheel(self, control, delta=120, position=None):
        point = control.rect().center() if position is None else position
        event = QWheelEvent(
            QPointF(point), QPointF(control.mapToGlobal(point)), QPoint(), QPoint(0, delta),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate, False,
        )
        QApplication.sendEvent(control, event)
        self.app.processEvents()
        self.assertTrue(event.isAccepted())

    def test_window_routed_wheel_requires_click_or_keyboard_navigation(self):
        self.select_custom_voice()
        self.window.resize(480, 400)
        self.refresh_devices()
        self.window.activateWindow()
        self.app.processEvents()
        scroll = self.window.scroll_area.verticalScrollBar()

        def wheel(widget, delta=120, point=None):
            point = widget.rect().center() if point is None else point
            QTest.wheelEvent(
                self.window.windowHandle(), QPointF(widget.mapTo(self.window, point)),
                QPoint(0, delta),
            )
            self.app.processEvents()

        for control in (
            *self.window.parameter_sliders.values(), self.window.pitch_input, self.window.formant_input,
            self.window.robot_input, self.window.echo_input, self.window.reverb_input,
            self.window.input_gain_input, self.window.output_gain_input,
        ):
            with self.subTest(control=control.accessibleName()):
                self.window.scroll_area.ensureWidgetVisible(control)
                self.window.close_button.setFocus(Qt.FocusReason.OtherFocusReason)
                self.app.processEvents()
                old_value, old_scroll = control.value(), scroll.value()
                wheel(control)
                self.assertEqual(control.value(), old_value)
                self.assertLess(scroll.value(), old_scroll)
                self.window.scroll_area.ensureWidgetVisible(control)
                self.app.processEvents()
                QTest.mouseClick(control, Qt.MouseButton.LeftButton)
                old_value, old_scroll = control.value(), scroll.value()
                wheel(control)
                self.assertGreater(control.value(), old_value)
                self.assertEqual(scroll.value(), old_scroll)
                for boundary, delta in ((control.maximum(), 120), (control.minimum(), -120)):
                    control.setValue(boundary)
                    old_scroll = scroll.value()
                    wheel(control, delta)
                    self.assertEqual(control.value(), boundary)
                    self.assertEqual(scroll.value(), old_scroll)
                control.setValue(old_value)
                QTest.mouseClick(self.window.listen_guidance, Qt.MouseButton.LeftButton)
                old_scroll = scroll.value()
                wheel(control)
                self.assertEqual(control.value(), old_value)
                self.assertLess(scroll.value(), old_scroll)
                self.window.scroll_area.ensureWidgetVisible(control)
                control.setFocus(Qt.FocusReason.OtherFocusReason)
                QTest.keyClick(control, Qt.Key.Key_Backtab)
                QTest.keyClick(self.app.focusWidget(), Qt.Key.Key_Tab)
                self.app.processEvents()
                self.assertTrue(control.hasFocus())
                old_value, old_scroll = control.value(), scroll.value()
                wheel(control)
                self.assertGreater(control.value(), old_value)
                self.assertEqual(scroll.value(), old_scroll)
                old_value = control.value()
                wheel(self.window.scroll_area.viewport(), point=QPoint(2, 2))
                self.assertEqual(control.value(), old_value)
                self.assertLess(scroll.value(), old_scroll)

    def test_click_and_tab_arm_parameter_wheel_without_mouse_scroll_jumps(self):
        self.select_custom_voice()
        self.device_library()
        self.window.resize(480, 400)
        self.refresh_devices()
        scroll = self.window.scroll_area.verticalScrollBar()
        for control in (self.window.pitch_input, self.window.pitch_slider,
                        self.window.formant_input, self.window.input_gain_input):
            with self.subTest(control=control.accessibleName()):
                self.window.scroll_area.ensureWidgetVisible(control)
                self.app.processEvents()
                before = scroll.value()
                QTest.mouseClick(control, Qt.MouseButton.LeftButton)
                self.app.processEvents()
                self.assertEqual(scroll.value(), before)
                value = control.value()
                self.wheel(control)
                self.assertGreater(control.value(), value)
                self.assertEqual(scroll.value(), before)
                control.setValue(control.maximum())
                self.wheel(control)
                self.assertEqual(control.value(), control.maximum())
                self.assertEqual(scroll.value(), before)
                control.setValue(control.minimum())
                self.wheel(control, -120)
                self.assertEqual(control.value(), control.minimum())
                self.assertEqual(scroll.value(), before)
                QTest.mouseClick(self.window.audio_status, Qt.MouseButton.LeftButton)
                self.wheel(control)
                self.assertEqual(control.value(), control.minimum())
                self.assertLess(scroll.value(), before)
                self.window.scroll_area.ensureWidgetVisible(control)
                self.app.processEvents()
                before = scroll.value()
                control.clearFocus()
                control.setFocus(Qt.FocusReason.TabFocusReason)
                self.app.processEvents()
                self.wheel(control)
                self.assertGreater(control.value(), control.minimum())
                self.assertEqual(scroll.value(), before)
                control.clearFocus()
                value = control.value()
                self.wheel(control)
                self.assertEqual(control.value(), value)

    def test_wheel_outside_armed_control_scrolls_page(self):
        self.select_custom_voice()
        self.device_library()
        self.window.resize(480, 400)
        self.refresh_devices()
        control = self.window.pitch_input
        self.window.scroll_area.ensureWidgetVisible(control)
        self.app.processEvents()
        QTest.mouseClick(control, Qt.MouseButton.LeftButton)
        before = self.window.scroll_area.verticalScrollBar().value()
        value = control.value()
        self.wheel(control, position=QPoint(-10, -10))
        self.assertEqual(control.value(), value)
        self.assertLess(self.window.scroll_area.verticalScrollBar().value(), before)

    def test_numeric_editor_click_typing_arrows_and_wheel_keep_page_still(self):
        self.select_custom_voice()
        self.device_library()
        self.window.resize(480, 400)
        self.refresh_devices()
        control = self.window.pitch_input
        self.window.scroll_area.ensureWidgetVisible(control)
        self.app.processEvents()
        scroll = self.window.scroll_area.verticalScrollBar()
        before = scroll.value()
        editor = control.lineEdit()
        QTest.mouseClick(editor, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertEqual(scroll.value(), before)
        editor.selectAll()
        QTest.keyClicks(editor, "2.555")
        QTest.keyClick(editor, Qt.Key.Key_Return)
        self.assertEqual(control.value(), 2.555)
        self.assertEqual(self.window.pitch_slider.value(), 26)
        QTest.keyClick(control, Qt.Key.Key_Up)
        self.assertGreater(control.value(), 2.555)
        value = control.value()
        self.wheel(control)
        self.assertGreater(control.value(), value)
        self.assertEqual(scroll.value(), before)

    def test_monitor_position_is_stable_and_keyboard_focus_is_revealed(self):
        self.device_library()
        self.window.resize(1000, 760)
        self.refresh_devices()
        body = self.window.scroll_area.widget()
        monitor_y = self.window.monitor_panel.mapTo(body, QPoint()).y()
        for key in ("custom", "original"):
            button = self.window.preset_buttons[key]
            self.window.scroll_area.ensureWidgetVisible(button)
            self.app.processEvents()
            before = self.window.scroll_area.verticalScrollBar().value()
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            self.app.processEvents()
            self.assertEqual(self.window.monitor_panel.mapTo(body, QPoint()).y(), monitor_y)
            self.assertEqual(self.window.scroll_area.verticalScrollBar().value(), before)
        self.window.preset_buttons["custom"].click()
        self.window.resize(480, 400)
        self.app.processEvents()
        scroll = self.window.scroll_area.verticalScrollBar()
        scroll.setValue(0)
        self.window.output_gain_input.setFocus(Qt.FocusReason.TabFocusReason)
        self.app.processEvents()
        bounds = self.window.output_gain_input.rect().translated(
            self.window.output_gain_input.mapTo(self.window.scroll_area.viewport(), QPoint()),
        )
        self.assertTrue(self.window.scroll_area.viewport().rect().contains(bounds))
        self.assertGreater(scroll.value(), 0)
        self.window.preset_buttons["original"].click()
        self.app.processEvents()
        self.assertLessEqual(scroll.value(), scroll.maximum())
        self.assertTrue(self.window.rect().contains(
            self.window.stop_button.rect().translated(self.window.stop_button.mapTo(self.window, QPoint())),
        ))

    def test_restore_from_wide_window_does_not_reserve_empty_grid_columns(self):
        self.device_library()
        self.window.resize(1000, 760)
        self.refresh_devices()
        self.window.resize(480, 400)
        self.app.processEvents()
        self.window.device_toggle.click()
        self.window.preset_toggle.click()
        self.app.processEvents()
        self.assertEqual(self.window.device_selectors.columnStretch(3), 0)
        self.assertEqual(self.window.preset_grid.columnStretch(2), 0)
        self.assertGreater(self.window.input_combo.width(), 280)
        self.assertGreater(self.window.preset_buttons["original"].width(), 190)

    def test_startup_refresh_is_async_single_and_does_not_open_audio(self):
        library = self.device_library()
        library.release.clear()
        self.window.show()
        self.wait_for(library.entered.is_set)
        self.assertIn("Refreshing", self.window.device_status.text())
        self.assertFalse(self.window.refresh_button.isEnabled())
        self.window.refresh_button.click()
        self.window._request_refresh()
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        self.wait_for(lambda: bool(heartbeat))
        self.assertEqual([name for name, _ in library.calls], ["devices"])
        library.release.set()
        self.wait_for(lambda: not self.window.controller.busy)
        self.assertEqual([name for name, _ in library.calls], ["devices", "hostapis"])
        self.window.hide()
        self.window.show()
        self.app.processEvents()
        self.assertEqual(len(library.calls), 2)
        self.assertFalse(self.window.controller.owns_devices)
        self.assertEqual(self.window.controller.state, EngineState.STOPPED)
        self.assertEqual(self.window.preset_combo.currentData(), "original")
        self.assertIn("未采集", self.window.input_level.display_text)
        self.assertIsNone(self.window.controller.device_selection.input_device)
        self.window.refresh_button.click()
        self.wait_for(lambda: not self.window.controller.busy)
        self.assertEqual(len(library.calls), 4)

    def test_startup_empty_and_failed_discovery_are_visible_and_retryable(self):
        library = self.device_library()
        library.error = library.PortAudioError("Synthetic discovery failure")
        self.refresh_devices()
        self.assertIn("查找失败", self.window.device_status.text())
        self.assertIn("Synthetic discovery failure", self.window.error_notice.text())
        self.assertTrue(self.window.refresh_button.isEnabled())
        library.error = None
        library.devices = []
        library.hostapis = []
        self.refresh_devices()
        self.assertIn("No usable devices", self.window.device_status.text())
        self.assertFalse(self.window.listen_button.isEnabled())
        self.assertFalse(self.window.record_button.isEnabled())

    def test_close_button_waits_for_controller_exit(self):
        self.window.show()
        self.window.close_button.click()
        self.wait_for(lambda: self.window.controller.is_closed)
        self.assertFalse(self.window.isVisible())

    def test_playback_track_status_uses_literal_track_values_without_qt_slot_errors(self):
        for track, name, english in (
            ("original", "原声", "Original"),
            ("effect", "效果", "Effect"),
        ):
            with self.subTest(track=track):
                self.window._on_media_status(MediaStatus(playback=track))
                self.assertIn(f"正在播放{name}", self.window.recording_status.text())
                self.assertNotIn(english, self.window.output_level.accessibleName())
                self.assertNotIn(english, self.window.output_level.toolTip())

    def test_unchanged_capture_snapshot_is_not_appended_again(self):
        controller = self.window.controller
        self.window.show()
        self.wait_for(lambda: controller.is_ready and not controller.busy)
        controller._state = EngineState.RUNNING
        capture = MediaStatus(monitoring=True, meter_source_id=1)
        controller._media_status = capture
        try:
            self.window._on_media_status(capture)
            levels = (0.2, 0.3)
            self.window._on_live_status((levels, 0, 0))
            for meter in (self.window.input_level, self.window.output_level):
                meter._sample()
            before = (self.window.input_level.history, self.window.output_level.history)
            self.window._on_live_status((levels, 0, 0))
            for meter in (self.window.input_level, self.window.output_level):
                meter._sample()
            self.assertEqual(
                (self.window.input_level.history, self.window.output_level.history), before,
            )

            next_levels = tuple([0.2, 0.3])
            self.assertIsNot(next_levels, levels)
            self.window._on_live_status((next_levels, 0, 0))
            for meter, history in zip(
                (self.window.input_level, self.window.output_level), before,
            ):
                meter._sample()
                self.assertEqual(len(meter.history), len(history) + 1)
        finally:
            controller._state = EngineState.STOPPED
            controller._media_status = MediaStatus()
            self.window._on_state(EngineState.STOPPED)

    def test_unchanged_playback_snapshot_is_not_appended_again(self):
        controller = self.window.controller
        self.window.show()
        self.wait_for(lambda: controller.is_ready and not controller.busy)
        controller._state = EngineState.RUNNING
        playback = MediaStatus(
            playback="original", meter_source_id=1,
            output_peak=0.25, output_peak_sequence=1,
        )
        controller._media_status = playback
        try:
            self.window._on_media_status(playback)
            self.window.output_level._sample()
            before = self.window.output_level.history
            self.window._on_media_status(playback)
            self.window.output_level._sample()
            self.assertEqual(self.window.output_level.history, before)

            playback = replace(playback, output_peak_sequence=2)
            controller._media_status = playback
            self.window._on_media_status(playback)
            self.window.output_level._sample()
            self.assertEqual(len(self.window.output_level.history), len(before) + 1)
        finally:
            controller._state = EngineState.STOPPED
            controller._media_status = MediaStatus()
            self.window._on_state(EngineState.STOPPED)

    def test_each_capture_channel_distinguishes_unknown_invalid_and_silence(self):
        controller = self.window.controller
        self.window.show()
        self.wait_for(lambda: controller.is_ready and not controller.busy)
        controller._state = EngineState.RUNNING
        capture = MediaStatus(monitoring=True, meter_source_id=1)
        controller._media_status = capture
        try:
            self.window._on_media_status(capture)
            self.window._on_live_status(((None, 0.0), 0, 0))
            self.assertIn("Waiting", self.window.input_level.display_text)
            self.assertEqual(self.window.output_level.display_text, "−∞ dBFS")
            self.window._on_live_status(((0.1, float("nan")), 0, 0))
            self.assertEqual(self.window.input_level.display_text, "-20.0 dBFS")
            self.assertIn("Invalid", self.window.output_level.display_text)
            self.assertEqual(self.window.output_level.history, ())
        finally:
            controller._state = EngineState.STOPPED
            controller._media_status = MediaStatus()
            self.window._on_state(EngineState.STOPPED)

    def test_late_capture_snapshots_do_not_contaminate_restart_or_playback(self):
        controller = self.window.controller
        self.window.show()
        self.wait_for(lambda: controller.is_ready and not controller.busy)
        controller._state = EngineState.RUNNING
        controller._live_control = LiveControl(PRESETS["original"])
        original = MediaStatus(monitoring=True, meter_source_id=1)
        controller._media_status = original
        try:
            self.window._on_media_status(original)
            old_snapshot = _PolledStatus(original, ((0.2, 0.3), 0, 0, 1))
            controller._polled_status.publish(old_snapshot)
            controller._receive_polled_status()
            self.window.input_level._sample()
            self.assertEqual(self.window.input_level.peak, 0.2)

            controller._state = EngineState.STOPPED
            self.window._on_state(EngineState.STOPPED)
            self.assertEqual(self.window.input_level.history, ())
            controller._polled_status.publish(old_snapshot)
            controller._receive_polled_status()
            self.assertEqual(self.window.input_level.history, ())

            controller._state = EngineState.RUNNING
            restarted = MediaStatus(monitoring=True, meter_source_id=2)
            controller._on_media_status(restarted)
            controller._polled_status.publish(old_snapshot)
            controller._receive_polled_status()
            controller._on_live_status(((0.8, 0.9), 0, 0, 1))
            self.assertIn("Waiting", self.window.input_level.display_text)
            self.assertEqual(self.window.input_level.history, ())

            controller._polled_status.publish(
                _PolledStatus(restarted, ((0.4, 0.5), 0, 0, 2)),
            )
            controller._receive_polled_status()
            self.window.input_level._sample()
            self.assertEqual(self.window.input_level.peak, 0.4)

            playback = MediaStatus(
                playback="effect", meter_source_id=3,
                output_peak=0.1, output_peak_sequence=1,
            )
            controller._on_media_status(playback)
            self.window.output_level._sample()
            history = self.window.output_level.history
            controller._polled_status.publish(
                _PolledStatus(restarted, ((0.8, 0.9), 0, 0, 2)),
            )
            controller._receive_polled_status()
            controller._on_live_status(((0.8, 0.9), 0, 0, 2))
            self.assertIsNone(self.window.input_level.peak)
            self.assertEqual(self.window.output_level.history, history)
            self.assertEqual(self.window.output_level.peak, 0.1)
        finally:
            controller._state = EngineState.STOPPED
            controller._live_control = None
            controller._media_status = MediaStatus()
            self.window._on_state(EngineState.STOPPED)

    def test_gui_status_to_both_painted_waves_p95_with_100_updates(self):
        controller = self.window.controller
        self.window.resize(1280, 731)
        self.window.show()
        self.wait_for(lambda: controller.is_ready and not controller.busy)
        self.window.scroll_area.ensureWidgetVisible(self.window.output_level, 0, 0)
        QTest.qWait(30)
        meters = (self.window.input_level, self.window.output_level)
        for meter in meters:
            bounds = meter.canvas.rect().translated(
                meter.canvas.mapTo(self.window.scroll_area.viewport(), QPoint(0, 0)),
            )
            self.assertTrue(self.window.scroll_area.viewport().rect().contains(bounds))

        class PaintWatcher(QObject):
            def __init__(self):
                super().__init__()
                self.painted = {}

            def eventFilter(self, watched, event):
                if event.type() == QEvent.Type.Paint:
                    meter = next(meter for meter in meters if watched is meter.canvas)
                    if meter.history:
                        peak = meter.history[-1][1]
                        QTimer.singleShot(
                            0,
                            lambda meter=meter, peak=peak: self.painted.__setitem__(
                                meter, (peak, time.monotonic()),
                            ),
                        )
                return False

        watcher = PaintWatcher()
        for meter in meters:
            meter.canvas.installEventFilter(watcher)
        controller._state = EngineState.RUNNING
        capture = MediaStatus(monitoring=True, meter_source_id=1)
        controller._media_status = capture
        self.window._on_media_status(capture)
        latencies = {meter: [] for meter in meters}
        try:
            for index in range(100):
                levels = (0.01 + index * 0.001, 0.21 + index * 0.001)
                sent = []

                def deliver():
                    self.assertEqual(QThread.currentThread(), self.app.thread())
                    sent.append(time.monotonic())
                    controller.liveStatus.emit((levels, 0, 0))

                QTimer.singleShot(0, deliver)
                self.assertTrue(wait_until(
                    lambda: sent and all(
                        meter in watcher.painted
                        and watcher.painted[meter][0] == peak
                        and watcher.painted[meter][1] >= sent[0]
                        for meter, peak in zip(meters, levels)
                    ),
                    timeout=0.5,
                ))
                for meter in meters:
                    latencies[meter].append(watcher.painted[meter][1] - sent[0])

            p95 = {
                meter: sorted(values)[math.ceil(len(values) * 0.95) - 1]
                for meter, values in latencies.items()
            }
            print(
                "W05: 100 GUI status injections; completed paint p95 "
                f"input={p95[meters[0]] * 1000:.1f} ms, "
                f"output={p95[meters[1]] * 1000:.1f} ms"
            )
            for meter in meters:
                self.assertEqual(len(latencies[meter]), 100)
                self.assertLessEqual(p95[meter], 0.150)
        finally:
            for meter in meters:
                meter.canvas.removeEventFilter(watcher)
            controller._state = EngineState.STOPPED
            controller._media_status = MediaStatus()
            self.window._on_state(EngineState.STOPPED)

    def test_analysis_gui_delivery_to_both_paints_p95_with_100_updates(self):
        window, controller = self.window, self.window.controller
        window.resize(1280, 731)
        window.show()
        self.wait_for(lambda: controller.is_ready and not controller.busy)
        window.wave_layout_button.setChecked(True)
        window.scroll_area.ensureWidgetVisible(window.wave_comparison, 0, 0)
        QTest.qWait(30)
        meters = (window.input_level, window.output_level)
        for meter in meters:
            bounds = meter.analysis_canvas.rect().translated(
                meter.analysis_canvas.mapTo(window.scroll_area.viewport(), QPoint(0, 0)),
            )
            self.assertTrue(window.scroll_area.viewport().rect().contains(bounds), str(bounds))

        class PaintWatcher(QObject):
            def __init__(self):
                super().__init__()
                self.painted = {}

            def eventFilter(self, watched, event):
                if event.type() == QEvent.Type.Paint:
                    meter = next(meter for meter in meters if watched is meter.analysis_canvas)
                    reading = meter.analysis_canvas.reading
                    if reading is not None:
                        QTimer.singleShot(
                            0,
                            lambda meter=meter, pitch=reading.pitch_hz: self.painted.__setitem__(
                                meter, (pitch, time.monotonic()),
                            ),
                        )
                return False

        watcher = PaintWatcher()
        for meter in meters:
            meter.analysis_canvas.installEventFilter(watcher)
        controller._state = EngineState.RUNNING
        capture = MediaStatus(monitoring=True, meter_source_id=1)
        controller._media_status = capture
        window._on_media_status(capture)
        latencies = {meter: [] for meter in meters}
        try:
            for index in range(100):
                pitches = (100.0 + index, 250.0 + index)
                readings = tuple(AnalysisReading((0.01, 0.1, 0.03), pitch) for pitch in pitches)
                status = replace(capture, analysis=AnalysisSnapshot(index + 1, 48000, *readings))
                sent = []

                def deliver():
                    self.assertEqual(QThread.currentThread(), self.app.thread())
                    sent.append(time.monotonic())
                    controller.mediaStatus.emit(status)

                QTimer.singleShot(0, deliver)
                self.assertTrue(wait_until(
                    lambda: sent and all(
                        meter in watcher.painted
                        and watcher.painted[meter][0] == pitch
                        and watcher.painted[meter][1] >= sent[0]
                        for meter, pitch in zip(meters, pitches)
                    ),
                    timeout=0.5,
                ))
                for meter in meters:
                    latencies[meter].append(watcher.painted[meter][1] - sent[0])
            p95 = {
                meter: sorted(values)[math.ceil(len(values) * 0.95) - 1]
                for meter, values in latencies.items()
            }
            print(
                "Analysis: 100 GUI status injections; completed paint p95 "
                f"input={p95[meters[0]] * 1000:.1f} ms, "
                f"output={p95[meters[1]] * 1000:.1f} ms"
            )
            for meter in meters:
                self.assertEqual(len(latencies[meter]), 100)
                self.assertLessEqual(p95[meter], 0.150)
        finally:
            for meter in meters:
                meter.analysis_canvas.removeEventFilter(watcher)
            controller._state = EngineState.STOPPED
            controller._media_status = MediaStatus()
            window._on_state(EngineState.STOPPED)

    def test_peak_envelopes_follow_capture_recording_playback_and_fault(self):
        controller = self.window.controller
        self.window.show()
        self.wait_for(lambda: controller.is_ready and not controller.busy)
        controller._state = EngineState.RUNNING
        capture = MediaStatus(monitoring=True, meter_source_id=1)
        controller._media_status = capture
        try:
            self.window._on_state(EngineState.RUNNING)
            self.window._on_media_status(capture)
            self.assertIn("Waiting", self.window.input_level.display_text)
            self.assertIn("Waiting", self.window.output_level.display_text)
            self.window._on_live_status(((0.01, 0.1), 0, 0))
            self.assertEqual(self.window.input_level.display_text, "-40.0 dBFS")
            self.assertEqual(self.window.output_level.display_text, "-20.0 dBFS")
            self.window.input_level._sample()
            original_history = self.window.input_level.history

            recording = replace(
                capture, recording=True, recording_state=RecordingState.RECORDING,
            )
            controller._media_status = recording
            self.window._on_media_status(recording)
            recording = replace(recording, monitoring=False)
            controller._media_status = recording
            self.window._on_media_status(recording)
            self.assertEqual(self.window.input_level.history, original_history)
            self.assertIn("监听关闭", self.window.audio_status.text())
            self.assertEqual(self.window.output_level.peak, 0.1)

            restarted = replace(recording, meter_source_id=2)
            controller._media_status = restarted
            self.window._on_media_status(restarted)
            self.assertEqual(self.window.input_level.history, ())
            self.assertIn("Waiting", self.window.input_level.display_text)
            self.window._on_live_status(((0.2, 0.3), 0, 0))
            self.assertEqual(self.window.input_level.peak, 0.2)

            playback = MediaStatus(playback="original", meter_source_id=3)
            controller._media_status = playback
            self.window._on_media_status(playback)
            self.assertIn("No capture", self.window.input_level.display_text)
            self.assertIn("Waiting", self.window.output_level.display_text)
            self.window._on_live_status(((0.8, 0.9), 0, 0))
            self.assertIsNone(self.window.input_level.peak)
            self.assertIsNone(self.window.output_level.peak)
            playback = replace(playback, output_peak=0.0)
            controller._media_status = playback
            self.window._on_media_status(playback)
            self.assertEqual(self.window.output_level.display_text, "−∞ dBFS")
            self.assertIn("回放音量", self.window.output_level.accessibleName())
            playback = replace(playback, playback="effect", meter_source_id=4, output_peak=None)
            controller._media_status = playback
            self.window._on_media_status(playback)
            self.assertIn("Waiting", self.window.output_level.display_text)
            self.assertEqual(self.window.output_level.history, ())

            controller._state = EngineState.FAULT
            self.window._on_state(EngineState.FAULT)
            self.window._on_live_status(((0.4, 0.4), 0, 0))
            self.assertIn("No capture", self.window.input_level.display_text)
            self.assertIn("No capture", self.window.output_level.display_text)
        finally:
            controller._state = EngineState.STOPPED
            controller._media_status = MediaStatus()
            self.window._on_state(EngineState.STOPPED)

    def test_analysis_layout_and_source_lifecycle_are_independent(self):
        window, controller = self.window, self.window.controller
        window.show()
        self.wait_for(lambda: controller.is_ready and not controller.busy)
        controller._state = EngineState.RUNNING
        first = AnalysisReading((0.01, 0.1, 0.03), 220)
        second = AnalysisReading((0.02, 0.04, 0.2), 330)
        snapshot = AnalysisSnapshot(1, 48000, first, second)
        capture = MediaStatus(monitoring=True, meter_source_id=1)
        try:
            with patch("voice_changer.ui.audio_wave.monotonic", return_value=1.0) as clock:
                controller._on_media_status(capture)
                self.assertIn("Waiting", window.input_level.analysis_status.text())
                controller._on_media_status(replace(capture, analysis=snapshot))
                window._on_live_status(((0.1, 0.2), 0, 0))
                for meter in (window.input_level, window.output_level):
                    meter._sample()
                self.assertIn("220.0 Hz", window.input_level.analysis_status.text())
                self.assertIn("330.0 Hz", window.output_level.analysis_status.text())
                self.assertEqual(window.input_level.pitch_history[-1][1], 220)
                self.assertEqual(window.output_level.pitch_history[-1][1], 330)
                histories = tuple(
                    (meter.history, meter.pitch_history)
                    for meter in (window.input_level, window.output_level)
                )
                self.assertFalse(window.wave_comparison.side_by_side)
                window.wave_layout_button.click()
                QTest.qWait(30)
                self.assertTrue(window.wave_comparison.side_by_side)
                self.assertTrue(window.wave_layout_button.isChecked())
                self.assertIn("上下", window.wave_layout_button.accessibleName())
                self.assertEqual(histories, tuple(
                    (meter.history, meter.pitch_history)
                    for meter in (window.input_level, window.output_level)
                ))
                self.assertLess(window.input_level.x(), window.output_level.x())
                window.wave_layout_button.click()
                QTest.qWait(30)
                self.assertFalse(window.wave_comparison.side_by_side)
                self.assertEqual(histories, tuple(
                    (meter.history, meter.pitch_history)
                    for meter in (window.input_level, window.output_level)
                ))

                recording = replace(
                    capture, recording=True, recording_state=RecordingState.RECORDING,
                    analysis=snapshot,
                )
                controller._on_media_status(recording)
                controller._on_media_status(replace(recording, monitoring=False))
                self.assertIn("监听关闭", window.audio_status.text())
                self.assertEqual(histories, tuple(
                    (meter.history, meter.pitch_history)
                    for meter in (window.input_level, window.output_level)
                ))

                clock.return_value = 1.2
                dropped = AnalysisSnapshot(2, 48000, None, None, 3)
                controller._on_media_status(replace(recording, monitoring=False, analysis=dropped))
                self.assertEqual(window.input_level.pitch_history, ())
                self.assertEqual(window.output_level.pitch_history, ())
                self.assertEqual(window.input_level.history, histories[0][0])
                self.assertIn("skipped 3 blocks", window.input_level.analysis_status.text())
                controller._on_media_status(replace(
                    recording, monitoring=False, analysis=AnalysisSnapshot(3, 48000, first, second, 3),
                ))
                for meter in (window.input_level, window.output_level):
                    meter._sample()
                self.assertIn("skipped 3 blocks", window.output_level.analysis_status.text())

                restarted = replace(capture, meter_source_id=2)
                controller._on_media_status(restarted)
                self.assertEqual(window.input_level.pitch_history, ())
                self.assertIn("Waiting", window.input_level.analysis_status.text())
                controller._on_media_status(replace(recording, analysis=snapshot))
                self.assertEqual(controller.media_status.meter_source_id, 2)
                self.assertEqual(window.input_level.pitch_history, ())

                for source_id, track, value in ((3, "original", 176), (4, "effect", 440)):
                    clock.return_value += 0.2
                    playback = MediaStatus(playback=track, meter_source_id=source_id)
                    controller._on_media_status(playback)
                    self.assertIn("No capture", window.input_level.analysis_status.text())
                    self.assertIn("Waiting", window.output_level.analysis_status.text())
                    result = AnalysisReading((0.1, 0.02, 0.01), value)
                    controller._on_media_status(replace(
                        playback, output_peak=0.0, output_peak_sequence=1,
                        analysis=AnalysisSnapshot(1, 48000, None, result),
                    ))
                    window.output_level._sample()
                    self.assertEqual(window.output_level.pitch_history[-1][1], value)
                    self.assertIn("频带与音高", window.output_level.analysis_canvas.accessibleName())
                    self.assertEqual(window.input_level.pitch_history, ())
                    self.assertEqual(window.output_level.display_text, "−∞ dBFS")
                    controller._on_media_status(replace(capture, analysis=snapshot))
                    self.assertEqual(controller.media_status.meter_source_id, source_id)

                controller._state = EngineState.STOPPED
                window._on_state(EngineState.STOPPED)
                controller._on_media_status(replace(playback, analysis=AnalysisSnapshot(2, 48000, None, second)))
                self.assertEqual(window.output_level.pitch_history, ())
                self.assertIn("未采集", window.output_level.analysis_status.text())
                controller._state = EngineState.RUNNING
                controller._on_media_status(replace(capture, meter_source_id=5, analysis=snapshot))
                self.assertIn("220.0 Hz", window.input_level.analysis_status.text())
                controller._state = EngineState.FAULT
                window._on_state(EngineState.FAULT)
                self.assertEqual(window.input_level.pitch_history, ())
                self.assertIn("No capture", window.input_level.analysis_status.text())
        finally:
            controller._state = EngineState.STOPPED
            controller._media_status = MediaStatus()
            window._on_state(EngineState.STOPPED)

    def test_live_status_keeps_footer_buttons_stable_before_and_after_paint(self):
        window, controller = self.window, self.window.controller
        window.resize(1280, 731)
        window.show()
        self.wait_for(lambda: controller.is_ready and not controller.busy)
        controller._state = EngineState.RUNNING
        capture = MediaStatus(monitoring=True, meter_source_id=1)
        buttons = (window.listen_button, window.record_button, window.stop_button)
        footer = window._mode_footers["dsp"]
        watched = (*buttons, footer)

        class GeometryEvents(QObject):
            def __init__(self):
                super().__init__()
                self.changes = []

            def eventFilter(self, obj, event):
                if event.type() in (QEvent.Type.Move, QEvent.Type.Resize):
                    self.changes.append((obj, event.type()))
                return False

        observer = GeometryEvents()
        try:
            for width, height in ((480, 400), (1280, 731)):
                window.resize(width, height)
                for language in ("zh", "en"):
                    if window._language != language:
                        window._switch_language()
                    for mode in (
                        capture,
                        replace(capture, recording=True, recording_state=RecordingState.RECORDING),
                        MediaStatus(playback="generated", meter_source_id=2),
                    ):
                        controller._media_status = mode
                        window._on_media_status(mode)
                        QTest.qWait(80)
                        for widget in watched:
                            widget.installEventFilter(observer)
                        observer.changes.clear()
                        baseline = tuple(widget.geometry() for widget in watched)
                        for sequence in range(1, 13):
                            reading = AnalysisReading((0.01, 0.1, 0.03), 220.0)
                            media = replace(mode, analysis=AnalysisSnapshot(
                                sequence, 48000, reading, reading,
                            ))
                            controller._media_status = media
                            window._on_media_status(media)
                            self.assertNotIn(" / ", window.record_button.text())
                            self.assertNotIn(" / ", window.stop_button.text())
                            self.assertNotIn(" / ", window.listen_guidance.text())
                            self.assertEqual(tuple(widget.geometry() for widget in watched), baseline)
                            QTest.qWait(25)
                            self.assertEqual(tuple(widget.geometry() for widget in watched), baseline)
                        self.assertEqual(observer.changes, [], f"{width=} {language=} {mode=}: {observer.changes}")
                        for widget in watched:
                            widget.removeEventFilter(observer)
        finally:
            for widget in watched:
                widget.removeEventFilter(observer)
            controller._state = EngineState.STOPPED
            controller._media_status = MediaStatus()
            window._on_state(EngineState.STOPPED)

    def test_each_wave_and_the_fixed_stop_control_are_reachable_in_compact_layout(self):
        self.window.show()
        self.wait_for(lambda: self.window.controller.is_ready and not self.window.controller.busy)
        self.window.input_level.set_waiting()
        self.window.output_level.set_waiting()
        area = self.window.scroll_area
        for maximized, width, height in (
            (False, 480, 400), (False, 640, 440), (False, 1280, 731), (True, 0, 0),
        ):
            with self.subTest(size="maximized" if maximized else (width, height)):
                if maximized:
                    self.window.showMaximized()
                else:
                    self.window.resize(width, height)
                QTest.qWait(30)
                if maximized:
                    self.assertTrue(self.window.isMaximized())
                meters = (self.window.input_level, self.window.output_level)
                small_layout = not maximized and width < 800
                if small_layout:
                    self.assertTrue(self.window.analysis_toggle.isVisibleTo(self.window))
                    self.assertTrue(self.window.device_toggle.isVisibleTo(self.window))
                    self.assertTrue(self.window.preset_toggle.isVisibleTo(self.window))
                    self.assertTrue(self.window.recording_toggle.isVisibleTo(self.window))
                    self.assertFalse(self.window.device_selector_container.isVisible())
                    self.assertFalse(self.window.preset_options.isVisible())
                    self.assertFalse(self.window.recording_content.isVisible())
                    self.assertTrue(all(not meter.analysis_canvas.isVisible() for meter in meters))
                    for meter in meters:
                        bounds = meter.rect().translated(
                            meter.mapTo(area.viewport(), QPoint(0, 0)),
                        )
                        self.assertTrue(area.viewport().rect().contains(bounds), str(bounds))
                    for toggle, content in (
                        (self.window.analysis_toggle, self.window.input_level.analysis_canvas),
                        (self.window.device_toggle, self.window.device_selector_container),
                        (self.window.preset_toggle, self.window.preset_options),
                        (self.window.recording_toggle, self.window.recording_content),
                    ):
                        toggle.click()
                        self.assertTrue(content.isVisible())
                        toggle.click()
                        self.assertFalse(content.isVisible())
                else:
                    self.assertFalse(self.window.analysis_toggle.isVisible())
                    self.assertFalse(self.window.device_toggle.isVisible())
                    self.assertFalse(self.window.preset_toggle.isVisible())
                    self.assertFalse(self.window.recording_toggle.isVisible())
                    self.assertTrue(self.window.device_selector_container.isVisible())
                    self.assertTrue(self.window.preset_options.isVisible())
                    self.assertTrue(self.window.recording_content.isVisible())
                    self.assertTrue(all(meter.analysis_canvas.isVisible() for meter in meters))
                    for meter in meters:
                        area.ensureWidgetVisible(meter, 0, 0)
                        QTest.qWait(20)
                        bounds = meter.rect().translated(
                            meter.mapTo(area.viewport(), QPoint(0, 0)),
                        )
                        self.assertTrue(area.viewport().rect().contains(bounds), str(bounds))
                stop = self.window.stop_button
                bounds = stop.rect().translated(stop.mapTo(self.window.centralWidget(), QPoint(0, 0)))
                self.assertTrue(self.window.centralWidget().rect().contains(bounds))
                self.assertTrue(stop.isVisibleTo(self.window))
                self.assertEqual(area.horizontalScrollBar().maximum(), 0)

    def test_analysis_labels_and_errors_fit_both_layouts_at_small_sizes(self):
        window = self.window
        window.show()
        self.wait_for(lambda: window.controller.is_ready and not window.controller.busy)
        area = window.scroll_area
        for maximized, width, height in (
            (False, 480, 400), (False, 640, 440), (True, 0, 0),
        ):
            if maximized:
                window.showMaximized()
            else:
                window.showNormal()
                window.resize(width, height)
            for side in (False, True):
                with self.subTest(size="maximized" if maximized else (width, height), side=side):
                    window.wave_layout_button.setChecked(side)
                    QTest.qWait(30)
                    labels = _ANALYSIS_TEXT[window._language]
                    for meter in (window.input_level, window.output_level):
                        canvas = meter.analysis_canvas
                        font = canvas.font()
                        font.setPointSizeF(9)
                        metrics = QFontMetrics(font)
                        chart_width = canvas.width() if side else canvas.width() / 2 - 4
                        self.assertLessEqual(
                            metrics.horizontalAdvance(f'{labels["bands"]} · dBFS'), chart_width,
                        )
                        self.assertLessEqual(
                            metrics.horizontalAdvance(f'{labels["pitch"]} · 50–800 Hz'), chart_width,
                        )
                        for text in (*labels["band_names"], "-120.0", "−∞", "N/A"):
                            self.assertLessEqual(metrics.horizontalAdvance(text), chart_width / 3)
                        meter.set_analysis(AnalysisReading(None, None, error="synthetic analysis failure"))
                        QTest.qWait(20)
                        area.ensureWidgetVisible(meter.analysis_status, 0, 0)
                        QTest.qWait(20)
                        label = meter.analysis_status
                        bounds = label.rect().translated(label.mapTo(area.viewport(), QPoint()))
                        self.assertTrue(area.viewport().rect().contains(bounds), str(bounds))
                        self.assertGreaterEqual(label.height(), label.heightForWidth(label.width()))
                        self.assertGreater(label.contentsRect().left(), 3)
                        self.assertIn("synthetic analysis failure", label.text())
                        meter.set_idle()
                    stop_bounds = window.stop_button.rect().translated(
                        window.stop_button.mapTo(window, QPoint()),
                    )
                    self.assertTrue(window.rect().contains(stop_bounds))
                    self.assertEqual(area.horizontalScrollBar().maximum(), 0)

    def test_bilingual_playback_and_fault_text_fit_with_reachable_stop(self):
        controller = self.window.controller
        self.window.show()
        self.wait_for(lambda: controller.is_ready and not controller.busy)
        area = self.window.scroll_area
        try:
            for maximized, width, height in (
                (False, 480, 400), (False, 640, 440), (True, 0, 0),
            ):
                if maximized:
                    self.window.showMaximized()
                else:
                    self.window.showNormal()
                    self.window.resize(width, height)
                for track, chinese, english in (
                    ("original", "原声", "Original"),
                    ("effect", "效果", "Effect"),
                ):
                    with self.subTest(size="maximized" if maximized else (width, height), track=track):
                        controller._state = EngineState.RUNNING
                        media = MediaStatus(
                            playback=track, meter_source_id=1,
                            output_peak=0.1, output_peak_sequence=1,
                        )
                        controller._media_status = media
                        self.window._on_media_status(media)
                        QTest.qWait(30)
                        if maximized:
                            self.assertTrue(self.window.isMaximized())
                        else:
                            self.assertEqual((self.window.width(), self.window.height()), (width, height))
                        self.assertIn(chinese, self.window.output_level.name_label.text())
                        self.assertNotIn(english, self.window.output_level.name_label.text())
                        self.assertEqual(self.window.output_level.display_text, "-20.0 dBFS")
                        for meter in (self.window.input_level, self.window.output_level):
                            ensure_visible_after_layout(area, meter)
                            bounds = meter.rect().translated(meter.mapTo(area.viewport(), QPoint(0, 0)))
                            self.assertTrue(area.viewport().rect().contains(bounds))
                            for label in (meter.name_label, meter.reading_label):
                                self.assertGreaterEqual(
                                    label.height(), label.heightForWidth(label.width()), label.text(),
                                )
                        for label in (self.window.activity_status, self.window.recording_status):
                            self.assertGreaterEqual(
                                label.height(), label.heightForWidth(label.width()), label.text(),
                            )
                        self.assertEqual(area.horizontalScrollBar().maximum(), 0)
                        self.assertTrue(self.window.stop_button.isEnabled())

                        controller._state = EngineState.FAULT
                        self.window._on_state(EngineState.FAULT)
                        self.window._on_error(EngineError("synthetic capture failure"))
                        self.wait_for(lambda: not self.window._feedback_scroll_timer.isActive())
                        self.app.processEvents()
                        error = self.window.error_notice
                        self.assertIn("操作失败", error.text())
                        self.assertIn("synthetic capture failure", error.text())
                        self.assertGreaterEqual(error.height(), error.heightForWidth(error.width()))
                        option = QStyleOption()
                        option.initFrom(error)
                        self.assertEqual(
                            error.contentsRect(),
                            error.style().subElementRect(QStyle.SubElement.SE_FrameContents, option, error),
                        )
                        self.assertGreater(error.contentsRect().left(), 3)
                        error_bounds = error.rect().translated(error.mapTo(area.viewport(), QPoint(0, 0)))
                        self.assertTrue(area.viewport().rect().contains(error_bounds))
                        stop = self.window.stop_button
                        stop_bounds = stop.rect().translated(
                            stop.mapTo(self.window.centralWidget(), QPoint(0, 0)),
                        )
                        self.assertTrue(stop.isVisibleTo(self.window))
                        self.assertTrue(stop.isEnabled())
                        self.assertTrue(self.window.centralWidget().rect().contains(stop_bounds))
        finally:
            controller._state = EngineState.STOPPED
            controller._media_status = MediaStatus()
            self.window._on_state(EngineState.STOPPED)

    def test_keyboard_exit_waits_for_controller_exit(self):
        self.window.show()
        self.window.close_button.setFocus()
        QTest.keyClick(self.window.close_button, Qt.Key.Key_Space)
        self.wait_for(lambda: self.window.controller.is_closed)
        self.assertFalse(self.window.isVisible())

    def test_labels_fit_compact_and_default_windows(self):
        self.wait_for(lambda: self.window.controller.is_ready)
        for preset in ("original", "custom"):
            self.window.preset_buttons[preset].click()
            for width, height in ((480, 400), (640, 440), (720, 540), (1000, 760), (1280, 731)):
                with self.subTest(preset=preset, size=(width, height)):
                    self.window.resize(width, height)
                    self.window.show()
                    self.app.processEvents()
                    self.assertEqual((self.window.width(), self.window.height()), (width, height))
                    content = self.window.scroll_area.widget()
                    self.wait_for(lambda: all(
                        label.height() >= label.heightForWidth(label.width())
                        for label in content.findChildren(QLabel) if label.isVisibleTo(content)
                    ))
                    for label in content.findChildren(QLabel):
                        if not label.isVisibleTo(content):
                            continue
                        label_bounds = label.rect().translated(
                            label.mapTo(content, QPoint(0, 0)),
                        )
                        self.assertTrue(content.rect().contains(label_bounds))
                        self.assertGreaterEqual(
                            label.height(), label.heightForWidth(label.width()),
                            label.text(),
                        )
                    self.assertEqual(self.window.scroll_area.horizontalScrollBar().maximum(), 0)

    def test_module_runs_event_loop_closes_and_keeps_user_dirs_empty(self):
        script = """
import importlib.abc
import runpy
import sys
from unittest.mock import patch
from PySide6.QtCore import QTimer

class NoAudioOrPrototype(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("sounddevice", "numpy", "voice_changer.audio.sounddevice_backend",
                                "voice_changer.dsp", "voice_changer.ai",
                                "voice_changer.ui.prototype")):
            raise AssertionError("Unexpected runtime import: " + fullname)

sys.meta_path.insert(0, NoAudioOrPrototype())
from device_fakes import FakeDeviceLibrary
sys.modules["sounddevice"] = FakeDeviceLibrary()
from voice_changer.ui.main_window import MainWindow

class ClosingWindow(MainWindow):
    def showMaximized(self):
        super().showMaximized()
        assert self.isMaximized() and not self.isFullScreen()
        print("window shown maximized", flush=True)
        QTimer.singleShot(0, self.close)

with patch("voice_changer.ui.main_window.MainWindow", ClosingWindow):
    runpy.run_module("voice_changer", run_name="__main__")
"""
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT / "tests/integration")))
            for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR"):
                env[name] = directory
            result = subprocess.run(
                [sys.executable, "-B", "-c", script], cwd=ROOT, env=env,
                capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("window shown", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_state_and_error_are_visible_without_enabling_audio_controls(self):
        controller = self.use_test_backend()
        self.backend.start_error = OSError("<device unavailable>")
        self.assertTrue(controller.request_start())
        self.wait_for(lambda: controller.state is EngineState.FAULT and not controller.busy)
        self.assertIn("故障", self.window.activity_status.text())
        self.assertTrue(self.window.error_notice.isVisible())
        self.assertIn("<device unavailable>", self.window.error_notice.text())
        self.assertEqual(self.window.error_notice.textFormat(), Qt.TextFormat.PlainText)

    def test_busy_close_keeps_gui_responsive_until_cleanup_completes(self):
        controller = self.use_test_backend()
        self.backend.start_release.clear()
        controller.request_start()
        self.wait_for(self.backend.start_entered.is_set)
        self.window.close()
        self.assertTrue(self.window.isVisible())
        self.assertFalse(self.window.close_button.isEnabled())
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        self.wait_for(lambda: bool(heartbeat))
        self.assertFalse(controller.is_closed)
        self.backend.start_release.set()
        self.wait_for(lambda: controller.is_closed)
        self.assertFalse(self.window.isVisible())

    def test_failed_close_retains_window_and_retry_releases_resources(self):
        controller = self.use_test_backend()
        controller.request_start()
        self.wait_for(lambda: controller.state is EngineState.RUNNING and not controller.busy)
        self.backend.close_error = OSError("close failed")
        self.window.close()
        self.wait_for(lambda: self.window.retry_button.isVisible())
        self.assertTrue(self.window.isVisible())
        self.assertFalse(controller.is_closed)
        self.assertIn("close failed", self.window.error_notice.text())
        self.assertIn("未完成", self.window.activity_status.text())
        self.backend.close_error = None
        self.window.retry_button.click()
        self.wait_for(lambda: controller.is_closed)
        self.assertFalse(self.window.isVisible())

    def test_long_shutdown_error_scrolls_without_clipping_or_hiding_exit(self):
        controller = self.use_test_backend()
        self.window.resize(480, 400)
        controller.request_start()
        self.wait_for(lambda: controller.state is EngineState.RUNNING and not controller.busy)
        message = "Synthetic cleanup diagnostic " * 100
        self.backend.close_error = OSError(message)
        self.window.close()
        self.wait_for(lambda: self.window.retry_button.isVisible())
        self.wait_for(lambda: not self.window._feedback_scroll_timer.isActive())
        self.app.processEvents()
        self.assertIn(message, self.window.error_notice.text())
        self.assertEqual((self.window.width(), self.window.height()), (480, 400))
        for label in self.window.findChildren(QLabel):
            if not label.isHidden():
                self.assertGreaterEqual(label.height(), label.heightForWidth(label.width()))
        self.assertEqual(self.window.scroll_area.horizontalScrollBar().maximum(), 0)
        scrollbar = self.window.scroll_area.verticalScrollBar()
        self.assertGreater(scrollbar.maximum(), 0)
        body = self.window.scroll_area.widget()
        viewport = self.window.scroll_area.viewport()
        error_bottom = self.window.error_notice.mapTo(
            body, self.window.error_notice.rect().bottomLeft(),
        ).y()
        scrollbar.setValue(error_bottom - viewport.height() + 10)
        self.app.processEvents()
        bottom = self.window.error_notice.mapTo(
            viewport, self.window.error_notice.rect().bottomLeft(),
        ).y()
        self.assertGreaterEqual(bottom, 0)
        self.assertLess(bottom, viewport.height())
        for button in (self.window.retry_button, self.window.close_button):
            bounds = button.rect().translated(
                button.mapTo(self.window.centralWidget(), QPoint(0, 0)),
            )
            self.assertTrue(self.window.centralWidget().rect().contains(bounds))
            self.assertTrue(button.isEnabled())
        self.backend.close_error = None
        QTest.keyClick(self.window.retry_button, Qt.Key.Key_Space)
        self.wait_for(lambda: controller.is_closed)
        self.assertFalse(self.window.isVisible())

    def test_explicit_refresh_filters_directions_and_keyboard_selects_without_audio(self):
        library = self.device_library()
        library.devices[1]["name"] = library.devices[0]["name"]
        self.refresh_devices()
        self.assertEqual([name for name, _ in library.calls], ["devices", "hostapis"])
        self.assertTrue(self.window.controller.devices_loaded)
        self.assertEqual(self.window.input_combo.count(), 2)
        self.assertEqual(self.window.output_combo.count(), 2)
        self.assertIsNone(self.window.input_combo.currentData())
        self.assertIsNone(self.window.output_combo.currentData())
        self.window.input_combo.setFocus()
        QTest.keyClick(self.window.input_combo, Qt.Key.Key_Down)
        self.window.output_combo.setFocus()
        QTest.keyClick(self.window.output_combo, Qt.Key.Key_Down)
        selection = self.window.controller.device_selection
        self.assertEqual(selection.input_device.index, 0)
        self.assertEqual(selection.output_device.index, 1)
        self.assertEqual(selection.input_device.name, selection.output_device.name)
        self.assertIn("test input", self.window.device_details.toPlainText())
        self.assertTrue(self.window.device_details.isReadOnly())
        self.assertEqual(self.window.input_label.buddy(), self.window.input_combo)
        self.assertEqual(self.window.output_label.buddy(), self.window.output_combo)

    def test_empty_success_is_distinct_from_unrefreshed_without_fake_options(self):
        library = self.device_library()
        library.devices = []
        library.hostapis = []
        self.refresh_devices()
        self.assertTrue(self.window.controller.devices_loaded)
        self.assertIn("No usable devices found", self.window.device_status.text())
        self.assertFalse(self.window.input_combo.isEnabled())
        self.assertFalse(self.window.output_combo.isEnabled())
        self.assertFalse(self.window.device_details.isVisible())
        self.assertTrue(self.window.refresh_button.isEnabled())

    def test_query_failure_preserves_last_list_and_explicit_retry_shows_results(self):
        self.window.resize(480, 400)
        library = self.device_library()
        self.refresh_devices()
        self.window.input_combo.setFocus()
        QTest.keyClick(self.window.input_combo, Qt.Key.Key_Down)
        selected = self.window.controller.device_selection.input_device
        library.error = OSError("synthetic refresh failure")
        self.refresh_devices()
        self.assertTrue(self.window.error_notice.isVisible())
        self.assertIn("synthetic refresh failure", self.window.error_notice.text())
        self.assertEqual(self.window.input_combo.currentData(), selected)
        self.assertIn("查找失败", self.window.device_status.text())
        self.wait_for(lambda: not self.window._feedback_scroll_timer.isActive())
        viewport = self.window.scroll_area.viewport()
        error_bounds = self.window.error_notice.rect().translated(
            self.window.error_notice.mapTo(viewport, QPoint(0, 0)),
        )
        self.assertTrue(
            viewport.rect().contains(error_bounds),
            f"error={error_bounds}, viewport={viewport.rect()}, "
            f"scroll={self.window.scroll_area.verticalScrollBar().value()}/"
            f"{self.window.scroll_area.verticalScrollBar().maximum()}, "
            f"body={self.window.scroll_area.widget().size()}, "
            f"content_pos={self.window.error_notice.mapTo(self.window.scroll_area.widget(), QPoint(0, 0))}",
        )
        library.error = None
        self.refresh_devices()
        self.assertFalse(self.window.error_notice.isVisible())
        self.assertEqual(self.window.input_combo.currentData(), selected)

    def test_refresh_invalidation_clears_only_lost_direction_and_keeps_error_visible(self):
        library = self.device_library()
        self.refresh_devices()
        for combo in (self.window.input_combo, self.window.output_combo):
            combo.setFocus()
            QTest.keyClick(combo, Qt.Key.Key_Down)
        output = self.window.controller.device_selection.output_device
        library.devices[0]["max_input_channels"] = 0
        self.refresh_devices()
        self.assertIsNone(self.window.input_combo.currentData())
        self.assertFalse(self.window.input_combo.isEnabled())
        self.assertEqual(self.window.output_combo.currentData(), output)
        self.assertIn("Selected input", self.window.error_notice.text())
        self.assertTrue(self.window.error_notice.isVisible())

    def test_query_in_progress_locks_controls_and_window_waits_for_async_shutdown(self):
        library = self.device_library()
        library.release.clear()
        self.window.show()
        self.wait_for(lambda: self.window.controller.is_ready)
        self.window.refresh_button.click()
        self.wait_for(library.entered.is_set)
        self.assertIn("Refreshing devices", self.window.device_status.text())
        self.assertFalse(self.window.refresh_button.isEnabled())
        for editor in self.window.config_inputs:
            self.assertFalse(editor.isEnabled())
        self.assertFalse(self.window.validate_button.isEnabled())
        self.window.close()
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        self.wait_for(lambda: bool(heartbeat))
        self.assertTrue(self.window.isVisible())
        self.assertIn("Closing.", self.window.device_status.text())
        library.release.set()
        self.wait_for(lambda: self.window.controller.is_closed)
        self.assertFalse(self.window.isVisible())
        self.assertFalse(self.window.controller.devices_loaded)

    def test_keyboard_config_input_emits_preview_without_query_or_audio(self):
        library = self.device_library()
        self.select_config_pair()
        self.window.activateWindow()
        self.wait_for(self.window.isActiveWindow)
        configs = []
        self.window.controller.streamConfigReady.connect(configs.append)
        calls = list(library.calls)
        for editor, value in zip(self.window.config_inputs, ("32000", "256", "1", "0.125")):
            self.assertTrue(editor.text())
            editor.setFocus()
            editor.selectAll()
            QTest.keyClicks(editor, value)
            self.assertTrue(any(
                label.buddy() == editor for label in self.window.findChildren(QLabel)
            ))
        self.window.validate_button.setFocus()
        QTest.keyClick(self.window.validate_button, Qt.Key.Key_Space)
        self.wait_for(lambda: not self.window._feedback_scroll_timer.isActive())
        self.assertEqual(len(configs), 1)
        self.assertEqual((configs[0].samplerate, configs[0].blocksize,
                          configs[0].channels, configs[0].output_ceiling), (32000, 256, 1, 0.125))
        text = self.window.config_preview.toPlainText()
        for fragment in ("未应用", "32000 Hz", "256 帧", "0.125",
                         "test input", "test output"):
            self.assertIn(fragment, text)
        self.assertTrue(self.window.config_preview.isReadOnly())
        self.assertTrue(
            self.window.config_preview.textInteractionFlags()
            & Qt.TextInteractionFlag.TextSelectableByKeyboard,
        )
        self.assertTrue(self.window.config_preview.hasFocus())
        self.assertEqual(library.calls, calls)
        self.assertFalse(self.window.controller.busy)
        self.assertFalse(self.window.controller.owns_devices)
        self.assertEqual(self.window.controller.state, EngineState.STOPPED)
        QTest.keyClick(self.window.config_preview, Qt.Key.Key_Tab)
        self.assertFalse(self.window.config_preview.hasFocus())

    def test_config_parse_and_model_errors_show_field_without_success_or_fallback(self):
        library = self.device_library()
        self.select_config_pair()
        self.window.resize(480, 400)
        configs = []
        self.window.controller.streamConfigReady.connect(configs.append)
        calls = list(library.calls)
        for index, value, fragment in (
            (0, "", "samplerate"), (1, "", "blocksize"), (2, "", "channels"),
            (3, "", "output_ceiling"), (0, "48000.5", "samplerate"),
            (1, "0", "blocksize"), (2, "2", "too few channels"),
            (3, "NaN", "output_ceiling"), (3, "inf", "output_ceiling"),
            (3, "1.1", "output_ceiling"), (3, "<invalid & ceiling>", "output_ceiling"),
        ):
            with self.subTest(field=index, value=value):
                self.fill_config_fields()
                self.window.config_inputs[index].setText(value)
                self.validate_config()
                self.assertIn(fragment, self.window.error_notice.text())
                self.assertEqual(self.window.error_notice.textFormat(), Qt.TextFormat.PlainText)
                self.assertFalse(self.window.config_preview.isVisible())
                viewport = self.window.scroll_area.viewport()
                bounds = self.window.error_notice.rect().translated(
                    self.window.error_notice.mapTo(viewport, QPoint(0, 0)),
                )
                self.assertTrue(viewport.rect().contains(bounds))
        self.assertEqual(configs, [])
        self.assertEqual(library.calls, calls)
        self.assertFalse(self.window.controller.busy)
        self.fill_config_fields()
        self.validate_config()
        self.assertFalse(self.window.error_notice.isVisible())
        self.assertEqual(len(configs), 1)
        self.assertTrue(self.window.config_preview.isVisible())

    def test_edit_selection_and_refresh_clear_old_preview_without_auto_revalidation(self):
        library = self.device_library()
        self.select_config_pair()
        self.fill_config_fields()
        self.validate_config()
        self.assertTrue(self.window.config_preview.isVisible())
        self.window.prepare_backend_button.click()
        self.wait_for(lambda: not self.window.controller.busy)
        self.assertIn("output remains unconfirmed", self.window.backend_prepare_status.text())
        self.assertTrue(self.window.backend_prepare_status.isVisible())
        self.window.samplerate_input.setText("44100")
        self.assertEqual(self.window.config_preview.toPlainText(), "")
        self.assertFalse(self.window.config_preview.isVisible())
        self.assertEqual(self.window.backend_prepare_status.text(), "")
        self.assertFalse(self.window.backend_prepare_status.isVisible())
        self.validate_config()
        self.assertIn("44100 Hz", self.window.config_preview.toPlainText())
        controller = self.window.controller
        ref = controller.device_selection.input_device
        self.window.prepare_backend_button.click()
        self.wait_for(lambda: not controller.busy)
        self.assertTrue(self.window.backend_prepare_status.isVisible())
        controller.select_device("input", None)
        self.assertFalse(self.window.validate_button.isEnabled())
        self.assertEqual(self.window.config_preview.toPlainText(), "")
        self.assertFalse(self.window.backend_prepare_status.isVisible())
        controller.select_device("input", ref)
        self.assertFalse(self.window.config_preview.isVisible())
        self.validate_config()
        self.window.prepare_backend_button.click()
        self.wait_for(lambda: not controller.busy)
        self.assertTrue(self.window.backend_prepare_status.isVisible())
        self.refresh_devices()
        self.assertEqual(self.window.config_preview.toPlainText(), "")
        self.assertFalse(self.window.config_preview.isVisible())
        self.assertFalse(self.window.backend_prepare_status.isVisible())
        self.assertEqual(self.window.samplerate_input.text(), "44100")
        self.validate_config()
        library.error = OSError("synthetic query failure after preview")
        self.refresh_devices()
        self.assertIn("synthetic query failure after preview", self.window.error_notice.text())
        self.assertEqual(self.window.config_preview.toPlainText(), "")

    def test_explicit_backend_prepare_uses_current_preview_without_opening_audio(self):
        library = self.device_library()
        self.select_config_pair()
        self.fill_config_fields()
        self.validate_config()
        config = self.window._previewed_config
        self.assertIsNotNone(config)
        prepared = []
        controller = self.window.controller
        controller.backendPrepared.connect(prepared.append)
        calls = list(library.calls)

        self.assertTrue(self.window.prepare_backend_button.isEnabled())
        self.window.prepare_backend_button.click()
        self.wait_for(
            lambda: len(prepared) == 1 and not controller.busy
            and "output remains unconfirmed" in self.window.backend_prepare_status.text()
        )
        self.wait_for(lambda: not self.window._feedback_scroll_timer.isActive())

        self.assertEqual(prepared, [config])
        self.assertEqual(library.calls, calls)
        self.assertEqual(controller.state, EngineState.STOPPED)
        self.assertFalse(controller.owns_devices)
        self.assertFalse(self.window.prepare_backend_button.isEnabled())
        self.assertIn("没有打开音频流", self.window.backend_prepare_status.text())
        self.assertIn("没有采集或播放", self.window.backend_prepare_status.text())
        viewport = self.window.scroll_area.viewport()
        status_bounds = self.window.backend_prepare_status.rect().translated(
            self.window.backend_prepare_status.mapTo(viewport, QPoint(0, 0)),
        )
        self.assertTrue(viewport.rect().contains(status_bounds))

    def test_backend_prepare_failure_keeps_error_visible_and_allows_retry(self):
        library = self.device_library()
        self.select_config_pair()
        self.fill_config_fields()
        self.validate_config()
        controller = self.window.controller
        cause = OSError("synthetic backend construction failure")
        errors = []
        prepared = []
        controller.errorOccurred.connect(errors.append)
        controller.backendPrepared.connect(prepared.append)
        calls = list(library.calls)

        with patch.object(sounddevice_backend, "SoundDeviceBackend", side_effect=cause):
            self.window.prepare_backend_button.click()
            self.wait_for(
                lambda: not controller.busy
                and "synthetic backend construction failure" in self.window.error_notice.text()
            )

        self.assertEqual(len(errors), 1)
        self.assertIs(errors[0].__cause__, cause)
        self.assertTrue(self.window.config_preview.isVisible())
        self.assertFalse(self.window.backend_prepare_status.isVisible())
        self.assertTrue(self.window.prepare_backend_button.isEnabled())
        self.assertEqual(library.calls, calls)
        self.assertEqual(controller.state, EngineState.STOPPED)
        self.assertFalse(controller.owns_devices)

        self.window.prepare_backend_button.click()
        self.wait_for(
            lambda: len(prepared) == 1 and not controller.busy
            and "output remains unconfirmed" in self.window.backend_prepare_status.text()
        )
        self.assertFalse(self.window.error_notice.isVisible())

    def test_close_during_backend_prepare_discards_late_ui_result(self):
        library = self.device_library()
        self.select_config_pair()
        self.fill_config_fields()
        self.validate_config()
        controller = self.window.controller
        entered = threading.Event()
        release = threading.Event()
        constructor = sounddevice_backend.SoundDeviceBackend

        def delayed_constructor(config, *, output_confirmed):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test backend constructor was not released")
            return constructor(config, output_confirmed=output_confirmed)

        with patch.object(
            sounddevice_backend, "SoundDeviceBackend", side_effect=delayed_constructor,
        ):
            self.window.prepare_backend_button.click()
            self.wait_for(entered.is_set)
            self.assertTrue(controller.busy)
            self.assertFalse(self.window.prepare_backend_button.isEnabled())
            self.window.close()
            self.assertTrue(self.window._closing_requested)
            self.assertEqual(self.window.backend_prepare_status.text(), "")
            release.set()
            self.wait_for(lambda: controller.is_closed)

        self.assertFalse(self.window.isVisible())
        self.assertEqual(self.window.backend_prepare_status.text(), "")
        self.assertFalse(self.window.config_preview.isVisible())
        self.assertEqual([name for name, _ in library.calls], ["devices", "hostapis"])

    def test_config_controls_and_preview_follow_owned_stream_and_cleanup_failure(self):
        controller = self.use_test_backend()
        self.device_library()
        self.select_config_pair()
        self.fill_config_fields()
        self.validate_config()
        controller.request_start()
        self.wait_for(lambda: controller.state is EngineState.RUNNING and not controller.busy)
        for editor in self.window.config_inputs:
            self.assertFalse(editor.isEnabled())
        self.assertFalse(self.window.validate_button.isEnabled())
        self.assertEqual(self.window.config_preview.toPlainText(), "")
        self.backend.close_error = OSError("synthetic config cleanup failure")
        controller.request_stop()
        self.wait_for(lambda: controller.state is EngineState.FAULT and not controller.busy)
        self.assertFalse(self.window.validate_button.isEnabled())
        self.assertTrue(all(not editor.isEnabled() for editor in self.window.config_inputs))
        self.backend.close_error = None
        controller.request_stop()
        self.wait_for(lambda: controller.state is EngineState.STOPPED and not controller.busy)
        self.assertTrue(self.window.validate_button.isEnabled())
        self.assertFalse(self.window.config_preview.isVisible())
        self.validate_config()
        self.assertTrue(self.window.config_preview.isVisible())
        self.assertEqual(sum(action == "open" for action, _ in self.backend.events), 1)

    def test_long_config_preview_is_plain_keyboard_readable_and_fully_reachable(self):
        library = self.device_library()
        name = "<device & configuration> " * 100
        library.devices[0]["name"] = name
        self.select_config_pair()
        self.window.resize(480, 400)
        self.fill_config_fields()
        self.validate_config()
        preview = self.window.config_preview
        self.assertIn(name, preview.toPlainText())
        viewport = self.window.scroll_area.viewport()
        bounds = preview.rect().translated(preview.mapTo(viewport, QPoint(0, 0)))
        self.assertTrue(viewport.rect().contains(bounds))
        self.assertEqual(preview.horizontalScrollBar().maximum(), 0)
        self.assertEqual(self.window.scroll_area.horizontalScrollBar().maximum(), 0)
        self.assertEqual((self.window.width(), self.window.height()), (480, 400))
        preview.setFocus()
        QTest.keyClick(preview, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        self.assertIn(name, preview.textCursor().selectedText())
        QTest.keyClick(preview, Qt.Key.Key_Home, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(preview.textCursor().position(), 0)
        QTest.keyClick(preview, Qt.Key.Key_End, Qt.KeyboardModifier.ControlModifier)
        self.assertGreater(preview.verticalScrollBar().maximum(), 0)
        self.assertEqual(preview.textCursor().position(), preview.document().characterCount() - 1)
        self.assertTrue(preview.viewport().rect().contains(preview.cursorRect()))
        footer_bounds = self.window.close_button.rect().translated(
            self.window.close_button.mapTo(self.window.centralWidget(), QPoint(0, 0)),
        )
        self.assertTrue(self.window.centralWidget().rect().contains(footer_bounds))

    def test_reentrant_close_before_preview_delivery_discards_result(self):
        self.cleanup_window()
        controller = EngineController()
        controller.streamConfigReady.connect(lambda config: controller.request_shutdown())
        self.window = MainWindow(controller)
        self.device_library()
        self.select_config_pair()
        self.fill_config_fields()
        self.window.validate_button.click()
        self.wait_for(lambda: controller.is_closed)
        self.assertFalse(self.window.isVisible())
        self.assertEqual(self.window.config_preview.toPlainText(), "")

    def test_reentrant_selection_change_before_preview_delivery_reports_stale_result(self):
        self.cleanup_window()
        controller = EngineController()
        controller.streamConfigReady.connect(lambda config: controller.select_device("input", None))
        self.window = MainWindow(controller)
        self.device_library()
        self.select_config_pair()
        self.fill_config_fields()
        self.validate_config()
        self.assertIn("no longer current", self.window.error_notice.text())
        self.assertEqual(self.window.config_preview.toPlainText(), "")
        self.assertFalse(self.window.validate_button.isEnabled())
        self.assertIsNone(controller.device_selection.input_device)

    def test_owned_stream_and_failed_cleanup_keep_device_widgets_locked(self):
        controller = self.use_test_backend()
        self.device_library()
        self.refresh_devices()
        controller.request_start()
        self.wait_for(lambda: controller.state is EngineState.RUNNING and not controller.busy)
        for widget in (self.window.refresh_button, self.window.input_combo, self.window.output_combo):
            self.assertFalse(widget.isEnabled())
        self.backend.close_error = OSError("test close failed")
        controller.request_stop()
        self.wait_for(lambda: controller.state is EngineState.FAULT and not controller.busy)
        self.assertFalse(self.window.refresh_button.isEnabled())
        self.backend.close_error = None
        controller.request_stop()
        self.wait_for(lambda: controller.state is EngineState.STOPPED and not controller.busy)
        self.assertTrue(self.window.refresh_button.isEnabled())
        self.assertTrue(self.window.input_combo.isEnabled())

    def test_long_names_fit_and_full_plain_details_are_keyboard_readable(self):
        self.select_custom_voice()
        library = self.device_library()
        name = "<device & slot> " * 100
        library.devices[0]["name"] = name
        self.refresh_devices()
        self.window.resize(480, 400)
        self.window.input_combo.setFocus()
        QTest.keyClick(self.window.input_combo, Qt.Key.Key_Down)
        self.app.processEvents()
        self.assertEqual((self.window.width(), self.window.height()), (480, 400))
        self.assertEqual(self.window.scroll_area.horizontalScrollBar().maximum(), 0)
        self.assertEqual(self.window.device_details.horizontalScrollBar().maximum(), 0)
        self.assertIn(name, self.window.device_details.toPlainText())
        self.assertIn("&lt;device &amp; slot&gt;", self.window.input_combo.itemData(
            1, Qt.ItemDataRole.ToolTipRole,
        ))
        self.window.device_details.setFocus(Qt.FocusReason.TabFocusReason)
        self.app.processEvents()
        QTest.keyClick(self.window.device_details, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        self.assertIn(name, self.window.device_details.textCursor().selectedText())
        QTest.keyClick(self.window.device_details, Qt.Key.Key_Home, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(self.window.device_details.textCursor().position(), 0)
        QTest.keyClick(self.window.device_details, Qt.Key.Key_End, Qt.KeyboardModifier.ControlModifier)
        self.assertGreater(self.window.device_details.verticalScrollBar().maximum(), 0)
        self.assertEqual(
            self.window.device_details.textCursor().position(),
            self.window.device_details.document().characterCount() - 1,
        )
        self.assertTrue(self.window.device_details.viewport().rect().contains(
            self.window.device_details.cursorRect(),
        ))
        viewport = self.window.scroll_area.viewport()
        body = self.window.scroll_area.widget()
        details_bottom = self.window.device_details.mapTo(
            body, QPoint(0, 0),
        ).y() + self.window.device_details.height()
        self.window.scroll_area.verticalScrollBar().setValue(
            details_bottom - viewport.height() + 24,
        )
        self.app.processEvents()
        details_bounds = self.window.device_details.rect().translated(
            self.window.device_details.mapTo(viewport, QPoint(0, 0)),
        )
        self.assertTrue(
            viewport.rect().contains(details_bounds),
            f"details={details_bounds}, viewport={viewport.rect()}, "
            f"scroll={self.window.scroll_area.verticalScrollBar().value()}/"
            f"{self.window.scroll_area.verticalScrollBar().maximum()}",
        )
        QTest.keyClick(self.window.device_details, Qt.Key.Key_Tab)
        self.assertFalse(self.window.device_details.hasFocus())
        bounds = self.window.close_button.rect().translated(
            self.window.close_button.mapTo(self.window.centralWidget(), QPoint(0, 0)),
        )
        self.assertTrue(self.window.centralWidget().rect().contains(bounds))


if __name__ == "__main__":
    unittest.main()
