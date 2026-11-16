"""No hardware, desktop session, personal audio or network required."""

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QPoint, QPointF, QTimer, Qt
from PySide6.QtGui import QFont, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton, QScrollArea, QStyle, QStyleOption

from qt_wait import ensure_visible_after_layout
from voice_changer.ui.prototype import PrototypeWindow
from voice_changer.ui.prototype_i18n import LANGUAGES, TEXT, load_language, save_language, settings_path
from voice_changer.ui.prototype_state import (
    BUILTIN_PRESETS, FAULTS, MAX_CUSTOM_PRESETS, MAX_PRESET_NAME, PrototypeState,
)


class TempConfig:
    """Point XDG_CONFIG_HOME at a private directory for one test."""

    def start(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = self.directory.name

    def stop(self):
        if self.previous is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = self.previous
        self.directory.cleanup()


class StateTests(unittest.TestCase):
    def setUp(self):
        self.state = PrototypeState(input_device="usb_mic", output_device="headphones")

    def test_safe_defaults(self):
        s = PrototypeState()
        self.assertFalse(s.active)
        self.assertFalse(s.capturing)
        self.assertEqual(s.preset, "original")
        self.assertEqual(s.custom_presets, {})
        self.assertEqual((s.input_gain, s.output_gain), (0, 0))
        with self.assertRaises(ValueError):
            s.toggle_monitor()
        with self.assertRaises(ValueError):
            s.start_recording()

    def test_record_without_output_and_independent_monitoring(self):
        s = self.state
        s.output_device = ""
        s.start_recording()
        self.assertTrue(s.capturing)
        self.assertFalse(s.monitoring)
        with self.assertRaises(ValueError):
            s.toggle_monitor()
        s.stop_recording()
        s.reviewed_save = True
        s.output_device = "headphones"
        s.toggle_monitor()
        s.start_recording()
        s.toggle_monitor()
        self.assertTrue(s.recording)
        s.toggle_monitor()
        s.stop_recording()
        self.assertTrue(s.monitoring)
        self.assertFalse(s.recording)

    def test_playback_exclusion_track_restart_and_natural_end(self):
        s = self.state
        s.start_recording()
        s.tick()
        s.tick()
        with self.assertRaises(ValueError):
            s.play("original")
        s.stop_recording()
        s.toggle_monitor()
        s.play("original")
        self.assertFalse(s.capturing)
        s.tick()
        s.play("effect")
        self.assertEqual(s.position, 0)
        with self.assertRaises(ValueError):
            s.toggle_monitor()
        with self.assertRaises(ValueError):
            s.start_recording()
        s.tick()
        s.tick()
        self.assertFalse(s.active)
        self.assertFalse(s.monitoring)
        s.input_device = ""
        s.play("original")
        self.assertEqual(s.playing, "original")

    def test_device_lock_and_unsaved_guard(self):
        s = self.state
        s.start_recording()
        with self.assertRaises(ValueError):
            s.select_device("output", "")
        s.stop_all()
        with self.assertRaises(ValueError):
            s.start_recording()
        s.select_device("output", "")
        self.assertTrue(s.unsaved)
        with self.assertRaises(ValueError):
            s.select_device("input", "not-a-device")

    def test_presets_and_reset_preserve_gains_and_activity(self):
        s = self.state
        s.input_gain = 1
        s.output_gain = 2
        s.start_recording()
        for preset in BUILTIN_PRESETS:
            s.select_preset(preset)
            s.pitch = 4
            s.formant = 1
            s.custom = True
            s.reset_effect()
            self.assertEqual(s.preset, preset)
            self.assertEqual((s.pitch, s.formant, s.reverb), (0, 0, 0))
            self.assertEqual((s.input_gain, s.output_gain), (1, 2))
            self.assertTrue(s.recording)
            self.assertFalse(s.custom)

    def test_fault_stops_and_keeps_incomplete_take(self):
        s = self.state
        s.start_recording()
        s.toggle_monitor()
        s.fail("write_failed")
        self.assertFalse(s.active)
        self.assertTrue(s.incomplete)
        self.assertTrue(s.unsaved)
        with self.assertRaises(ValueError):
            s.play("original")
        s.fault = ""
        self.assertFalse(s.active)

    def test_custom_preset_save_select_reset_rename_delete(self):
        s = self.state
        s.input_gain = 2
        s.select_preset("m2f")
        s.pitch, s.formant, s.reverb, s.custom = 5, 2, 1, True
        key = s.save_custom_preset("  低沉旁白 ")
        self.assertEqual(s.preset, key)
        self.assertFalse(s.custom)
        self.assertEqual(s.custom_presets[key].name, "低沉旁白")
        self.assertEqual((s.custom_presets[key].pitch, s.custom_presets[key].formant), (5, 2))
        s.select_preset("original")
        self.assertEqual((s.pitch, s.formant, s.reverb), (0, 0, 0))
        s.select_preset(key)
        self.assertEqual((s.pitch, s.formant, s.reverb), (5, 2, 1))
        self.assertEqual(s.input_gain, 2)
        s.pitch, s.custom = -3, True
        s.reset_effect()
        self.assertEqual((s.pitch, s.custom), (5, False))
        s.rename_custom_preset(key, "Narrator")
        self.assertEqual(s.custom_presets[key].name, "Narrator")
        s.rename_custom_preset(key, "narrator")
        s.start_recording()
        s.delete_custom_preset(key)
        self.assertEqual(s.preset, "original")
        self.assertTrue(s.custom)
        self.assertEqual(s.pitch, 5)
        self.assertTrue(s.recording)
        with self.assertRaises(ValueError):
            s.select_preset(key)
        with self.assertRaises(ValueError):
            s.delete_custom_preset("m2f")

    def test_custom_preset_name_rules_and_limit(self):
        s = self.state
        for name, error in (("   ", "err_name_empty"), ("x" * (MAX_PRESET_NAME + 1), "err_name_too_long"),
                            ("ROBOT", "err_name_duplicate")):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as caught:
                    s.save_custom_preset(name, reserved={"Robot"})
                self.assertEqual(caught.exception.args[0], error)
        first = s.save_custom_preset("A")
        second = s.save_custom_preset("B")
        with self.assertRaises(ValueError):
            s.save_custom_preset("a")
        with self.assertRaises(ValueError):
            s.rename_custom_preset(second, "A")
        self.assertEqual(s.custom_presets[second].name, "B")
        s.delete_custom_preset(first)
        for index in range(MAX_CUSTOM_PRESETS - 1):
            s.save_custom_preset(f"P{index}")
        with self.assertRaises(ValueError) as caught:
            s.save_custom_preset("one more")
        self.assertEqual(caught.exception.args[0], "err_custom_limit")


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.config = TempConfig()
        self.config.start()

    def tearDown(self):
        self.config.stop()

    def test_every_text_has_both_languages(self):
        for key, values in TEXT.items():
            self.assertEqual(len(values), len(LANGUAGES), key)
            self.assertTrue(all(values), key)

    def test_missing_valid_invalid_and_round_trip(self):
        self.assertEqual(load_language(), ("zh", "", ""))
        self.assertTrue(str(settings_path()).startswith(self.config.directory.name))
        save_language("en")
        self.assertEqual(json.loads(settings_path().read_text()), {"language": "en"})
        self.assertEqual(load_language(), ("en", "", ""))
        self.assertFalse(settings_path().with_name("prototype-ui.json.tmp").exists())
        for content in ("{not json", '["en"]', '{"language": "fr"}'):
            with self.subTest(content=content):
                settings_path().write_text(content)
                self.assertEqual(load_language()[:2], ("zh", "settings_invalid"))

    def test_relative_xdg_falls_back_to_home(self):
        os.environ["XDG_CONFIG_HOME"] = "relative/path"
        self.assertEqual(settings_path(), Path.home() / ".config" / "voice-changer" / "prototype-ui.json")


class WidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setFont(QFont("Noto Sans CJK SC", 10))

    def setUp(self):
        self.config = TempConfig()
        self.config.start()
        self.window = self.new_window()

    def new_window(self):
        window = PrototypeWindow()
        window.timer.stop()
        window.show()
        self.app.processEvents()
        return window

    def tearDown(self):
        self.window.state.take = False
        self.window.state.recording = False
        self.window.close()
        self.config.stop()

    def select_devices(self):
        self.window.input.setCurrentIndex(1)
        self.window.output.setCurrentIndex(1)

    def answer_dialog(self, prefix):
        def answer():
            dialog = self.app.activeModalWidget()
            self.assertIsInstance(dialog, QMessageBox)
            button = next(b for b in dialog.buttons() if b.text().startswith(prefix))
            button.click()
        QTimer.singleShot(0, answer)

    def assert_fits(self, w):
        scroll = w.findChild(QScrollArea)
        self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
        stop_bounds = w.stop.rect().translated(w.stop.mapTo(w, QPoint(0, 0)))
        self.assertTrue(w.rect().contains(stop_bounds))
        self.assertTrue(w.stop.isVisible())
        for button in w.findChildren(QPushButton):
            if not button.isVisible():
                continue
            self.assertGreaterEqual(button.height(), 32, button.text())
            self.assertGreaterEqual(
                button.width(), button.fontMetrics().horizontalAdvance(button.text()) + 16,
                button.text(),
            )
        return scroll

    def test_empty_devices_and_visible_simulation(self):
        w = self.window
        self.assertFalse(w.monitor.isEnabled())
        self.assertFalse(w.record.isEnabled())
        self.assertIsNone(w.input_meter.peak)
        self.assertIn("未采集", w.input_meter.display_text)
        self.assertIn("未采集", w.effect_meter.display_text)
        w.input.setCurrentIndex(1)
        self.assertEqual(w.state.input_device, "usb_mic")
        self.assertTrue(w.record.isEnabled())
        self.assertFalse(w.monitor.isEnabled())
        QTest.mouseClick(w.record, Qt.MouseButton.LeftButton)
        self.assertTrue(w.state.recording)
        self.assertFalse(w.input.isEnabled())
        self.assertFalse(w.play_effect.isEnabled())
        self.assertIsNone(w.input_meter.peak)
        self.assertIsNone(w.effect_meter.peak)
        self.assertIn("模拟状态", w.activity.text())

    def test_parameter_sync_and_gain_preservation(self):
        w = self.window
        w.parameters["input_gain"].setValue(2)
        self.assertFalse(w.state.custom)
        QTest.mouseClick(w.presets["m2f"], Qt.MouseButton.LeftButton)
        self.assertEqual(w.state.input_gain, 2)
        w.pitch.setValue(5)
        self.assertIn("自定义（基于男变女）", w.current.text())
        w.reset_effect()
        self.assertEqual(w.pitch.value(), 0)
        self.assertEqual(w.parameters["input_gain"].value(), 2)
        self.assertEqual(w.current.text(), "男变女")

    def test_all_seven_simulated_sliders_have_editors_and_survive_language_rebuild(self):
        w = self.window
        w.presets["custom"].click()
        self.assertEqual(len(w.parameters), 6)
        pairs = [(w.pitch, w.pitch_editor), *(
            (slider, w.parameter_editors[key]) for key, slider in w.parameters.items()
        )]
        for slider, editor in pairs:
            slider.setValue(slider.maximum())
            self.assertEqual(editor.value(), slider.maximum())
            editor.setValue(editor.minimum())
            self.assertEqual(slider.value(), slider.minimum())
        w.parameter_editors["robot"].setValue(2)
        w.parameter_editors["echo"].setValue(1)
        w.parameter_editors["input_gain"].setValue(2)
        w.toggle_language()
        self.assertEqual(w.parameters["robot"].value(), 2)
        self.assertEqual(w.parameter_editors["echo"].value(), 1)
        self.assertEqual(w.parameters["input_gain"].value(), 2)
        w.reset_effect()
        self.assertEqual(w.parameters["robot"].value(), 0)
        self.assertEqual(w.parameter_editors["echo"].value(), 0)
        self.assertEqual(w.parameters["input_gain"].value(), 2)
        self.assertTrue(w.tuning_panel.isHidden())

    def test_unrelated_output_disconnect_preserves_recording(self):
        w = self.window
        self.select_devices()
        w.toggle_record()
        w.scenario.setCurrentIndex(FAULTS.index("output_lost"))
        w.inject_fault()
        self.assertTrue(w.state.recording)
        self.assertFalse(w.state.fault)
        self.assertEqual(w.output.currentIndex(), 0)
        w.scenario.setCurrentIndex(FAULTS.index("input_lost"))
        w.inject_fault()
        self.assertFalse(w.state.active)
        self.assertTrue(w.state.incomplete)
        self.assertIn("模拟故障：输入断开", w.notice.text())
        w.clear_fault()
        self.assertFalse(w.state.active)

    def test_input_disconnect_during_playback_does_not_stop(self):
        w = self.window
        self.select_devices()
        w.toggle_record()
        w.toggle_record()
        w.play("effect")
        w.scenario.setCurrentIndex(FAULTS.index("input_lost"))
        w.inject_fault()
        self.assertEqual(w.state.playing, "effect")
        self.assertFalse(w.state.fault)

    def test_unsaved_cancel_save_cancel_confirm_and_discard(self):
        w = self.window
        self.select_devices()
        w.toggle_record()
        w.tick()
        w.toggle_record()
        self.answer_dialog("取消")
        w.toggle_record()
        self.assertFalse(w.state.recording)
        self.assertTrue(w.state.unsaved)
        self.answer_dialog("取消")
        self.assertFalse(w.save_demo())
        self.assertTrue(w.state.unsaved)
        self.answer_dialog("模拟保存失败")
        self.assertFalse(w.save_demo())
        self.assertTrue(w.state.unsaved)
        self.assertIn("模拟保存失败", w.notice.text())
        self.answer_dialog("标记流程")
        self.assertTrue(w.save_demo())
        self.assertIn("未写文件", w.take_status.text())
        self.assertFalse(w.state.unsaved)
        w.toggle_record()
        w.toggle_record()
        self.answer_dialog("丢弃")
        w.toggle_record()
        self.assertTrue(w.state.recording)

    def test_close_cancel_stops_without_auto_resume(self):
        w = self.window
        self.select_devices()
        w.toggle_monitor()
        w.toggle_record()
        self.answer_dialog("取消")
        self.assertFalse(w.close())
        self.assertTrue(w.isVisible())
        self.assertFalse(w.state.active)
        self.assertTrue(w.state.unsaved)

    def test_keyboard_slider_and_focus(self):
        w = self.window
        w.presets["custom"].click()
        w.pitch.setFocus()
        QTest.keyClick(w.pitch, Qt.Key.Key_Right)
        self.assertEqual(w.state.pitch, 1)
        self.assertTrue(w.state.custom)
        QTest.keyClick(w.pitch, Qt.Key.Key_Tab)
        self.assertIsNotNone(self.app.focusWidget())

    def test_simulated_startup_discovery_and_manual_refresh_remain_silent(self):
        w = self.window
        self.app.processEvents()
        self.assertFalse(w._startup_refresh_due)
        self.assertFalse(w._refresh_pending)
        self.assertIn("模拟列表已就绪", w.discovery_status.text())
        self.assertEqual((w.input.currentIndex(), w.output.currentIndex()), (0, 0))
        w.refresh_devices()
        self.assertTrue(w._refresh_pending)
        self.assertFalse(w.refresh.isEnabled())
        self.assertIn("正在查找模拟", w.discovery_status.text())
        w.refresh_devices()
        self.app.processEvents()
        self.assertFalse(w._refresh_pending)
        self.assertTrue(w.refresh.isEnabled())
        self.assertFalse(w.state.active)
        self.assertEqual((w.input.currentIndex(), w.output.currentIndex()), (0, 0))
        self.assertIsNone(w.input_meter.peak)
        self.assertIn("未采集", w.input_meter.display_text)

    def test_window_routed_wheel_does_not_focus_unpressed_parameters(self):
        w = self.window
        for language in ("zh", "en"):
            w.language = language
            w.build_ui()
            w.presets["custom"].click()
            w.resize(480, 400)
            w.activateWindow()
            self.app.processEvents()
            scroll = w.scroll_area.verticalScrollBar()
            for control in (
                w.pitch, *w.parameters.values(), w.pitch_editor, *w.parameter_editors.values(),
            ):
                with self.subTest(language=language, control=control.accessibleName()):
                    control.setValue(control.minimum())
                    def value():
                        return control.value()

                    def wheel(delta=120):
                        QTest.wheelEvent(
                            w.windowHandle(),
                            QPointF(control.mapTo(w, control.rect().center())), QPoint(0, delta),
                        )
                        self.app.processEvents()

                    w.scroll_area.ensureWidgetVisible(control)
                    w.scenarios_toggle.setFocus(Qt.FocusReason.OtherFocusReason)
                    self.app.processEvents()
                    old_value, old_scroll = value(), scroll.value()
                    wheel()
                    self.assertEqual(value(), old_value)
                    self.assertLess(scroll.value(), old_scroll)
                    w.scroll_area.ensureWidgetVisible(control)
                    control.setFocus(Qt.FocusReason.OtherFocusReason)
                    QTest.keyClick(control, Qt.Key.Key_Backtab)
                    QTest.keyClick(self.app.focusWidget(), Qt.Key.Key_Tab)
                    self.app.processEvents()
                    self.assertTrue(control.hasFocus())
                    old_value, old_scroll = value(), scroll.value()
                    wheel()
                    self.assertGreater(value(), old_value)
                    self.assertEqual(scroll.value(), old_scroll)
                    for boundary, delta in ((control.maximum(), 120), (control.minimum(), -120)):
                        control.setValue(boundary)
                        old_scroll = scroll.value()
                        wheel(delta)
                        self.assertEqual(value(), boundary)
                        self.assertEqual(scroll.value(), old_scroll)

    def test_parameter_wheel_focus_and_stable_fixed_actions_in_both_languages(self):
        w = self.window
        for language in ("zh", "en"):
            w.language = language
            w.build_ui()
            w.presets["custom"].click()
            w.resize(480, 400)
            w.activateWindow()
            self.app.processEvents()
            scroll = w.scroll_area.verticalScrollBar()
            for control in (w.pitch, w.parameters["formant"]):
                with self.subTest(language=language, control=control.accessibleName()):
                    def wheel():
                        point = control.rect().center()
                        event = QWheelEvent(
                            QPointF(point), QPointF(control.mapToGlobal(point)),
                            QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False,
                        )
                        QApplication.sendEvent(control, event)
                        self.app.processEvents()
                        self.assertTrue(event.isAccepted())
                    def value():
                        return control.value()
                    w.scroll_area.ensureWidgetVisible(control)
                    self.app.processEvents()
                    old = value()
                    before = scroll.value()
                    wheel()
                    self.assertEqual(value(), old)
                    self.assertLess(scroll.value(), before)
                    w.scroll_area.ensureWidgetVisible(control)
                    self.app.processEvents()
                    control.setValue(control.minimum())
                    QTest.mouseClick(control, Qt.MouseButton.LeftButton)
                    control.setValue(control.minimum())
                    self.app.processEvents()
                    old = value()
                    before = scroll.value()
                    wheel()
                    self.assertGreater(value(), old)
                    self.assertEqual(scroll.value(), before)
                    control.clearFocus()
                    old = value()
                    wheel()
                    self.assertEqual(value(), old)
                    control.setFocus(Qt.FocusReason.TabFocusReason)
                    self.app.processEvents()
                    bounds = control.rect().translated(control.mapTo(w.scroll_area.viewport(), QPoint()))
                    self.assertTrue(w.scroll_area.viewport().rect().contains(bounds))
            for button in (w.monitor, w.record, w.stop):
                self.assertTrue(w.rect().contains(button.rect().translated(button.mapTo(w, QPoint()))))
            w.scroll_area.ensureWidgetVisible(w.presets["original"])
            self.app.processEvents()
            before = scroll.value()
            QTest.mouseClick(w.presets["original"], Qt.MouseButton.LeftButton)
            self.app.processEvents()
            self.assertLessEqual(scroll.value(), before)
            self.assertTrue(w.tuning_panel.isHidden())

    def test_p0_prototype_has_no_custom_preset_library(self):
        w = self.window
        self.assertFalse(hasattr(w, "save_preset_button"))
        self.assertFalse(hasattr(w, "manage_preset_button"))
        self.assertEqual(set(w.presets), {*BUILTIN_PRESETS, "custom"})
        self.assertEqual(w.state.custom_presets, {})
        w.choose_preset("robot")
        w.pitch.setValue(-4)
        self.assertEqual(w.current.text(), "自定义（基于机器人）")
        self.assertEqual(w.pitch.value(), -4)
        self.assertEqual(list(Path(self.config.directory.name).rglob("*")), [])

    def test_custom_only_configuration_and_language_rebuild_keep_simulated_values(self):
        w = self.window
        self.assertTrue(w.tuning_panel.isHidden())
        self.assertFalse(hasattr(w, "tuning_toggle"))
        self.assertEqual([editor.text() for editor in w.config_inputs],
                         ["48000", "1024", "1", "0.05"])
        w.choose_preset("m2f")
        w.presets["custom"].click()
        self.assertFalse(w.tuning_panel.isHidden())
        self.assertTrue(w.presets["custom"].isChecked())
        self.assertEqual(w.state.preset, "m2f")
        w.config_inputs[1].setText("512")
        w.toggle_language()
        self.assertFalse(w.tuning_panel.isHidden())
        self.assertEqual(w.config_inputs[1].text(), "512")
        w.presets["original"].click()
        self.assertTrue(w.tuning_panel.isHidden())
        w.presets["custom"].click()
        self.assertEqual(w.config_inputs[1].text(), "512")
        self.assertFalse(w.state.active)
        self.assertEqual(
            {path.name for path in Path(self.config.directory.name).rglob("*") if path.is_file()},
            {"prototype-ui.json"},
        )

    def test_language_switch_keeps_state_and_persists(self):
        w = self.window
        self.select_devices()
        w.toggle_scenarios()
        w.scenario.setCurrentIndex(FAULTS.index("busy"))
        w.choose_preset("m2f")
        w.pitch.setValue(3)
        w.toggle_monitor()
        QTest.mouseClick(w.language_button, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertEqual(w.language, "en")
        self.assertIn("Voice Changer", w.windowTitle())
        self.assertEqual(w.language_button.text(), "中文")
        self.assertTrue(w.state.monitoring)
        self.assertEqual(w.monitor.text(), "Stop (sim)")
        self.assertEqual(w.monitor.accessibleName(), "Stop monitoring (simulated)")
        self.assertEqual((w.input.currentIndex(), w.output.currentIndex()), (1, 1))
        self.assertEqual(w.input.currentText(), "Sim · USB microphone")
        self.assertEqual(w.pitch.value(), 3)
        self.assertEqual(w.current.text(), "Custom (based on Male → Female)")
        self.assertTrue(w.scenarios_panel.isVisible())
        self.assertEqual(w.scenario.currentText(), "Device busy / DSP failure")
        self.assertIn("capture sim", w.activity.text())
        w.inject_fault()
        self.assertIn("Simulated fault: Device busy", w.notice.text())
        self.assertEqual(load_language(), ("en", "", ""))
        second = self.new_window()
        self.assertEqual(second.language, "en")
        self.assertEqual(second.monitor.text(), "Start (sim)")
        second.close()
        w.toggle_language()
        self.assertEqual(w.language, "zh")
        self.assertIn("模拟故障：设备忙", w.notice.text())
        self.assertEqual(load_language(), ("zh", "", ""))

    def test_language_setting_problems_are_visible(self):
        path = settings_path()
        path.parent.mkdir(parents=True)
        path.write_text('{"language": "klingon"}')
        w = self.new_window()
        self.assertEqual(w.language, "zh")
        self.assertIn("语言设置无效", w.notice.text())
        self.assertEqual(w.notice.objectName(), "warning")
        w.toggle_language()
        self.assertNotIn("Invalid", w.notice.text())
        w.close()
        path.unlink()
        path.mkdir()  # a directory where the file should be makes the write fail
        w = self.new_window()
        self.assertIn("无法读取语言设置", w.notice.text())
        w.toggle_language()
        self.assertEqual(w.language, "en")
        self.assertIn("could not be saved", w.notice.text())
        self.assertEqual(w.notice.objectName(), "warning")
        w.close()

    def test_layout_at_desktop_sizes_in_both_languages(self):
        w = self.window
        for language in LANGUAGES:
            if w.language != language:
                w.toggle_language()
            for width, height in ((480, 400), (640, 440), (720, 540), (1120, 850)):
                with self.subTest(language=language, size=(width, height)):
                    w.resize(width, height)
                    self.app.processEvents()
                    self.assert_fits(w)

    def test_silent_waves_and_fault_remain_readable_in_both_languages(self):
        w = self.window
        w.input.setCurrentIndex(1)
        w.record.click()
        self.assertTrue(w.state.recording)
        w.scenario.setCurrentIndex(0)
        w.inject_fault()
        self.assertTrue(w.state.fault)
        for language in LANGUAGES:
            if w.language != language:
                w.toggle_language()
            for maximized, width, height in (
                (False, 480, 400), (False, 640, 440), (True, 0, 0),
            ):
                with self.subTest(language=language, size="maximized" if maximized else (width, height)):
                    if maximized:
                        w.showMaximized()
                    else:
                        w.showNormal()
                        w.resize(width, height)
                    QTest.qWait(30)
                    if maximized:
                        self.assertTrue(w.isMaximized())
                    else:
                        self.assertEqual((w.width(), w.height()), (width, height))
                    self.assertIn("原型" if language == "zh" else "prototype", w.windowTitle())
                    meters = (w.input_meter, w.effect_meter)
                    compact = not maximized and (
                        w.width() < 800 or w.height() < 600
                    )
                    if compact:
                        self.assertTrue(w._small_layout)
                        self.assertTrue(w.analysis_toggle.isVisible())
                        self.assertTrue(w.device_toggle.isVisible())
                        self.assertTrue(w.preset_toggle.isVisible())
                        self.assertTrue(w.recording_toggle.isVisible())
                        self.assertFalse(w.device_selectors.isVisible())
                        self.assertFalse(w.preset_options.isVisible())
                        self.assertFalse(w.recording_content.isVisible())
                        self.assertTrue(all(not meter.analysis_canvas.isVisible() for meter in meters))
                        for toggle, content in (
                            (w.device_toggle, w.device_selectors),
                            (w.preset_toggle, w.preset_options),
                            (w.recording_toggle, w.recording_content),
                        ):
                            toggle.click()
                            self.assertTrue(content.isVisible())
                            toggle.click()
                            self.assertFalse(content.isVisible())
                    else:
                        self.assertFalse(w._small_layout)
                        self.assertFalse(w.analysis_toggle.isVisible())
                        self.assertFalse(w.device_toggle.isVisible())
                        self.assertFalse(w.preset_toggle.isVisible())
                        self.assertFalse(w.recording_toggle.isVisible())
                        self.assertTrue(w.device_selectors.isVisible())
                        self.assertTrue(w.preset_options.isVisible())
                        self.assertTrue(w.recording_content.isVisible())
                        self.assertTrue(all(meter.analysis_canvas.isVisible() for meter in meters))
                    for meter in meters:
                        ensure_visible_after_layout(w.scroll_area, meter)
                        bounds = meter.rect().translated(
                            meter.mapTo(w.scroll_area.viewport(), QPoint(0, 0)),
                        )
                        self.assertTrue(
                            w.scroll_area.viewport().rect().contains(bounds),
                            f"{bounds}, viewport={w.scroll_area.viewport().rect()}, "
                            f"scroll={w.scroll_area.verticalScrollBar().value()}",
                        )
                        self.assertIsNone(meter.peak)
                        self.assertEqual(meter.history, ())
                        self.assertFalse(meter._timer.isActive())
                        self.assertIn("未采集" if language == "zh" else "No capture", meter.display_text)
                        for label in (meter.name_label, meter.reading_label):
                            self.assertGreaterEqual(
                                label.height(), label.heightForWidth(label.width()), label.text(),
                            )
                    self.assertTrue(w.notice.isVisible())
                    self.assertGreaterEqual(w.notice.height(), w.notice.heightForWidth(w.notice.width()))
                    option = QStyleOption()
                    option.initFrom(w.notice)
                    self.assertEqual(
                        w.notice.contentsRect(),
                        w.notice.style().subElementRect(
                            QStyle.SubElement.SE_FrameContents, option, w.notice,
                        ),
                    )
                    self.assertGreater(w.notice.contentsRect().left(), 3)
                    notice_bounds = w.notice.rect().translated(w.notice.mapTo(w, QPoint(0, 0)))
                    stop_bounds = w.stop.rect().translated(w.stop.mapTo(w, QPoint(0, 0)))
                    self.assertTrue(w.rect().contains(notice_bounds))
                    self.assertTrue(w.rect().contains(stop_bounds))
                    self.assertFalse(notice_bounds.intersects(stop_bounds))
                    self.assertTrue(w.stop.isVisible())
                    self.assertEqual(w.scroll_area.horizontalScrollBar().maximum(), 0)

    def test_compact_layout_keeps_both_waves_visible_and_details_reachable(self):
        w = self.window
        for maximized, width, height in (
            (False, 480, 400), (False, 640, 440), (False, 1280, 731), (True, 0, 0),
        ):
            if maximized:
                w.showMaximized()
            else:
                w.showNormal()
                w.resize(width, height)
            QTest.qWait(30)
            with self.subTest(size="maximized" if maximized else (width, height)):
                compact = not maximized and (w.width() < 800 or w.height() < 600)
                self.assertEqual(w._small_layout, compact)
                meters = (w.input_meter, w.effect_meter)
                if compact:
                    self.assertTrue(all(not meter.analysis_canvas.isVisible() for meter in meters))
                    for meter in meters:
                        bounds = meter.rect().translated(
                            meter.mapTo(w.scroll_area.viewport(), QPoint(0, 0)),
                        )
                        self.assertTrue(
                            w.scroll_area.viewport().rect().contains(bounds), str(bounds),
                        )
                    for toggle, content in (
                        (w.analysis_toggle, w.input_meter.analysis_canvas),
                        (w.device_toggle, w.device_selectors),
                        (w.preset_toggle, w.preset_options),
                        (w.recording_toggle, w.recording_content),
                    ):
                        self.assertTrue(toggle.isVisible())
                        toggle.click()
                        self.assertTrue(content.isVisible())
                        toggle.click()
                        self.assertFalse(content.isVisible())
                else:
                    self.assertTrue(all(meter.analysis_canvas.isVisible() for meter in meters))
                    self.assertTrue(w.device_selectors.isVisible())
                    self.assertTrue(w.preset_options.isVisible())
                    self.assertTrue(w.recording_content.isVisible())
                    for toggle in (
                        w.analysis_toggle, w.device_toggle,
                        w.preset_toggle, w.recording_toggle,
                    ):
                        self.assertFalse(toggle.isVisible())
                self.assertTrue(w.stop.isVisible())
                self.assertEqual(w.scroll_area.horizontalScrollBar().maximum(), 0)

    def test_both_comparison_layouts_remain_silent_and_readable(self):
        w = self.window
        for language in LANGUAGES:
            if w.language != language:
                w.toggle_language()
            for maximized, width, height in (
                (False, 480, 400), (False, 640, 440), (True, 0, 0),
            ):
                if maximized:
                    w.showMaximized()
                else:
                    w.showNormal()
                    w.resize(width, height)
                for side in (False, True):
                    with self.subTest(language=language, size=(width, height), side=side):
                        w.wave_layout_button.setChecked(side)
                        QTest.qWait(30)
                        self.assertEqual(w.wave_comparison.side_by_side, side)
                        self.assertEqual(w.wave_layout_button.isChecked(), side)
                        self.assertTrue(w.wave_layout_button.accessibleName())
                        for meter in (w.input_meter, w.effect_meter):
                            ensure_visible_after_layout(w.scroll_area, meter)
                            bounds = meter.rect().translated(
                                meter.mapTo(w.scroll_area.viewport(), QPoint(0, 0)),
                            )
                            self.assertTrue(w.scroll_area.viewport().rect().contains(bounds), str(bounds))
                            self.assertIsNone(meter.peak)
                            self.assertEqual(meter.history, ())
                            self.assertEqual(meter.pitch_history, ())
                            self.assertIsNone(meter.analysis_canvas.reading)
                            self.assertFalse(meter._timer.isActive())
                            self.assertIn(
                                "未采集" if language == "zh" else "No capture",
                                meter.analysis_status.text(),
                            )
                            for item in (meter.name_label, meter.reading_label, meter.analysis_status):
                                self.assertGreaterEqual(
                                    item.height(), item.heightForWidth(item.width()), item.text(),
                                )
                        self.assertTrue(w.stop.isVisible())
                        self.assertTrue(w.rect().contains(
                            w.stop.rect().translated(w.stop.mapTo(w, QPoint())),
                        ))
                        self.assertEqual(w.scroll_area.horizontalScrollBar().maximum(), 0)
            previous = w.wave_comparison.side_by_side
            w.toggle_language()
            self.assertEqual(w.wave_comparison.side_by_side, previous)

    def test_default_window_has_no_horizontal_scroll(self):
        w = self.window
        w.resize(640, 440)
        self.app.processEvents()
        self.assertEqual(w.findChild(QScrollArea).horizontalScrollBar().maximum(), 0)
        self.assertTrue(w.stop.isVisible())

    def test_expanded_scenarios_keep_stop_visible_at_minimum_size(self):
        w = self.window
        for language in LANGUAGES:
            with self.subTest(language=language):
                if w.language != language:
                    w.toggle_language()
                w.resize(480, 400)
                w.choose_preset("echo")
                w.pitch.setValue(2)
                if not w.scenarios_visible:
                    w.toggle_scenarios()
                self.app.processEvents()
                self.assertTrue(w.stop.isVisible())
                self.assertTrue(w.rect().contains(
                    w.stop.rect().translated(w.stop.mapTo(w, QPoint(0, 0))),
                ))
                self.assertEqual(w.findChild(QScrollArea).horizontalScrollBar().maximum(), 0)


if __name__ == "__main__":
    unittest.main()
