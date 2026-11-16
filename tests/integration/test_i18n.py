"""In-place UI localization preserves dynamic profile and model names."""

import os
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QLabel, QPlainTextEdit, QWidget

from voice_changer.ui.i18n import LanguageFilter


class LanguageFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_dynamic_choice_names_survive_repeated_language_switches(self):
        root = QWidget()
        combo = QComboBox(root)
        combo.addItem("我的音色 / Mine · 声场 / USB — 损坏 / damaged", "profile:1")
        combo.addItem("启动配置 / Configured · voice / test.onnx", "model:1")
        language = ["zh"]
        language_filter = LanguageFilter(root, lambda: language[0])

        language_filter.localize_tree()
        self.assertEqual(combo.itemText(0), "我的音色 · 声场 / USB — 损坏")
        self.assertEqual(combo.itemText(1), "启动配置 · voice / test.onnx")

        language[0] = "en"
        language_filter.localize_tree()
        self.assertEqual(combo.itemText(0), "My voice · 声场 / USB — damaged")
        self.assertEqual(combo.itemText(1), "Configured · voice / test.onnx")

        language[0] = "zh"
        language_filter.localize_tree()
        self.assertEqual(combo.itemText(0), "我的音色 · 声场 / USB — 损坏")
        self.assertEqual(combo.itemText(1), "启动配置 · voice / test.onnx")
        root.deleteLater()

    def test_device_placeholder_and_editor_hint_localize_without_touching_user_data(self):
        root = QWidget()
        combo = QComboBox(root)
        combo.setObjectName("inputDeviceCombo")
        combo.addItem("未选择 / Not selected", None)
        combo.addItem("声场 / USB", "device:1")
        editor = QPlainTextEdit(root)
        editor.setPlainText("我的文字 / My text")
        editor.setPlaceholderText("输入文字 / Enter text")
        language = ["zh"]
        language_filter = LanguageFilter(root, lambda: language[0])
        for code, placeholder, hint in (
            ("zh", "未选择", "输入文字"),
            ("en", "Not selected", "Enter text"),
            ("zh", "未选择", "输入文字"),
        ):
            language[0] = code
            language_filter.localize_tree()
            self.assertEqual(combo.itemText(0), placeholder)
            self.assertEqual(combo.itemText(1), "声场 / USB")
            self.assertEqual(editor.placeholderText(), hint)
            self.assertEqual(editor.toPlainText(), "我的文字 / My text")
        root.deleteLater()

    def test_parented_dialogs_are_localized_without_affecting_unrelated_windows(self):
        root = QWidget()
        dialog = QDialog(root)
        nested_dialog = QDialog(dialog)
        label = QLabel("授权确认 / Consent", nested_dialog)
        unrelated = QDialog()
        other_label = QLabel("不变 / Unchanged", unrelated)
        language = ["zh"]
        language_filter = LanguageFilter(root, lambda: language[0])
        language_filter.localize_tree()
        self.assertEqual(label.text(), "授权确认")
        language[0] = "en"
        language_filter.localize_tree()
        self.assertEqual(label.text(), "Consent")
        language_filter.localize_subtree(unrelated)
        self.assertEqual(other_label.text(), "不变 / Unchanged")
        root.deleteLater()
        unrelated.deleteLater()

    def test_dynamic_copy_is_localized_before_display_and_retains_source(self):
        root = QWidget()
        label = QLabel(root)
        language = ["zh"]
        language_filter = LanguageFilter(root, lambda: language[0])
        language_filter.set_text(label, "尚未应用。\nNot applied.")
        self.assertEqual(label.text(), "尚未应用。")
        language[0] = "en"
        language_filter.localize_tree()
        self.assertEqual(label.text(), "Not applied.")
        language_filter.set_text(label, "等待应用 / Pending")
        self.assertEqual(label.text(), "Pending")
        language[0] = "zh"
        language_filter.localize_tree()
        self.assertEqual(label.text(), "等待应用")
        root.deleteLater()


if __name__ == "__main__":
    unittest.main()
