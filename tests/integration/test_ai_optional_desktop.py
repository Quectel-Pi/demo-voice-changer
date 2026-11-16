"""The existing desktop remains usable when optional AI packages are unavailable."""

import unittest

from ai_optional_test_support import run_without_ai

SMOKE_TEST = r"""
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from device_fakes import FakeDeviceLibrary
from qt_wait import wait_until

library = FakeDeviceLibrary()
sys.modules["sounddevice"] = library
from voice_changer.audio.engine import EngineState
from voice_changer.ui.main_window import MainWindow

app = QApplication([])
window = MainWindow()
try:
    window.show()
    if not wait_until(lambda: window.controller.is_ready and not window.controller.busy
                      and not window._startup_refresh_due, 5):
        raise RuntimeError("desktop did not finish starting")
    if window.controller.state != EngineState.STOPPED or window.controller.owns_devices:
        raise AssertionError("desktop started audio without a user action")
    if window.preset_combo.currentData() != "original":
        raise AssertionError("desktop did not start with the original DSP preset")
    if [name for name, _ in library.calls] != ["devices", "hostapis"]:
        raise AssertionError("unexpected device discovery")
    window.close_button.setFocus()
    QTest.keyClick(window.close_button, Qt.Key.Key_Space)
    if not wait_until(lambda: window.controller.is_closed, 5):
        raise RuntimeError("desktop did not close")
finally:
    window.close()
    if not wait_until(lambda: window.controller.is_closed, 5):
        raise RuntimeError("desktop cleanup did not finish")

assert_ai_unloaded()
print("AI-optional offscreen desktop passed")
"""


class AIOptionalDesktopTests(unittest.TestCase):
    def test_existing_desktop_starts_and_closes_without_ai_dependencies(self):
        result = run_without_ai(SMOKE_TEST)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("AI-optional offscreen desktop passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
