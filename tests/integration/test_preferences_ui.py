"""Preferences use temporary XDG paths and test-only device discovery."""

import os
from pathlib import Path
import sys
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QCoreApplication, QEvent, QTimer
from PySide6.QtWidgets import QApplication

from voice_changer.app.engine_controller import EngineController
from voice_changer.app.preferences import AudioSettings, DeviceIdentity, Preferences, PreferencesStore
from voice_changer.audio.engine import EngineState
from voice_changer.ui.main_window import MainWindow
from device_fakes import FakeDeviceLibrary
from qt_wait import wait_until


class PreferencesUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "preferences.json"
        self.library = FakeDeviceLibrary()
        self.patcher = patch.dict(sys.modules, {"sounddevice": self.library})
        self.patcher.start()
        self.window = None
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.library.release.set()
        if self.window is not None:
            self.window.close()
            self.wait(lambda: self.window.controller.is_closed)
            self.window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.patcher.stop()
        self.temp.cleanup()

    def wait(self, condition):
        self.assertTrue(wait_until(condition, 4))

    def seed(self):
        preferences = Preferences(
            DeviceIdentity("test input", "test API"), DeviceIdentity("test output", "test API"),
            AudioSettings(48000, 512, 1, 0.05), str(self.root),
        )
        PreferencesStore(self.path).save(preferences)
        return preferences

    def open(self):
        self.window = MainWindow(EngineController(preferences_file=self.path))
        self.window.show()
        self.wait(lambda: self.window.controller.is_ready and not self.window.controller.busy
                  and not self.window._startup_refresh_due)

    def test_saved_format_outside_dsp_demo_is_not_prefilled(self):
        from dataclasses import replace
        original = self.seed()
        PreferencesStore(self.path).save(
            replace(original, audio=AudioSettings(24000, 256, 1, 0.05))
        )
        self.open()
        self.assertEqual(
            [editor.text() for editor in self.window.config_inputs],
            ["48000", "1024", "1", "0.05"],
        )
        self.assertIn("未预填", self.window.notice.text())
        self.assertIsNone(self.window._previewed_config)

    def test_startup_refresh_restores_history_without_automatic_activity(self):
        self.seed()
        old = self.path.read_bytes()
        self.open()
        self.assertEqual([name for name, _ in self.library.calls], ["devices", "hostapis"])
        self.assertEqual([e.text() for e in self.window.config_inputs], ["48000", "512", "1", "0.05"])
        self.assertEqual(self.window.controller.device_selection.input_device.name, "test input")
        self.assertIsNone(self.window._previewed_config)
        self.assertIn("默认原声", self.window.notice.text())
        self.assertEqual(self.window.preset_combo.currentData(), "original")
        self.assertEqual(self.window.controller.state, EngineState.STOPPED)
        self.assertFalse(self.window.controller.owns_devices)
        self.assertEqual(self.window.input_gain_input.value(), 0)
        self.assertEqual(self.window.output_gain_input.value(), 0)
        self.assertEqual(self.window.controller.device_selection.input_device.name, "test input")
        self.assertEqual(self.window.controller.device_selection.output_device.name, "test output")
        self.assertEqual(self.path.read_bytes(), old)
        self.assertTrue(all(ident != threading.get_ident() for _, ident in self.library.calls))

    def test_invalid_configuration_reports_error_preserves_file_and_safe_defaults(self):
        self.path.write_text('{"version": true}')
        self.open()
        self.assertIn("preferences", self.window.error_notice.text())
        self.assertEqual([e.text() for e in self.window.config_inputs],
                         ["48000", "1024", "1", "0.05"])
        self.assertEqual([name for name, _ in self.library.calls], ["devices", "hostapis"])
        self.assertEqual(self.path.read_text(), '{"version": true}')

    def test_missing_saved_device_does_not_fall_back_or_retry_restore_on_every_refresh(self):
        self.seed()
        self.library.devices[0]["name"] = "replacement input"
        self.open()
        self.assertIsNone(self.window.controller.device_selection.input_device)
        self.assertIn("reselect", self.window.error_notice.text())
        self.library.devices[0]["name"] = "test input"
        self.window.refresh_button.click()
        self.wait(lambda: not self.window.controller.busy)
        self.assertIsNone(self.window.controller.device_selection.input_device)

    def test_ambiguous_history_requires_selection_after_automatic_discovery(self):
        self.seed()
        self.library.devices.append(dict(self.library.devices[0]))
        self.library.hostapis[0]["devices"].append(2)
        self.open()
        self.assertIsNone(self.window.controller.device_selection.input_device)
        self.assertEqual(self.window.controller.device_selection.output_device.name, "test output")
        self.assertIn("reselect", self.window.error_notice.text())
        self.assertIn("2 input(s)", self.window.device_status.text())
        self.assertFalse(self.window.controller.owns_devices)
        self.assertFalse(self.window.listen_button.isEnabled())
        self.assertFalse(self.window.record_button.isEnabled())

    def test_settings_file_read_runs_off_gui_and_close_remains_responsive(self):
        self.seed()
        entered, release = threading.Event(), threading.Event()
        thread_ids = []
        original = PreferencesStore.load
        def gated(store):
            thread_ids.append(threading.get_ident())
            entered.set()
            if not release.wait(3):
                raise RuntimeError("Synthetic settings barrier timed out")
            return original(store)
        with patch.object(PreferencesStore, "load", gated):
            self.window = MainWindow(EngineController(preferences_file=self.path))
            self.window.show()
            self.wait(entered.is_set)
            heartbeat = []
            QTimer.singleShot(0, lambda: heartbeat.append(True))
            self.wait(lambda: bool(heartbeat))
            self.window.close()
            release.set()
            self.wait(lambda: self.window.controller.is_closed)
        self.assertNotEqual(thread_ids, [threading.get_ident()])

    def test_invalid_save_directory_is_reported_and_not_silently_restored(self):
        original = self.seed()
        from dataclasses import replace
        PreferencesStore(self.path).save(replace(original, save_directory=str(self.root / "missing")))
        self.open()
        self.assertIsNone(self.window.controller.preferences.save_directory)
        self.assertIn("choose again", self.window.error_notice.text())
        self.assertEqual(self.window.samplerate_input.text(), "48000")

    def test_desktop_wrapper_opens_real_offscreen_window_from_other_cwd_without_audio(self):
        script = self.root / "sitecustomize.py"
        script.write_text(
            "import sys\nfrom importlib.abc import MetaPathFinder\n"
            "class NoAudio(MetaPathFinder):\n"
            " def find_spec(self,fullname,path=None,target=None):\n"
            "  if fullname.split('.')[0] in ('numpy','sounddevice'):\n"
            "   raise AssertionError('Unexpected audio import: '+fullname)\n"
            "sys.meta_path.insert(0,NoAudio())\n"
            "from types import ModuleType\n"
            "devices=ModuleType('sounddevice')\n"
            "devices.query_devices=lambda: []\n"
            "devices.query_hostapis=lambda: []\n"
            "sys.modules['sounddevice']=devices\n"
            "from PySide6.QtCore import QTimer\n"
            "from voice_changer.ui import main_window\n"
            "class ClosingWindow(main_window.MainWindow):\n"
            " def showMaximized(self):\n"
            "  super().showMaximized()\n"
            "  assert self.isMaximized() and not self.isFullScreen()\n"
            "  print('REAL_WRAPPER_WINDOW_SHOWN_NO_AUDIO',flush=True)\n"
            "  self.controller.shutdownComplete.connect(lambda: print('WRAPPER_SHUTDOWN_COMPLETE',flush=True))\n"
            "  QTimer.singleShot(0,self.close)\n"
            "main_window.MainWindow=ClosingWindow\n",
        )
        env = dict(os.environ, PYTHONPATH=str(self.root), PYTHONDONTWRITEBYTECODE="1")
        for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
            env[name] = str(self.root)
        entry = Path(__file__).resolve().parents[2] / "scripts/launch_voice_changer.py"
        result = subprocess.run(
            [sys.executable, "-B", str(entry)], cwd=self.root,
            env=env, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("REAL_WRAPPER_WINDOW_SHOWN_NO_AUDIO", result.stdout)
        self.assertIn("WRAPPER_SHUTDOWN_COMPLETE", result.stdout)
        self.assertEqual([p.name for p in self.root.iterdir()], ["sitecustomize.py"])
