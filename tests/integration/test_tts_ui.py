"""Full desktop read-aloud flow with a Kokoro subprocess substitute and synthetic output."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QMessageBox, QSizePolicy

from voice_changer.ai.kokoro_tts import VOICES, _FILES
from voice_changer.audio.engine import EngineState
from voice_changer.ui.main_window import MainWindow
from voice_changer.ui.tts_job import audio_info

from media_fakes import MediaLibrary
from qt_wait import wait_until


KOKORO_RUNTIME = '''
import signal
import time
import numpy as np


class Config:
    def __init__(self, **kwargs):
        pass

    def validate(self):
        return True


OfflineTtsKokoroModelConfig = Config
OfflineTtsModelConfig = Config
OfflineTtsConfig = Config


class _Audio:
    def __init__(self, samples, sample_rate):
        self.samples = samples
        self.sample_rate = sample_rate


class OfflineTts:
    def __init__(self, config):
        pass

    def generate(self, text, sid, speed):
        assert sid in (45, 47, 49)
        assert speed == 1.0
        if text == "FAIL":
            raise RuntimeError("kokoro deliberate failure")
        if text == "WAIT":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            print("WAITING", flush=True)
            time.sleep(60)
        frames = 240000 if text == "LONG" else 1200
        return _Audio(np.full(frames, 0.045, dtype=np.float32), 24000)
'''


class TtsUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        (self.runtime / "sherpa_onnx.py").write_text(KOKORO_RUNTIME, encoding="utf-8")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        scope = self.bin / "systemd-run"
        scope.write_text('#!/bin/sh\nshift 5\nexec "$@"\n', encoding="utf-8")
        scope.chmod(0o755)
        self.model = self.root / "kokoro"
        self.model.mkdir()
        for name in _FILES:
            (self.model / name).write_bytes(b"fixture")
        (self.model / "espeak-ng-data").mkdir()
        self.environment = patch.dict(os.environ, {
            "XDG_DATA_HOME": str(self.root / "data"),
            "XDG_CONFIG_HOME": str(self.root / "config"),
            "VOICE_CHANGER_KOKORO_MODEL": str(self.model),
            "VOICE_CHANGER_KOKORO_PYTHON": sys.executable,
            "PYTHONPATH": str(self.runtime),
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
        })
        self.environment.start()
        self.devices = MediaLibrary()
        self.sounddevice = patch.dict(sys.modules, {"sounddevice": self.devices})
        self.sounddevice.start()
        self.window = MainWindow()
        self.window.resize(640, 440)
        self.window.show()
        self.wait(lambda: self.window.controller.is_ready and not self.window.controller.busy
                  and not self.window._startup_refresh_due)
        self.window._select_mode("tts")
        self.page = self.window.tts_page
        self.assertEqual(self.page.text_input.toPlainText(), "Hello, 欢迎使用语音工作台")
        self.page.text_input.setPlainText("Hello")

    def tearDown(self):
        self.window.close()
        self.wait(lambda: self.window.controller.is_closed, timeout=6)
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.sounddevice.stop()
        self.environment.stop()
        self.temporary.cleanup()

    def wait(self, condition, timeout=5):
        self.assertTrue(wait_until(condition, timeout), "Timed out")

    def generate(self, text="Hello"):
        self.page.text_input.setPlainText(text)
        self.page.generate_button.click()
        self.assertTrue(self.page.job.busy)
        self.wait(lambda: not self.page.job.busy)

    def select_output(self):
        controller = self.window.controller
        selection = controller.device_selection
        output = next(info.ref for info in selection.devices if info.max_output_channels)
        selection.select("output", output)

    def test_voice_buttons_are_single_select_presets(self):
        buttons = self.page.voice_buttons
        self.assertEqual(
            set(buttons), {"kokoro:female", "kokoro:xiaoxiao", "kokoro:male"},
        )
        self.assertTrue(all(button.isEnabled() for button in buttons.values()))
        self.assertFalse(hasattr(self.page, "manage_button"))
        self.assertFalse(hasattr(self.page, "preset_button"))
        buttons["kokoro:male"].click()
        self.assertEqual(self.page.selected_voice, "kokoro:male")
        self.assertTrue(buttons["kokoro:male"].isChecked())
        self.assertFalse(buttons["kokoro:female"].isChecked())
        buttons["kokoro:xiaoxiao"].click()
        self.assertEqual(self.page.selected_voice, "kokoro:xiaoxiao")
        self.assertTrue(buttons["kokoro:xiaoxiao"].isChecked())
        self.assertFalse(buttons["kokoro:male"].isChecked())

    def test_kokoro_voice_ids_include_xiaoxiao(self):
        self.assertEqual(
            VOICES, {"female": 45, "xiaoxiao": 47, "male": 49},
        )
        for key in ("kokoro:female", "kokoro:xiaoxiao", "kokoro:male"):
            self.page.voice_buttons[key].click()
            self.assertEqual(self.page.selected_voice, key)
            self.generate()
            self.assertEqual(audio_info(self.page.job.result), (24000, 1200))

    def test_text_input_is_compact_fixed_height(self):
        editor = self.page.text_input
        self.assertEqual(
            editor.sizePolicy().verticalPolicy(), QSizePolicy.Policy.Fixed,
        )
        expected = editor.fontMetrics().lineSpacing() * 4 + 28
        self.assertEqual(editor.height(), expected)
        self.assertLessEqual(editor.height(), 140)

    def test_generation_produces_valid_audio_without_opening_devices(self):
        self.assertTrue(self.page.generate_button.isEnabled())
        self.generate()
        job = self.page.job
        self.assertEqual(audio_info(job.result), (24000, 1200))
        self.assertTrue(self.page.play_button.isEnabled())
        self.assertTrue(self.page.save_button.isEnabled())
        self.assertFalse(self.page.stop_button.isEnabled())
        self.assertEqual(self.devices.streams, [])
        self.assertIn("生成完成", self.page.status.text())
        with wave.open(str(job.result), "rb") as reader:
            self.assertEqual((reader.getframerate(), reader.getnchannels()), (24000, 1))
            self.assertEqual(reader.getnframes(), 1200)

    def test_failure_retains_previous_audio_and_surfaces_error(self):
        self.generate()
        previous = self.page.job.result
        self.generate("FAIL")
        self.assertEqual(self.page.job.result, previous)
        self.assertTrue(previous.is_file())
        self.assertIn("kokoro deliberate failure", self.page.status.text())
        self.assertTrue(self.page.play_button.isEnabled())

    def test_text_limit_blocks_generation(self):
        self.page.text_input.setPlainText("字" * 201)
        self.assertFalse(self.page.generate_button.isEnabled())
        self.assertIn("200", self.page.status.text())

    def test_cancel_kills_unresponsive_child_and_blocks_switch(self):
        self.generate()
        previous = self.page.job.result
        self.page.text_input.setPlainText("WAIT")
        self.page.generate_button.click()
        self.wait(lambda: "WAITING" in self.page.job._log)
        self.assertFalse(self.window._select_mode("dsp"))
        self.assertTrue(self.page.stop_button.isEnabled())
        self.page.stop_button.click()
        self.wait(lambda: not self.page.job.busy)
        self.assertEqual(self.page.job.result, previous)
        self.assertIn("已取消", self.page.status.text())
        self.assertEqual(self.devices.streams, [])
        self.assertTrue(self.window._select_mode("dsp"))

    def test_play_stop_uses_shared_controller_and_only_selected_output(self):
        self.generate("LONG")
        self.select_output()
        with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Yes):
            self.page.play_button.click()
        controller = self.window.controller
        self.wait(lambda: controller.media_status.playback == "generated" and not controller.busy)
        self.assertTrue(controller.owns_devices)
        self.assertIn("朗读结果", self.window.recording_status.text())
        self.assertIn("朗读结果", self.window.audio_status.text())
        self.assertFalse(self.page.generate_button.isEnabled())
        self.assertFalse(self.window._select_mode("vc"))
        self.assertTrue(all(stream.kind == "output" for stream in self.devices.streams))
        self.page.stop_button.click()
        self.wait(lambda: controller.state is EngineState.STOPPED and not controller.busy)
        self.assertFalse(controller.owns_devices)
        self.assertTrue(self.page.generate_button.isEnabled())

    def test_playback_rejects_missing_device_and_requires_confirmation(self):
        self.generate()
        controller = self.window.controller
        self.assertFalse(controller.request_play_generated(
            self.page.job.result, 24000, output_ceiling=0.5,
        ))
        with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Yes):
            self.page.play_button.click()
        self.assertIn("Select an output", self.page.status.text())
        self.assertEqual(self.devices.streams, [])

    def test_playback_completes_naturally_and_result_is_replayable(self):
        self.generate()
        self.select_output()
        with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Yes):
            self.page.play_button.click()
        controller = self.window.controller
        self.wait(lambda: controller.media_status.playback == "generated" and not controller.busy)
        self.wait(lambda: controller.state is EngineState.STOPPED and not controller.busy)
        self.assertTrue(self.page.play_button.isEnabled())
        self.assertFalse(self.page.stop_button.isEnabled())

    def test_save_is_persistent_and_never_overwrites_existing_file(self):
        self.generate()
        destination = self.root / "saved.wav"
        with patch("voice_changer.ui.ai_pages.QFileDialog.getSaveFileName",
                   return_value=(str(destination), "WAV")):
            self.page.save_button.click()
            self.wait(lambda: not self.page.busy)
            content = destination.read_bytes()
            self.assertEqual(content, self.page.job.result.read_bytes())
            self.assertIn("已保存", self.page.status.text())
            self.page.save_button.click()
            self.wait(lambda: not self.page.busy)
            self.assertIn("Save failed", self.page._message)
            self.assertEqual(destination.read_bytes(), content)
        temporary_result = self.page.job.result
        self.window.close()
        self.wait(lambda: self.window.controller.is_closed)
        self.assertTrue(destination.is_file())
        self.assertFalse(temporary_result.exists())

    def test_missing_kokoro_model_disables_generation(self):
        self.page.kokoro_model = None
        self.page.refresh_model()
        self.page.update_controls()
        self.assertFalse(self.page.generate_button.isEnabled())
        self.assertIn("Kokoro", self.page.availability_notice.text())

    def test_unusable_kokoro_python_disables_generation(self):
        self.page.kokoro_python = str(self.root / "missing-python")
        self.page.refresh_model()
        self.page.update_controls()
        self.assertFalse(self.page.generate_button.isEnabled())
        self.assertIn("Kokoro", self.page.availability_notice.text())

    def test_failed_process_start_is_visible_and_cleans_temporary_files(self):
        self.page.job.start_kokoro(
            "female", "Hello", self.model, python=str(self.root / "missing-python"),
        )
        directory = Path(self.page.job._pending.name)
        self.wait(lambda: not self.page.job.busy)
        self.assertIn("Speech generation failed", self.page.job.message)
        self.assertFalse(directory.exists())
        self.assertIsNone(self.page.job.result)

    def test_close_cancels_child_and_cleans_temporary_files(self):
        self.page.text_input.setPlainText("WAIT")
        self.page.generate_button.click()
        self.wait(lambda: "WAITING" in self.page.job._log)
        directory = Path(self.page.job._pending.name)
        self.window.close()
        self.wait(lambda: self.window.controller.is_closed, timeout=6)
        self.assertFalse(self.page.job.busy)
        self.assertFalse(directory.exists())

    def test_playback_error_can_be_reset_with_stop(self):
        self.generate()
        self.select_output()
        self.devices.output_error = "deliberate output device failure"
        self.page.play_button.click()
        controller = self.window.controller
        self.wait(lambda: controller.state is EngineState.FAULT and not controller.busy)
        self.assertTrue(self.page.stop_button.isEnabled())
        self.page.stop_button.click()
        self.wait(lambda: controller.state is EngineState.STOPPED and not controller.busy)
        self.assertTrue(self.page.generate_button.isEnabled())

    def test_invalid_wave_is_reported_without_opening_output(self):
        self.generate()
        self.select_output()
        self.page.job.result.write_bytes(b"not WAV")
        self.page.play_button.click()
        controller = self.window.controller
        self.wait(lambda: not controller.busy)
        self.assertIn("Audio operation failed", self.page._message)
        self.assertEqual(self.devices.streams, [])

    def test_truncated_result_is_rejected(self):
        path = self.root / "truncated.wav"
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(24000)
            writer.writeframes(b"\x01\x00" * 24000)
        with path.open("r+b") as output:
            output.truncate(path.stat().st_size - 100)
        with self.assertRaisesRegex(ValueError, "truncated"):
            audio_info(path)


if __name__ == "__main__":
    unittest.main()
