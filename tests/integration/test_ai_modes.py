"""Offscreen tests for the model-free AI modes; synthetic/fake devices only."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QCoreApplication, QEvent, QPoint
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton, QScrollArea

from voice_changer.app.engine_controller import EngineController
from voice_changer.audio.engine import EngineState
from voice_changer.ui import ai_pages
from voice_changer.ui.main_window import MainWindow

from controller_fakes import FakeBackend
from device_fakes import FakeDeviceLibrary
from qt_wait import wait_until


class TempDataMixin:
    def make_temp_data(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        environment = patch.dict(os.environ, {
            "XDG_CONFIG_HOME": str(self.base / "config"),
            "XDG_DATA_HOME": str(self.base / "data"),
        })
        environment.start()
        self.addCleanup(environment.stop)

    def wait_for(self, condition, timeout=3):
        self.assertTrue(wait_until(condition, timeout), "Timed out")


class AiModeWindowTests(TempDataMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.make_temp_data()
        library = patch.dict(sys.modules, {"sounddevice": FakeDeviceLibrary()})
        library.start()
        self.addCleanup(library.stop)
        self.backend = None
        self.window = None

    def tearDown(self):
        if self.backend is not None:
            self.backend.start_release.set()
        if self.window is not None:
            self.window.close()
            self.wait_for(lambda: self.window.controller.is_closed)
            self.window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def open_window(self, controller=None):
        self.window = MainWindow(controller)
        self.window.resize(1280, 731)
        self.window.show()
        self.wait_for(lambda: self.window.controller.is_ready and not self.window.controller.busy
                      and not self.window._startup_refresh_due)
        return self.window

    def test_modes_switch_lazily_and_ai_actions_stay_unavailable(self):
        window = self.open_window()
        self.assertEqual(window.mode, "dsp")
        self.assertIsNone(window.tts_page)
        self.assertEqual(
            [window.mode_buttons[key].shortcut() for key in ("dsp", "tts", "vc")],
            [QKeySequence("Ctrl+1"), QKeySequence("Ctrl+2"), QKeySequence("Ctrl+3")],
        )

        window.mode_buttons["tts"].click()
        self.assertEqual(window.mode, "tts")
        self.assertIs(window.mode_stack.currentWidget(), window.tts_page)
        self.assertTrue(window.mode_buttons["tts"].isChecked())
        self.assertFalse(window.mode_buttons["dsp"].isChecked())
        page = window.tts_page
        self.assertTrue(page.availability_notice.isVisible())
        self.assertIn("AI 朗读不可用", page.availability_notice.text())
        for button in (page.generate_button, page.play_button, page.stop_button, page.save_button):
            self.assertFalse(button.isEnabled(), button.text())
        self.assertFalse(hasattr(page, "manage_button"))

        window.mode_buttons["vc"].click()
        page = window.vc_page
        self.assertIs(window.mode_stack.currentWidget(), page)
        self.assertIn("不会用原声直通", page.availability_notice.text())
        self.assertFalse(page.start_button.isEnabled())
        self.assertFalse(page.stop_button.isEnabled())
        texts = "\n".join(label.text() for label in page.findChildren(type(page.availability_notice)))
        self.assertIn("尚未测量", texts)
        self.assertIn("暂不设合格线", texts)
        self.assertEqual(window.controller.state, EngineState.STOPPED)
        self.assertFalse(window.controller.owns_devices)

        window.mode_buttons["dsp"].click()
        self.assertIs(window.mode_stack.currentWidget(), window.dsp_page)

    def test_navigation_has_only_three_modes_in_both_languages_and_sizes(self):
        window = self.open_window()
        for width, height in ((1280, 731), (480, 400)):
            window.resize(width, height)
            for _ in range(2):
                self.app.processEvents()
                self.assertEqual(
                    set(window.navigation.findChildren(QPushButton)),
                    set(window.mode_buttons.values()),
                )
                self.assertEqual(list(window.mode_buttons), ["dsp", "tts", "vc"])
                for mode in ("tts", "vc", "dsp"):
                    window.mode_buttons[mode].click()
                    self.app.processEvents()
                    self.assertEqual(window.mode, mode)
                    self.assertTrue(window.mode_buttons[mode].isChecked())
                self.assertFalse(hasattr(window.tts_page, "manage_button"))
                window.language_button.click()

    def test_workspaces_share_geometry_in_both_languages_and_all_window_sizes(self):
        window = self.open_window()
        window._ensure_ai_pages()
        draft = "用户输入 / user input\nKeep this exactly."
        window.tts_page.text_input.setPlainText(draft)
        for width, height in ((480, 400), (640, 440), (899, 600), (900, 600), (1280, 800)):
            window.resize(width, height)
            self.app.processEvents()
            workspace_geometry = window.mode_stack.geometry()
            footer_geometry = window.footer_stack.geometry()
            for _ in range(2):
                for mode in ("dsp", "vc", "tts"):
                    with self.subTest(size=(width, height), mode=mode, language=window._language):
                        window.mode_buttons[mode].click()
                        self.app.processEvents()
                        self.assertEqual(window.size().toTuple(), (width, height))
                        self.assertEqual(window.mode_stack.geometry(), workspace_geometry)
                        self.assertEqual(window.footer_stack.geometry(), footer_geometry)
                        self.assertEqual(window.tts_page.text_input.toPlainText(), draft)
                        footer = window.footer_stack.currentWidget()
                        for button in (
                            (window.listen_button, window.record_button, window.stop_button)
                            if mode == "dsp" else
                            (window.vc_page.start_button, window.vc_page.stop_button)
                            if mode == "vc" else
                            (window.tts_page.generate_button, window.tts_page.stop_button)
                        ):
                            bounds = button.rect().translated(button.mapTo(footer, QPoint()))
                            self.assertTrue(footer.rect().contains(bounds), str(bounds))
                        for button in window.mode_buttons.values():
                            self.assertTrue(button.isVisibleTo(window))
                            self.assertTrue(window.rect().contains(
                                button.rect().translated(button.mapTo(window, QPoint())),
                            ))
                        self.assertEqual(window.controller.state, EngineState.STOPPED)
                        self.assertFalse(window.controller.owns_devices)
                window.language_button.click()
                self.app.processEvents()

    def test_long_ai_status_scrolls_without_displacing_fixed_actions(self):
        window = self.open_window()
        window.resize(480, 400)
        window._select_mode("vc")
        page = window.vc_page
        page.show_error("Model error " * 200)
        self.app.processEvents()
        footer = window.footer_stack.currentWidget()
        scroll = footer.findChild(QScrollArea)
        self.assertGreater(scroll.verticalScrollBar().maximum(), 0)
        self.assertTrue(footer.rect().contains(
            page.stop_button.rect().translated(page.stop_button.mapTo(footer, QPoint())),
        ))
        self.assertEqual(window.size().toTuple(), (480, 400))

    def test_ai_device_shortcut_reveals_compact_dsp_selection_without_starting_audio(self):
        window = self.open_window()
        window.resize(480, 400)
        self.app.processEvents()
        window._select_mode("vc")
        window.vc_page.devices_button.click()
        self.app.processEvents()
        self.assertEqual(window.mode, "dsp")
        self.assertTrue(window.device_selector_container.isVisible())
        self.assertTrue(window.device_toggle.isChecked())
        self.assertEqual(window.controller.state, EngineState.STOPPED)
        self.assertFalse(window.controller.owns_devices)

    def test_llvc_start_without_confirmation_passes_fixed_runtime_factory(self):
        from voice_changer.ai.llvc import LlvcProcessorFactory
        from voice_changer.ai.model_inspection import ModelInspection

        model = self.base / "llvc.onnx"
        model.write_bytes(b"test")
        with patch.dict(os.environ, {"VOICE_CHANGER_LLVC_MODEL": str(model)}), patch(
            "voice_changer.ai.llvc.unavailable_reason", return_value=None,
        ), patch(
            "voice_changer.ai.model_inspection.inspect_model_file",
            return_value=ModelInspection("llvc_onnx", "interface accepted", 832),
        ):
            window = self.open_window()
            window._select_mode("vc")
            page = window.vc_page
            self.wait_for(lambda: page._selected_model_info() is not None)
        self.assertTrue(page._model_buttons[model].isChecked())
        self.assertTrue(page._model_buttons[model].isEnabled())
        self.assertFalse(page.start_button.isEnabled())
        selection = window.controller.device_selection
        for kind in ("input", "output"):
            ref = next(
                d.ref for d in selection.devices
                if getattr(d, f"max_{kind}_channels") > 0
            )
            window.controller.select_device(kind, ref)
        self.assertTrue(page.start_button.isEnabled())
        with patch.dict(os.environ, {"VOICE_CHANGER_LLVC_MODEL": str(model)}), patch(
            "voice_changer.ui.ai_pages.QMessageBox.question",
        ) as question, patch.object(
            window.controller, "request_listen", return_value=True,
        ) as listen:
            page.start_button.click()
            question.assert_not_called()
            listen.assert_called_once()
            config = listen.call_args.args[0]
            self.assertEqual((config.samplerate, config.blocksize, config.channels), (48000, 1024, 1))
            self.assertEqual(config.output_ceiling, 0.5)
            self.assertTrue(listen.call_args.kwargs["output_confirmed"])
            self.assertEqual(listen.call_args.kwargs["processor_factory"], LlvcProcessorFactory(model))
        window.controller._state = EngineState.RUNNING
        try:
            self.assertFalse(window._select_mode("dsp"))
            page.update_controls()
            self.assertIn("running", page.status.text())
        finally:
            window.controller._state = EngineState.STOPPED
            page.update_controls()

    def test_vc_model_buttons_switch_models_without_exposing_reference_voices(self):
        import importlib
        from voice_changer.ai.availability import Availability
        from voice_changer.ai.model_inspection import ModelInspection

        model = self.base / "llvc.onnx"
        other = self.base / "other.pth"
        model.write_bytes(b"test")
        other.write_bytes(b"test")
        # setUp() patches sys.modules; mock restores exact snapshots, so any module
        # imported while patched loses its sys.modules entry on cleanup.  Hold a
        # module reference and patch the live object instead of a dotted string,
        # otherwise the patch can miss the freshly imported ai_pages module.
        with patch.dict(os.environ, {"VOICE_CHANGER_LLVC_MODEL": str(model)}), patch.object(
            ai_pages, "availability", return_value=Availability("vc", True, ""),
        ), patch.object(
            importlib.import_module("voice_changer.ai.model_inspection"), "inspect_model_file",
            return_value=ModelInspection("llvc_onnx", "interface accepted", 832),
        ):
            window = self.open_window()
            window._select_mode("vc")
            page = window.vc_page
            self.assertTrue(
                wait_until(lambda: page._selected_model_info() is not None, 3),
                f"configured={page._configured_model}, state={page.state}, "
                f"pending={page._model_check_pending}, selected={page._selected_model_path()}, "
                f"status={page.model_status.text()}",
            )
            self.assertTrue(page._model_buttons[model].isChecked())
            self.assertIn("晓晓", page._model_buttons[model].text())
            self.assertFalse(hasattr(page, "voice_buttons"))
            self.assertFalse(hasattr(page, "manage_button"))
            with patch.object(
                importlib.import_module("voice_changer.ai.model_inspection"), "inspect_model_file",
                return_value=ModelInspection("pytorch_checkpoint", "not accepted", None),
            ):
                page._inspect_model(other)
                self.wait_for(lambda: not page._model_check_pending)
            self.assertTrue(page._model_buttons[other].isChecked())
            self.assertFalse(page._model_buttons[model].isChecked())
            self.assertFalse(page.start_button.isEnabled())
            page._model_buttons[model].click()
            self.wait_for(lambda: not page._model_check_pending)
            self.assertEqual(page._selected_model_path(), model)
            self.assertTrue(page._model_buttons[model].isChecked())
            self.assertFalse(hasattr(window.tts_page, "manage_button"))

    def test_vc_model_buttons_adapt_and_preserve_filenames_in_both_languages(self):
        window = self.open_window()
        window._select_mode("vc")
        page = window.vc_page
        models = [self.base / name for name in ("模型 one.pth", "Second.pth", "Third.pth")]
        for model in models:
            page._add_model_button(model)
        self.assertTrue(page.model_button.isEnabled())
        for width, height in ((480, 400), (640, 440), (1280, 731)):
            window.resize(width, height)
            for _ in range(2):
                self.app.processEvents()
                for model in models:
                    self.assertEqual(page._model_buttons[model].text(), model.stem)
                self.assertEqual(page.scroll_area.horizontalScrollBar().maximum(), 0)
                self.assertEqual(window.size().toTuple(), (width, height))
                columns = 3 if page.width() >= 700 else 2
                for index, button in enumerate(page._model_buttons.values()):
                    grid_index = page.model_grid.indexOf(button)
                    self.assertEqual(
                        page.model_grid.getItemPosition(grid_index)[:2],
                        (index // columns, index % columns),
                    )
                window.language_button.click()

    def test_vc_device_names_are_not_localized_or_split(self):
        window = self.open_window()
        window._select_mode("vc")
        page = window.vc_page
        selection = window.controller.device_selection
        ref = next(info.ref for info in selection.devices if info.max_output_channels)
        window.controller.select_device("output", ref)
        for _ in range(2):
            self.app.processEvents()
            self.assertEqual(page.output_label.text(), ref.name)
            window.language_button.click()

    def test_local_llvc_onnx_is_interface_checked_and_selected_for_start(self):
        from voice_changer.ai.model_inspection import ModelInspection

        model = self.base / "local-llvc.onnx"
        model.write_bytes(b"test")
        window = self.open_window()
        window._select_mode("vc")
        page = window.vc_page
        with patch(
            "voice_changer.ui.ai_pages.QFileDialog.getOpenFileName",
            return_value=(str(model), ""),
        ), patch(
            "voice_changer.ai.model_inspection.inspect_model_file",
            return_value=ModelInspection("llvc_onnx", "interface accepted", 832),
        ), patch(
            "voice_changer.ui.ai_pages.importlib.util.find_spec", return_value=object(),
        ):
            page.choose_local_model()
            self.wait_for(lambda: model.resolve() in page._model_inspections)
            self.assertEqual(page._selected_model_path(), model.resolve())
            self.assertIn("52 ms", page.model_status.text())
            self.assertIn("interface accepted", page.model_status.text())
            self.assertFalse(page.start_button.isEnabled())
            selection = window.controller.device_selection
            for kind in ("input", "output"):
                ref = next(
                    d.ref for d in selection.devices
                    if getattr(d, f"max_{kind}_channels") > 0
                )
                window.controller.select_device(kind, ref)
        with patch(
            "voice_changer.ui.ai_pages.importlib.util.find_spec", return_value=object(),
        ), patch.object(
            window.controller, "request_listen", return_value=True,
        ) as listen:
            page.update_controls()
            self.assertTrue(page.start_button.isEnabled())
            page.start_button.click()
            factory = listen.call_args.kwargs["processor_factory"]
            self.assertEqual(factory.model_path, model.resolve())
            self.assertTrue(factory.allow_custom_model)

    def test_invalid_local_model_is_reported_without_selecting_a_fallback(self):
        model = self.base / "invalid.onnx"
        model.write_bytes(b"invalid")
        window = self.open_window()
        window._select_mode("vc")
        page = window.vc_page
        with patch(
            "voice_changer.ui.ai_pages.QFileDialog.getOpenFileName",
            return_value=(str(model), ""),
        ), patch(
            "voice_changer.ai.model_inspection.inspect_model_file",
            side_effect=ValueError("state shape mismatch"),
        ):
            page.choose_local_model()
            self.wait_for(lambda: not page._model_check_pending)
        self.assertEqual(page._selected_model_path(), model)
        self.assertIn("state shape mismatch", page.model_error.text())
        self.assertFalse(page.start_button.isEnabled())

    def test_vc_remembers_models_and_rechecks_when_selected(self):
        from voice_changer.ai.model_inspection import ModelInspection
        from voice_changer.ui.ai_pages import AiVcPage

        model = self.base / "saved-model.pth"
        model.write_bytes(b"test")
        window = self.open_window()
        window._select_mode("vc")
        page = window.vc_page
        with patch(
            "voice_changer.ui.ai_pages.QFileDialog.getOpenFileName", return_value=(str(model), ""),
        ), patch(
            "voice_changer.ai.model_inspection.inspect_model_file",
            return_value=ModelInspection("pytorch_checkpoint", "not accepted"),
        ):
            page.choose_local_model()
            self.wait_for(lambda: not page._model_check_pending)
        restored = AiVcPage(window.controller)
        self.addCleanup(restored.deleteLater)
        self.assertIn(model, restored._model_buttons)
        self.assertNotIn(model, restored._model_inspections)
        model.unlink()
        with patch(
            "voice_changer.ai.model_inspection.inspect_model_file",
            side_effect=ValueError("model is missing"),
        ) as inspect:
            restored._model_buttons[model].click()
            self.wait_for(lambda: not restored._model_check_pending)
        inspect.assert_called_once_with(model)
        self.assertIn("model is missing", restored.model_error.text())
        self.assertFalse(restored.start_button.isEnabled())

    def test_vc_cancel_and_busy_model_checks_do_not_change_selection(self):
        window = self.open_window()
        window._select_mode("vc")
        page = window.vc_page
        selected = page._selected_model_path()
        with patch("voice_changer.ui.ai_pages.QFileDialog.getOpenFileName", return_value=("", "")):
            page.choose_local_model()
        self.assertEqual(page._selected_model_path(), selected)
        page._model_check_pending = True
        page.update_controls()
        self.assertFalse(page.model_button.isEnabled())
        self.assertFalse(page.start_button.isEnabled())
        page._inspect_model(self.base / "other.onnx")
        self.assertEqual(page._selected_model_path(), selected)
        page._model_check_pending = False
        page.update_controls()

    def test_selected_model_is_auto_detected_and_experimental_types_stay_disabled(self):
        from voice_changer.ai.model_inspection import ModelInspection

        window = self.open_window()
        window._select_mode("vc")
        page = window.vc_page
        self.assertFalse(hasattr(page, "backend_combo"))
        candidates = (
            ("rvc-model.pth", "pytorch_checkpoint"),
            ("rvc-synth.onnx", "rvc_onnx_synthesis"),
            ("rvc-synth.rknn", "rvc_rknn"),
            ("llvc.rknn", "llvc_rknn"),
            ("hubert-component.onnx", "rvc_onnx_content"),
            ("rmvpe-component.onnx", "rvc_onnx_pitch"),
        )
        for filename, model_type in candidates:
            path = self.base / filename
            path.write_bytes(b"diagnostic candidate")
            with patch(
                "voice_changer.ui.ai_pages.QFileDialog.getOpenFileName",
                return_value=(str(path), ""),
            ) as dialog, patch(
                "voice_changer.ai.model_inspection.inspect_model_file",
                return_value=ModelInspection(model_type, "experimental model"),
            ):
                page.choose_local_model()
                self.wait_for(lambda: path.resolve() in page._model_inspections)
            self.assertEqual(page._selected_model_path(), path.resolve())
            self.assertIn("不会加载或推理", page.model_status.text())
            self.assertIn(
                page._MODEL_LABELS[model_type].split(" / ", 1)[0],
                page.model_status.text(),
            )
            self.assertFalse(page.start_button.isEnabled())
            self.assertIn("All files (*)", dialog.call_args.args[3])
            with patch.object(window.controller, "request_listen") as listen:
                page.start_conversion()
                listen.assert_not_called()

    def test_experimental_hevc_models_are_named_and_renamable_but_never_start(self):
        from voice_changer.ai.model_inspection import ModelInspection

        window = self.open_window()
        window._select_mode("vc")
        page = window.vc_page
        samples = (
            ("female_voice_hfg_new.onnx", "自然女生", "Natural Female"),
            ("female_voice_old_hq.onnx", "清亮女声", "Bright Female"),
            ("female_voice_old.onnx", "柔和女声", "Gentle Female"),
        )
        self.assertFalse(page.model_hint.isVisible())
        self.assertFalse(page.model_status.isVisible())
        for filename, zh_label, en_label in samples:
            path = self.base / filename
            path.write_bytes(b"hevc interface fixture")
            with patch(
                "voice_changer.ui.ai_pages.QFileDialog.getOpenFileName",
                return_value=(str(path), ""),
            ), patch(
                "voice_changer.ai.model_inspection.inspect_model_file",
                return_value=ModelInspection("hevc_onnx", "HEVoicechanger ONNX detected."),
            ):
                page.choose_local_model()
                self.wait_for(lambda: path.resolve() in page._model_inspections)
            self.assertEqual(page._selected_model_path(), path.resolve())
            button = page._model_buttons[path.resolve()]
            self.wait_for(lambda: button.text() == zh_label)
            self.assertEqual(button.text(), zh_label)
            window.language_button.click()
            self.wait_for(lambda: button.text() == en_label)
            self.assertEqual(button.text(), en_label)
            window.language_button.click()
            self.wait_for(lambda: button.text() == zh_label)
            self.assertIn("HEVoicechanger", page.model_status.text())
            self.assertIn("不会加载或推理", page.model_status.text())
            self.assertIn("无法实时", page.model_status.text())
            self.assertIn("无法实时", page.availability_notice.text())
            self.assertFalse(page.start_button.isEnabled())
            with patch.object(window.controller, "request_listen") as listen:
                page.start_conversion()
                listen.assert_not_called()

        first = (self.base / samples[0][0]).resolve()
        page._selected_model = first
        page.update_controls()
        self.assertTrue(page.rename_button.isEnabled())
        with patch(
            "voice_changer.ui.ai_pages.QInputDialog.getText",
            return_value=("我的实验女声", True),
        ):
            page.rename_model()
        names_file = self.base / "config" / "voice-changer" / "model-names.json"
        payload = json.loads(names_file.read_text(encoding="utf-8"))
        self.assertEqual(payload["names"]["hevc_hfg_new"], "我的实验女声")
        self.assertEqual(page._model_buttons[first].text(), "我的实验女声")
        self.assertFalse(page.start_button.isEnabled())
        with patch(
            "voice_changer.ui.ai_pages.QInputDialog.getText", return_value=("", True),
        ):
            page.rename_model()
        self.wait_for(lambda: page._model_buttons[first].text() == "自然女生")
        self.assertNotIn(
            "hevc_hfg_new",
            json.loads(names_file.read_text(encoding="utf-8"))["names"],
        )

    def test_offline_preview_records_converts_plays_and_saves_for_hevc(self):
        import numpy as np
        from voice_changer.ai.model_inspection import ModelInspection

        model = self.base / "female_voice_hfg_new.onnx"
        model.write_bytes(b"hevc preview fixture")
        window = self.open_window()
        window._select_mode("vc")
        page = window.vc_page
        with patch(
            "voice_changer.ui.ai_pages.QFileDialog.getOpenFileName",
            return_value=(str(model), ""),
        ), patch(
            "voice_changer.ai.model_inspection.inspect_model_file",
            return_value=ModelInspection("hevc_onnx", "HEVoicechanger ONNX detected."),
        ):
            page.choose_local_model()
            self.wait_for(lambda: model.resolve() in page._model_inspections)
        selection = window.controller.device_selection
        for kind in ("input", "output"):
            ref = next(
                device.ref for device in selection.devices
                if getattr(device, f"max_{kind}_channels") > 0
            )
            window.controller.select_device(kind, ref)
        output_index = selection.output_device.index
        saved = self.base / "data" / "voice-changer" / "previews" / "preview.wav"
        recorded = {}

        def fake_record(seconds, *, device=None):
            recorded["record"] = (seconds, device)
            return np.zeros(48000, dtype=np.float32)

        def fake_convert(samples, rate, target):
            recorded["convert"] = (samples.size, rate, target)
            return np.ones(48000, dtype=np.float32)

        def fake_save(path, samples, rate=48000):
            recorded["saved_path"] = path
            recorded["save_rate"] = rate
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(b"wav")
            return saved

        def fake_play(samples, *, device=None):
            recorded["played"] = (samples.size, device)

        with patch(
            "voice_changer.ai.hevc_preview.record", side_effect=fake_record,
        ), patch(
            "voice_changer.ai.hevc_preview.convert", side_effect=fake_convert,
        ), patch(
            "voice_changer.ai.hevc_preview.save_wav", side_effect=fake_save,
        ), patch(
            "voice_changer.ai.hevc_preview.play", side_effect=fake_play,
        ), patch(
            "voice_changer.ui.ai_pages.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ), patch(
            "voice_changer.ui.ai_pages.QMessageBox.information",
            return_value=None,
        ), patch.object(window.controller, "request_listen") as listen:
            page.preview_button.click()
            self.wait_for(lambda: page._preview_note is not None)
            listen.assert_not_called()

        self.assertTrue(page.preview_button.isEnabled())
        self.assertEqual(recorded["record"][0], 5)
        self.assertEqual(recorded["convert"], (48000, 48000, model.resolve()))
        self.assertEqual(recorded["save_rate"], 48000)
        self.assertTrue(recorded["saved_path"].name.startswith("hevc-female_voice_hfg_new-"))
        self.assertEqual(recorded["played"], (48000, output_index))
        self.assertTrue(saved.is_file())
        self.assertIn("离线试听", page.status.text())
        self.assertIn("已保存", page.status.text())
        self.assertFalse(page.start_button.isEnabled())

    def make_beatrice_tree(self):
        from voice_changer.ai import beatrice

        api_dir = self.base / "beaapi"
        package = api_dir / "beatrice"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        model_dir = self.base / "bea-model"
        model_dir.mkdir()
        for name in beatrice.PARAMETER_FILES:
            (model_dir / name).write_bytes(b"\0")
        return api_dir, model_dir

    def test_beatrice_experiment_toggle_enables_start_without_llvc_model(self):
        from voice_changer.ai import beatrice

        api_dir, model_dir = self.make_beatrice_tree()
        environment = {
            beatrice.API_ENV: str(api_dir),
            beatrice.MODEL_ENV: str(model_dir),
        }
        with patch.dict(os.environ, environment):
            window = self.open_window()
            window._select_mode("vc")
            page = window.vc_page
            self.wait_for(lambda: page.beatrice_checkbox.isEnabled())
            self.assertTrue(page.beatrice_checkbox.isEnabled())
            self.assertIsNone(page._beatrice_engine())

            page.beatrice_checkbox.setChecked(True)
            engine = page._beatrice_engine()
            self.assertIsNotNone(engine)
            self.assertEqual(engine.api_dir, api_dir)
            self.assertEqual(engine.speaker, page._beatrice_speaker)
            self.assertEqual(engine.pitch_shift_semitone, 12.0)
            self.assertEqual(engine.formant_shift, 0.0)
            self.assertEqual(engine.external_formant, 1.18)
            page._select_beatrice_voice(10)
            selected = page._beatrice_engine()
            self.assertEqual(selected.speaker, 9)
            self.assertEqual(selected.pitch_shift_semitone, 12.0)
            self.assertTrue(page.beatrice_voices[10].isChecked())
            page.update_controls()
            label = page._beatrice_voice_label()
            self.assertIn("JVS 说话人 10", label)
            self.assertIn("升调 +12.0 半音", label)
            self.assertFalse(page.availability_notice.isVisible())
            self.assertFalse(page.start_button.isEnabled())

            selection = window.controller.device_selection
            for kind in ("input", "output"):
                ref = next(
                    d.ref for d in selection.devices
                    if getattr(d, f"max_{kind}_channels") > 0
                )
                window.controller.select_device(kind, ref)
            page.update_controls()
            self.assertTrue(page.start_button.isEnabled())

            with patch(
                "voice_changer.ui.ai_pages.QMessageBox.question",
            ) as question, patch.object(
                window.controller, "request_listen", return_value=True,
            ) as listen:
                page.start_button.click()
                question.assert_not_called()
                listen.assert_called_once()
                config = listen.call_args.args[0]
                self.assertEqual(
                    (config.samplerate, config.blocksize, config.channels), (48000, 1024, 1),
                )
                factory = listen.call_args.kwargs["processor_factory"]
                self.assertIsInstance(factory, beatrice.BeatriceProcessorFactory)
                self.assertEqual(factory.engine_config.speaker, 9)
                self.assertEqual(factory.engine_config.pitch_shift_semitone, 12.0)
                self.assertEqual(factory.engine_config.formant_shift, 0.0)
                self.assertEqual(factory.engine_config.external_formant, 1.18)
                self.assertEqual(factory.engine_config.api_dir, api_dir)
                self.assertIsNone(page._beatrice_reason)

    def test_beatrice_offline_preview_uses_the_selected_jvs_voice(self):
        import numpy as np

        from voice_changer.ai import beatrice
        from voice_changer.ui import ai_pages

        api_dir, model_dir = self.make_beatrice_tree()
        environment = {
            beatrice.API_ENV: str(api_dir),
            beatrice.MODEL_ENV: str(model_dir),
        }
        with patch.dict(os.environ, environment):
            window = self.open_window()
            window._select_mode("vc")
            page = window.vc_page
            self.wait_for(lambda: page.beatrice_checkbox.isEnabled())
            page.beatrice_checkbox.setChecked(True)
            self.wait_for(lambda: page._beatrice_engine() is not None)
            # Speaker choice is button-based; the default is the first female candidate.
            self.assertEqual(len(page.beatrice_voices), 4)
            self.assertTrue(page.beatrice_voices[10].isChecked())
            self.assertEqual(page._beatrice_speaker, 9)
            page.beatrice_voices[14].click()
            self.assertTrue(page.beatrice_voices[14].isChecked())
            self.assertFalse(page.beatrice_voices[10].isChecked())
            self.assertEqual(page._beatrice_speaker, 13)

            selection = window.controller.device_selection
            for kind in ("input", "output"):
                ref = next(
                    device.ref for device in selection.devices
                    if getattr(device, f"max_{kind}_channels") > 0
                )
                window.controller.select_device(kind, ref)
            output_index = selection.output_device.index
            page.update_controls()
            self.assertTrue(page.preview_button.isEnabled())

            recorded = {}

            def fake_record(seconds, *, device=None):
                recorded["record"] = (seconds, device)
                return np.zeros(48000, dtype=np.float32)

            def fake_convert(samples, rate, engine):
                recorded["convert"] = (
                    samples.size, rate, engine.speaker,
                    engine.pitch_shift_semitone, engine.formant_shift,
                    engine.external_formant,
                )
                return np.ones(48000, dtype=np.float32)

            def fake_save(path, samples, rate=48000):
                recorded["saved_path"] = path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"wav")
                return path

            def fake_play(samples, *, device=None):
                recorded["played"] = (samples.size, device)

            with patch(
                "voice_changer.ai.hevc_preview.record", side_effect=fake_record,
            ), patch(
                "voice_changer.ai.beatrice_preview.convert", side_effect=fake_convert,
            ), patch(
                "voice_changer.ai.hevc_preview.save_wav", side_effect=fake_save,
            ), patch(
                "voice_changer.ai.hevc_preview.play", side_effect=fake_play,
            ), patch(
                "voice_changer.ui.ai_pages.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ), patch(
                "voice_changer.ui.ai_pages.QMessageBox.information",
                return_value=None,
            ), patch.object(window.controller, "request_listen") as listen:
                page.preview_button.click()
                self.wait_for(lambda: page._preview_note is not None)
                listen.assert_not_called()

            self.assertEqual(recorded["record"][0], 5)
            self.assertEqual(recorded["convert"], (48000, 48000, 13, 12.0, 0.0, 1.18))
            self.assertEqual(recorded["played"], (48000, output_index))
            self.assertTrue(
                recorded["saved_path"].name.startswith("beatrice-jvs014-"),
            )
            self.assertTrue(recorded["saved_path"].is_file())
            self.assertIn("已保存", page.status.text())
            self.assertIn("离线试听", page.status.text())
            # The Beatrice live path stays selectable (with a preflight warning);
            # the offline preview does not change that.
            self.assertTrue(page.start_button.isEnabled())

    def test_beatrice_underrun_error_is_explained(self):
        window = self.open_window()
        window._select_mode("vc")
        page = window.vc_page
        page.show_error(
            RuntimeError(
                "Audio callback processing failed: RVC output queue underrun; AI listening must stop"
            )
        )
        status = page.status.text()
        self.assertIn("欠载", status)
        self.assertIn("LLVC", status)
        self.assertIn("underran", status)

    def test_beatrice_toggle_is_disabled_without_configured_engine(self):
        from voice_changer.ai import beatrice

        with patch.dict(os.environ, {beatrice.API_ENV: "", beatrice.MODEL_ENV: ""}):
            window = self.open_window()
            window._select_mode("vc")
            page = window.vc_page
            page.update_controls()
            self.assertFalse(page.beatrice_checkbox.isEnabled())
            self.assertIn(beatrice.API_ENV, page.beatrice_checkbox.toolTip())
            self.assertIsNone(page._beatrice_engine())

    def test_mode_switch_is_refused_while_dsp_audio_runs(self):
        self.backend = FakeBackend()
        controller = EngineController(self.backend)
        window = self.open_window(controller)
        self.assertTrue(controller.request_start())
        self.wait_for(lambda: controller.state is EngineState.RUNNING and not controller.busy)
        window.mode_buttons["vc"].click()
        self.assertEqual(window.mode, "dsp")
        self.assertTrue(window.mode_buttons["dsp"].isChecked())
        self.assertFalse(window.mode_buttons["vc"].isChecked())
        self.assertIn("不会自动停止", window.error_notice.text())
        self.assertEqual(controller.state, EngineState.RUNNING)
        self.assertTrue(controller.request_stop())
        self.wait_for(lambda: controller.state is EngineState.STOPPED and not controller.busy)
        window.mode_buttons["vc"].click()
        self.assertEqual(window.mode, "vc")

    def test_compact_window_keeps_mode_buttons_visible(self):
        window = self.open_window()
        window.resize(480, 400)
        self.app.processEvents()
        self.wait_for(lambda: window._small_layout)
        self.assertFalse(window.app_title.isVisible())
        for button in window.mode_buttons.values():
            self.assertTrue(button.isVisible())
            self.assertLessEqual(button.mapTo(window, button.rect().bottomRight()).x(), window.width())
        window.mode_buttons["tts"].click()
        self.app.processEvents()
        self.assertTrue(window.tts_page.generate_button.isVisible())


    def test_hevc_rknn_model_starts_the_npu_factory(self):
        import hashlib
        import importlib

        from voice_changer.ai import hevc_rknn
        from voice_changer.ai.model_inspection import ModelInspection

        model = self.base / "female_voice_hfg_new_rknn.rknn"
        model.write_bytes(b"hevc rknn integration fixture")
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        Path(f"{model}.json").write_text(json.dumps({
            "format": "voice-changer-hevc-rknn-v1",
            "target_platform": "rk3576",
            "toolkit_version": hevc_rknn.TOOLKIT_VERSION,
            "model_sha256": digest,
            "frames": 11264,
            "history": 8192,
            "chunk": 3072,
            "source_onnx_sha256": (
                "e0084cf7eaa0956f284037dad6ac8344563c936132c47d8ed3cb74202d0633b6"
            ),
            "simulator_snr_db": 60.0,
        }), encoding="utf-8")

        inspection_module = importlib.import_module("voice_changer.ai.model_inspection")
        llvc_module = importlib.import_module("voice_changer.ai.llvc")
        inspection = ModelInspection(
            "hevc_rknn",
            "HEVoicechanger RKNN / NPU package detected. Window and simulator "
            "accuracy are validated; live latency and quality still need listening "
            "acceptance.",
            11264,
            digest,
        )
        with patch.object(
            inspection_module, "inspect_model_file", return_value=inspection,
        ), patch.object(hevc_rknn, "unavailable_reason", return_value=None):
            window = self.open_window()
            window._select_mode("vc")
            page = window.vc_page
            with patch(
                "voice_changer.ui.ai_pages.QFileDialog.getOpenFileName",
                return_value=(str(model), ""),
            ):
                page.choose_local_model()
                self.wait_for(lambda: model.resolve() in page._model_inspections)
            button = page._model_buttons[model.resolve()]
            self.wait_for(lambda: button.text() == "自然女生")
            self.assertEqual(button.text(), "自然女生")
            self.assertIn("HEVoicechanger RKNN / NPU", page.model_status.text())
            self.assertEqual(page._selected_model_path(), model.resolve())
            self.assertFalse(page.start_button.isEnabled())
            self.assertFalse(page.availability_notice.isVisible())

            selection = window.controller.device_selection
            for kind in ("input", "output"):
                ref = next(
                    d.ref for d in selection.devices
                    if getattr(d, f"max_{kind}_channels") > 0
                )
                window.controller.select_device(kind, ref)
            page.update_controls()
            self.assertTrue(page.start_button.isEnabled())

            with patch.object(llvc_module, "LlvcProcessorFactory") as llvc_factory, patch.object(
                window.controller, "request_listen", return_value=True,
            ) as listen:
                page.start_button.click()

            llvc_factory.assert_not_called()
            listen.assert_called_once()
            config = listen.call_args.args[0]
            self.assertEqual(
                (config.samplerate, config.blocksize, config.channels), (48000, 1024, 1),
            )
            self.assertEqual(config.output_ceiling, 0.5)
            self.assertTrue(listen.call_args.kwargs["output_confirmed"])
            factory = listen.call_args.kwargs["processor_factory"]
            self.assertIsInstance(factory, hevc_rknn.HevcRknnProcessorFactory)
            self.assertEqual(factory.model_path, model.resolve())
            self.assertEqual(window.controller.state, EngineState.STOPPED)

if __name__ == "__main__":
    unittest.main()
