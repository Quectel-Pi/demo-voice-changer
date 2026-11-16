"""Explicitly opt-in, silent PySide6 prototype for layout and interaction review."""

import argparse
import sys

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from .prototype_i18n import load_language, save_language, settings_path, tr
from .audio_wave import AudioWave, WaveComparison
from .prototype_state import (
    BUILTIN_PRESETS, FAULTS, INPUT_DEVICES, OUTPUT_DEVICES, PrototypeState,
)
from .theme import set_visual_state, stylesheet
from .interactions import (
    FormInteractions, PageScrollDoubleSpinBox, parameter_slider, sync_parameter_slider,
)


class Key(str):
    """A message key that is translated when the notice is rendered."""


def label(text: str, name: str = "", wrap: bool = False) -> QLabel:
    item = QLabel(text)
    item.setObjectName(name)
    item.setWordWrap(wrap)
    return item


def panel() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(6)
    return frame, layout


class PrototypeWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.state = PrototypeState()
        self.language, problem, detail = load_language()
        self.notice_message: tuple[str, dict] = ("initial_notice", {})
        if problem:
            self.say(problem, path=str(settings_path()), error=detail)
        self.scenarios_visible = False
        self.audio_values = ["48000", "1024", "1", "0.05"]
        self._startup_refresh_due = True
        self._refresh_pending = False
        self._wave_side_by_side = False
        self._small_layout: bool | None = None
        self._monitor_in_priority_slot = False
        self.resize(640, 440)
        self.setMinimumSize(480, 400)
        self.setStyleSheet(stylesheet())
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        self.build_ui()
        self.timer.start()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._update_responsive_layout()
        if self._startup_refresh_due:
            QTimer.singleShot(0, lambda: self.refresh_devices(manual=False))

    # Text helpers ---------------------------------------------------------

    def t(self, key: str, **values) -> str:
        return tr(self.language, key, **{k: self.text_of(v) for k, v in values.items()})

    def text_of(self, value):
        return tr(self.language, value) if isinstance(value, Key) else value

    def say(self, key: str, **values) -> None:
        self.notice_message = (key, values)

    def preset_name(self, key: str):
        return Key(f"preset_{key}")

    # Construction ---------------------------------------------------------

    def build_ui(self) -> None:
        """(Re)create every widget in the current language; simulated state is kept."""
        old = self.takeCentralWidget()
        if old is not None:
            old.deleteLater()
        self._small_layout = None
        self._monitor_in_priority_slot = False
        self.setWindowTitle(self.t("window_title"))
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)

        header = QHBoxLayout()
        header.addWidget(label(self.t("app_title"), "title"))
        header.addStretch()
        self.language_button = self.button(self.t("switch_language"), self.toggle_language)
        self.language_button.setAccessibleName(self.t("switch_language_name"))
        self.language_button.setToolTip(self.t("switch_language_name"))
        header.addWidget(self.language_button)
        self.scenarios_toggle = self.button(self.t("scenarios"), self.toggle_scenarios)
        self.scenarios_toggle.setAccessibleName(self.t("scenarios"))
        header.addWidget(self.scenarios_toggle)
        outer.addLayout(header)
        outer.addWidget(label(self.t("banner"), "muted", True))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area = scroll
        self.form_interactions = FormInteractions(scroll)
        outer.addWidget(scroll, 1)
        canvas = QWidget()
        canvas.setObjectName("canvas")
        scroll.setWidget(canvas)
        content = QVBoxLayout(canvas)
        self.content_layout = content
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(8)
        self.devices_panel = self.build_devices()
        self.presets_panel = self.build_presets()
        self.monitoring_panel = self.build_monitoring()
        content.addWidget(self.devices_panel)
        content.addWidget(self.presets_panel)
        content.addWidget(self.monitoring_panel)
        self.tuning_panel = self.build_workspace()
        content.addWidget(self.tuning_panel)
        self.recording_panel = self.build_recording()
        content.addWidget(self.recording_panel)
        self.scenarios_panel = self.build_scenarios()
        self.scenarios_panel.setVisible(self.scenarios_visible)
        content.addWidget(self.scenarios_panel)
        content.addStretch()
        self.notice = label("", "muted", True)
        self.notice.hide()
        outer.addWidget(self.notice)

        footer = QWidget()
        footer.setObjectName("fixedFooter")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(8, 4, 8, 4)
        self.activity = label("", "muted", True)
        self.activity.setAccessibleName("模拟活动状态 / Simulated activity status")
        footer_layout.addWidget(self.activity, 1)
        actions = QHBoxLayout()
        actions.addWidget(self.monitor, 1)
        actions.addWidget(self.record, 1)
        self.stop = self.button(self.t("stop_all"), self.stop_all)
        self.stop.setAccessibleName(self.t("stop_all_name"))
        self.stop.setToolTip(self.t("stop_all_name"))
        actions.addWidget(self.stop)
        footer_layout.addLayout(actions)
        outer.addWidget(footer)
        self.sync_controls()
        self.render()
        self._update_responsive_layout()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "content_layout"):
            self._update_responsive_layout()

    def button(self, text: str, callback, primary: bool = False) -> QPushButton:
        item = QPushButton(text)
        item.setAccessibleName(text)
        item.clicked.connect(callback)
        if primary:
            item.setProperty("role", "primary")
        return item

    def build_devices(self) -> QFrame:
        frame, box = panel()
        row = QHBoxLayout()
        row.addWidget(label(self.t("devices"), "heading"))
        row.addStretch()
        self.device_toggle = QPushButton()
        self.device_toggle.setText(self.t("show_devices"))
        self.device_toggle.setAccessibleName(self.t("show_devices"))
        self.device_toggle.setCheckable(True)
        self.device_toggle.toggled.connect(self._update_device_disclosure)
        row.addWidget(self.device_toggle)
        self.refresh = self.button(self.t("refresh"), self.refresh_devices)
        row.addWidget(self.refresh)
        box.addLayout(row)
        box.addWidget(label(self.t("devices_hint"), "muted", True))
        self.discovery_status = label(self.t("discovered_simulated"), "muted", True)
        box.addWidget(self.discovery_status)
        self.device_selectors = QWidget()
        grid = QGridLayout(self.device_selectors)
        self.input = QComboBox()
        self.output = QComboBox()
        for kind, combo, devices, col in (
            ("input", self.input, INPUT_DEVICES, 0),
            ("output", self.output, OUTPUT_DEVICES, 1),
        ):
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon,
            )
            combo.setMinimumContentsLength(8)
            title = label(self.t(f"{kind}_title"))
            title.setBuddy(combo)
            combo.setAccessibleName(title.text())
            combo.addItems([self.t(f"{kind}_none")] + [self.t(f"device_{d}") for d in devices])
            combo.currentIndexChanged.connect(lambda index, k=kind: self.device_changed(k, index))
            grid.addWidget(title, col, 0)
            grid.addWidget(combo, col, 1)
            grid.setColumnStretch(1, 1)
        box.addWidget(self.device_selectors)
        self.device_hint = label("", "muted", True)
        box.addWidget(self.device_hint)
        return frame

    def build_presets(self) -> QFrame:
        frame, box = panel()
        row = QHBoxLayout()
        row.addWidget(label(self.t("presets"), "heading"))
        row.addStretch()
        self.preset_toggle = QPushButton()
        self.preset_toggle.setText(self.t("show_presets"))
        self.preset_toggle.setAccessibleName(self.t("show_presets"))
        self.preset_toggle.setCheckable(True)
        self.preset_toggle.toggled.connect(self._update_preset_disclosure)
        row.addWidget(self.preset_toggle)
        box.addLayout(row)
        box.addWidget(label(self.t("presets_hint"), "muted", True))
        self.preset_summary = label("", "muted", True)
        self.preset_summary.setAccessibleName(self.t("current_preset"))
        box.addWidget(self.preset_summary)
        self.presets: dict[str, QPushButton] = {}
        self.preset_options = QWidget()
        grid = QGridLayout(self.preset_options)
        for index, key in enumerate((*BUILTIN_PRESETS, "custom")):
            grid.addWidget(self.preset_button(key, self.t(f"preset_{key}")), index // 2, index % 2)
        box.addWidget(self.preset_options)
        return frame

    def preset_button(self, key: str, text: str) -> QPushButton:
        button = self.button(text, lambda checked=False, k=key: self.choose_preset(k))
        button.setCheckable(True)
        self.presets[key] = button
        return button

    def build_workspace(self) -> QFrame:
        frame, box = panel()
        box.addWidget(label(self.t("custom_voice"), "heading"))
        self.tuning_controls = QWidget()
        box.addWidget(self.tuning_controls)
        box = QVBoxLayout(self.tuning_controls)
        box.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.current = label("", "current", True)
        top.addWidget(self.current, 1)
        top.addWidget(self.button(self.t("reset_effect"), self.reset_effect))
        box.addLayout(top)
        box.addWidget(label(self.t("workspace_hint"), "muted", True))
        self.pitch_editor = PageScrollDoubleSpinBox()
        self.pitch_editor.setRange(-12, 12)
        self.pitch_editor.setDecimals(0)
        self.pitch_editor.setKeyboardTracking(False)
        self.pitch_editor.setAccessibleName(self.t("pitch_name"))
        self.pitch = parameter_slider(self.pitch_editor, 1)
        self.pitch_editor.valueChanged.connect(lambda value: self.pitch_changed(int(value)))
        pitch_row = QHBoxLayout()
        title = label(self.t("pitch_title"))
        title.setBuddy(self.pitch)
        pitch_row.addWidget(title)
        pitch_row.addStretch()
        self.pitch_value = label("")
        pitch_row.addWidget(self.pitch_value)
        box.addLayout(pitch_row)
        pitch_controls = QHBoxLayout()
        pitch_controls.addWidget(self.pitch, 1)
        pitch_controls.addWidget(self.pitch_editor)
        box.addLayout(pitch_controls)
        endpoints = QHBoxLayout()
        endpoints.addWidget(label(self.t("pitch_min"), "muted"))
        endpoints.addStretch()
        endpoints.addWidget(label(self.t("pitch_max"), "muted"))
        box.addLayout(endpoints)
        self.parameters = {}
        self.parameter_editors = {}
        self.parameter_levels = {}
        for key, options in (
            ("formant", ("level_baseline", "level_lower", "level_higher")),
            ("robot", ("level_baseline", "level_less", "level_more")),
            ("echo", ("level_baseline", "level_less", "level_more")),
            ("reverb", ("level_baseline", "level_less", "level_more")),
            ("input_gain", ("level_gain_default", "level_decrease", "level_increase")),
            ("output_gain", ("level_gain_default", "level_decrease", "level_increase")),
        ):
            name = self.t(f"param_{key}")
            editor = PageScrollDoubleSpinBox()
            editor.setRange(0, 2)
            editor.setDecimals(0)
            editor.setKeyboardTracking(False)
            editor.setAccessibleName(self.t("param_name", name=name))
            slider = parameter_slider(editor, 1)
            title = label(self.t("param_title", name=name), wrap=True)
            title.setBuddy(slider)
            box.addWidget(title)
            row = QHBoxLayout()
            row.addWidget(slider, 1)
            row.addWidget(editor)
            box.addLayout(row)
            level = label("", "muted", True)
            box.addWidget(level)
            editor.valueChanged.connect(lambda value, k=key: self.parameter_changed(k, int(value)))
            self.parameters[key] = slider
            self.parameter_editors[key] = editor
            self.parameter_levels[key] = (level, options)
        box.addWidget(label(self.t("audio_configuration_sim"), "muted", True))
        self.config_inputs = []
        for index, key in enumerate(("samplerate", "blocksize", "channels", "ceiling")):
            title = label(self.t(f"audio_{key}"), wrap=True)
            editor = QLineEdit(self.audio_values[index])
            editor.setAccessibleName(title.text())
            title.setBuddy(editor)
            editor.textChanged.connect(
                lambda text, i=index: self.audio_configuration_changed(i, text),
            )
            box.addWidget(title)
            box.addWidget(editor)
            self.config_inputs.append(editor)
        return frame

    def build_monitoring(self) -> QFrame:
        frame, box = panel()
        monitor_row = QHBoxLayout()
        monitor_row.addWidget(label(self.t("monitoring"), "heading"))
        monitor_row.addStretch()
        self.analysis_toggle = QPushButton()
        self.analysis_toggle.setCheckable(True)
        self.analysis_toggle.toggled.connect(self._update_analysis_disclosure)
        monitor_row.addWidget(self.analysis_toggle)
        self.wave_layout_button = QPushButton()
        self.wave_layout_button.setCheckable(True)
        self.wave_layout_button.setChecked(self._wave_side_by_side)
        self.wave_layout_button.toggled.connect(self.set_wave_layout)
        monitor_row.addWidget(self.wave_layout_button)
        self.monitor = self.button(self.t("monitor_start"), self.toggle_monitor, True)
        box.addLayout(monitor_row)
        self.monitor.setToolTip(self.t("monitor_hint"))
        self.input_meter = AudioWave(
            self.t("input_level"), idle_text=self.t("level_idle"), language=self.language,
        )
        self.effect_meter = AudioWave(
            self.t("effect_level"), idle_text=self.t("level_idle"), language=self.language,
        )
        self.wave_comparison = WaveComparison(self.input_meter, self.effect_meter)
        box.addWidget(self.wave_comparison)
        self.set_wave_layout(self._wave_side_by_side)
        return frame

    def build_recording(self) -> QFrame:
        frame, box = panel()
        row = QHBoxLayout()
        row.addWidget(label(self.t("recording"), "heading"))
        row.addStretch()
        self.duration = label("00:00", "current")
        row.addWidget(self.duration)
        self.recording_toggle = QPushButton()
        self.recording_toggle.setText(self.t("show_recording"))
        self.recording_toggle.setAccessibleName(self.t("show_recording"))
        self.recording_toggle.setCheckable(True)
        self.recording_toggle.toggled.connect(self._update_recording_disclosure)
        row.addWidget(self.recording_toggle)
        box.addLayout(row)
        self.recording_content = QWidget()
        content_layout = QVBoxLayout(self.recording_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        self.take_status = label("", "muted", True)
        content_layout.addWidget(self.take_status)
        actions = QGridLayout()
        self.record = self.button(self.t("record_start"), self.toggle_record)
        self.play_original = self.button(self.t("play_original"), lambda: self.play("original"))
        self.play_effect = self.button(self.t("play_effect"), lambda: self.play("effect"))
        self.stop_play = self.button(self.t("stop_playback"), self.stop_playback)
        self.save = self.button(self.t("save_demo"), self.save_demo)
        for index, button in enumerate((self.play_original, self.play_effect, self.stop_play, self.save)):
            actions.addWidget(button, index // 3, index % 3)
        content_layout.addLayout(actions)
        self.play_hint = label("", "muted", True)
        content_layout.addWidget(self.play_hint)
        box.addWidget(self.recording_content)
        frame.setToolTip(self.t("recording_tooltip"))
        return frame

    def build_scenarios(self) -> QFrame:
        frame, box = panel()
        row = QHBoxLayout()
        row.addWidget(label(self.t("scenarios"), "heading"))
        row.addWidget(label(self.t("scenarios_hint"), "muted"))
        row.addStretch()
        box.addLayout(row)
        actions = QGridLayout()
        self.scenario = QComboBox()
        self.scenario.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.scenario.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon,
        )
        self.scenario.setMinimumContentsLength(10)
        self.scenario.setAccessibleName(self.t("scenario_name"))
        self.scenario.addItems([self.t(f"fault_{fault}") for fault in FAULTS])
        actions.addWidget(self.scenario, 0, 0, 1, 2)
        actions.addWidget(self.button(self.t("inject_fault"), self.inject_fault), 1, 0)
        actions.addWidget(self.button(self.t("clear_fault"), self.clear_fault), 1, 1)
        box.addLayout(actions)
        return frame

    def sync_controls(self) -> None:
        s = self.state
        values = [
            (self.input, INPUT_DEVICES.index(s.input_device) + 1 if s.input_device else 0),
            (self.output, OUTPUT_DEVICES.index(s.output_device) + 1 if s.output_device else 0),
        ]
        for control, value in values:
            control.blockSignals(True)
            control.setCurrentIndex(value)
            control.blockSignals(False)
        for editor, slider, value in (
            (self.pitch_editor, self.pitch, s.pitch),
            *((self.parameter_editors[key], slider, getattr(s, key))
              for key, slider in self.parameters.items()),
        ):
            editor.blockSignals(True)
            editor.setValue(value)
            editor.blockSignals(False)
            sync_parameter_slider(slider, value, 1)

    # Actions --------------------------------------------------------------

    def set_wave_layout(self, side_by_side: bool) -> None:
        self._wave_side_by_side = side_by_side
        self.wave_comparison.set_side_by_side(side_by_side)
        key = "layout_stack" if side_by_side else "layout_side"
        short_key = "layout_stack_short" if side_by_side else "layout_side_short"
        action = self.t(key)
        current = self.t("layout_current_side" if side_by_side else "layout_current_stack")
        self.wave_layout_button.setText(
            self.t(short_key) if self._small_layout else action
        )
        self.wave_layout_button.setAccessibleName(action)
        self.wave_layout_button.setAccessibleDescription(current)
        self.wave_layout_button.setToolTip(f"{current} · {action}")

    def _update_responsive_layout(self) -> None:
        small_layout = self.isVisible() and (
            not self.isMaximized() and (self.width() < 800 or self.height() < 600)
        )
        if small_layout == self._small_layout:
            return
        self._small_layout = small_layout
        self._set_disclosure(
            self.analysis_toggle,
            (self.input_meter.analysis_canvas, self.effect_meter.analysis_canvas),
            not small_layout,
            "show_analysis", "hide_analysis", visible=small_layout,
        )
        self._set_disclosure(
            self.device_toggle, self.device_selectors, not small_layout,
            "show_devices", "hide_devices", visible=small_layout,
        )
        self._set_disclosure(
            self.recording_toggle, self.recording_content, not small_layout,
            "show_recording", "hide_recording", visible=small_layout,
        )
        self._update_preset_disclosure(not small_layout, visible=small_layout)
        if small_layout and not self._monitor_in_priority_slot:
            self.content_layout.removeWidget(self.monitoring_panel)
            self.content_layout.insertWidget(
                self.content_layout.indexOf(self.devices_panel),
                self.monitoring_panel,
            )
            self._monitor_in_priority_slot = True
        elif not small_layout and self._monitor_in_priority_slot:
            self.content_layout.removeWidget(self.monitoring_panel)
            self.content_layout.insertWidget(2, self.monitoring_panel)
            self._monitor_in_priority_slot = False
        self.refresh.setText(
            self.t("refresh_short" if small_layout else "refresh")
        )
        self.refresh.setAccessibleName(self.t("refresh"))
        self.set_wave_layout(self._wave_side_by_side)

    def _set_disclosure(
        self, button: QPushButton, content: QWidget | tuple[QWidget, ...],
        expanded: bool, collapsed_key: str, expanded_key: str, *,
        visible: bool = True,
    ) -> None:
        with QSignalBlocker(button):
            button.setChecked(expanded)
        text = self.t(expanded_key if expanded else collapsed_key)
        button.setText(text)
        button.setAccessibleName(text)
        button.setAccessibleDescription(
            self.t("expanded" if expanded else "collapsed")
        )
        button.setVisible(visible)
        widgets = content if isinstance(content, tuple) else (content,)
        for widget in widgets:
            widget.setVisible(expanded)

    def _update_device_disclosure(self, expanded: bool) -> None:
        self._set_disclosure(
            self.device_toggle, self.device_selectors, expanded,
            "show_devices", "hide_devices", visible=self._small_layout is True,
        )

    def _update_preset_disclosure(
        self, expanded: bool, *, visible: bool | None = None,
    ) -> None:
        compact = self._small_layout is True if visible is None else visible
        self._set_disclosure(
            self.preset_toggle, self.preset_options, expanded,
            "show_presets", "hide_presets", visible=compact,
        )
        self.preset_summary.setVisible(compact and not expanded)

    def _update_analysis_disclosure(self, expanded: bool) -> None:
        self._set_disclosure(
            self.analysis_toggle,
            (self.input_meter.analysis_canvas, self.effect_meter.analysis_canvas),
            expanded, "show_analysis", "hide_analysis",
            visible=self._small_layout is True,
        )

    def _update_recording_disclosure(self, expanded: bool) -> None:
        self._set_disclosure(
            self.recording_toggle, self.recording_content, expanded,
            "show_recording", "hide_recording",
            visible=self._small_layout is True,
        )

    def toggle_language(self) -> None:
        self.language = "en" if self.language == "zh" else "zh"
        scenario = self.scenario.currentIndex()
        try:
            save_language(self.language)
        except OSError as error:
            self.say("settings_write_failed", path=str(settings_path()), error=error.strerror or str(error))
        else:
            if self.notice_message[0] in ("settings_invalid", "settings_read_failed", "settings_write_failed"):
                self.say("initial_notice")
        self.build_ui()
        self.scenario.setCurrentIndex(scenario)

    def toggle_scenarios(self) -> None:
        self.scenarios_visible = not self.scenarios_visible
        self.scenarios_panel.setVisible(self.scenarios_visible)

    def run_action(self, action) -> bool:
        try:
            action()
        except ValueError as error:
            self.say("not_executed", reason=Key(error.args[0]))
            self.render()
            return False
        self.render()
        return True

    def device_changed(self, kind: str, index: int) -> None:
        devices = INPUT_DEVICES if kind == "input" else OUTPUT_DEVICES
        if not self.run_action(lambda: self.state.select_device(kind, devices[index - 1] if index else "")):
            self.sync_controls()

    def refresh_devices(self, *, manual: bool = True) -> None:
        if self._refresh_pending or self.state.active or (not manual and not self._startup_refresh_due):
            return
        self._startup_refresh_due = False
        self._refresh_pending = True
        self.discovery_status.setText(self.t("discovering_simulated"))
        self.render()
        QTimer.singleShot(0, self._finish_refresh)

    def _finish_refresh(self) -> None:
        self._refresh_pending = False
        self.discovery_status.setText(self.t("discovered_simulated"))
        self.render()

    def choose_preset(self, key: str) -> None:
        if key == "custom":
            self.state.custom = True
            self.render()
            return
        if self.run_action(lambda: self.state.select_preset(key)):
            self.say("preset_chosen", name=self.preset_name(key))
            self.sync_controls()
            self.render()

    def reset_effect(self) -> None:
        self.state.reset_effect()
        self.sync_controls()
        self.render()

    def pitch_changed(self, value: int) -> None:
        self.state.pitch = value
        self.state.custom = True
        self.render()

    def parameter_changed(self, key: str, value: int) -> None:
        setattr(self.state, key, value)
        if key in ("formant", "robot", "echo", "reverb"):
            self.state.custom = True
        self.render()

    def audio_configuration_changed(self, index: int, text: str) -> None:
        self.audio_values[index] = text

    def toggle_monitor(self) -> None:
        self.run_action(self.state.toggle_monitor)

    def resolve_unsaved(self) -> bool:
        if not self.state.unsaved:
            return True
        dialog = QMessageBox(self)
        dialog.setWindowTitle(self.t("unsaved_title"))
        dialog.setText(self.t("unsaved_text"))
        save = dialog.addButton(self.t("unsaved_save"), QMessageBox.ButtonRole.AcceptRole)
        discard = dialog.addButton(self.t("unsaved_discard"), QMessageBox.ButtonRole.DestructiveRole)
        cancel = dialog.addButton(self.t("cancel"), QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(cancel)
        dialog.setEscapeButton(cancel)
        dialog.exec()
        if dialog.clickedButton() == save:
            return self.save_demo()
        if dialog.clickedButton() == discard:
            self.state.take = False
            self.state.incomplete = False
            return True
        return False

    def toggle_record(self) -> None:
        if self.state.recording:
            self.state.stop_recording()
            self.render()
        elif self.resolve_unsaved():
            self.run_action(self.state.start_recording)

    def play(self, track: str) -> None:
        self.run_action(lambda: self.state.play(track))

    def stop_playback(self) -> None:
        self.state.playing = ""
        self.state.position = 0
        self.render()

    def stop_all(self) -> None:
        self.state.stop_all()
        self.render()

    def save_demo(self) -> bool:
        if not self.state.take or self.state.recording:
            return False
        dialog = QMessageBox(self)
        dialog.setWindowTitle(self.t("save_title"))
        dialog.setText(self.t("save_text"))
        dialog.setInformativeText(self.t("save_info"))
        confirm = dialog.addButton(self.t("save_confirm"), QMessageBox.ButtonRole.AcceptRole)
        fail = dialog.addButton(self.t("save_fail"), QMessageBox.ButtonRole.ActionRole)
        cancel = dialog.addButton(self.t("cancel"), QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(cancel)
        dialog.setEscapeButton(cancel)
        dialog.exec()
        if dialog.clickedButton() == fail:
            self.state.reviewed_save = False
            self.say("save_failed_notice")
            self.render()
            return False
        if dialog.clickedButton() != confirm:
            return False
        self.state.reviewed_save = True
        self.render()
        return True

    def inject_fault(self) -> None:
        scenario = FAULTS[self.scenario.currentIndex()]
        s = self.state
        if scenario == "input_lost":
            dependent = s.capturing
            s.input_device = ""
        elif scenario == "output_lost":
            dependent = s.monitoring or bool(s.playing)
            s.output_device = ""
        elif scenario == "write_failed":
            dependent = s.recording
        else:
            dependent = s.active
        self.sync_controls()
        if dependent:
            s.fail(scenario)
        else:
            self.say("fault_event", scenario=Key(f"fault_{scenario}"))
        self.render()

    def clear_fault(self) -> None:
        self.state.fault = ""
        self.say("fault_cleared")
        self.render()

    def tick(self) -> None:
        self.state.tick()
        self.render()

    # Rendering ------------------------------------------------------------

    @staticmethod
    def availability(button: QPushButton, enabled: bool, reason: str) -> None:
        button.setEnabled(enabled)
        button.setToolTip(reason)
        button.setAccessibleDescription(reason)

    def render(self) -> None:
        s = self.state
        self.tuning_panel.setVisible(s.custom)
        for combo in (self.input, self.output):
            combo.setEnabled(not s.active)
        self.refresh.setEnabled(not s.active and not self._refresh_pending)
        self.device_hint.setText(self.t("devices_locked" if s.active else "devices_unlocked"))
        for key, button in self.presets.items():
            button.setChecked(key == ("custom" if s.custom else s.preset))
        current_preset = self.presets.get(
            "custom" if s.custom else s.preset, self.presets["custom"],
        ).text()
        self.preset_summary.setText(
            f'{self.t("current_preset")}: {current_preset}'
        )
        for editor in self.config_inputs:
            editor.setEnabled(not s.active)
        name = self.text_of(self.preset_name(s.preset))
        self.current.setText(self.t("custom_based", base=name) if s.custom else name)
        self.pitch_value.setText(self.t("pitch_value", value=f"{s.pitch:+d}"))
        for key, (level, options) in self.parameter_levels.items():
            level.setText(self.t(options[getattr(s, key)]))
        self.monitor.setText(self.t("monitor_stop" if s.monitoring else "monitor_start"))
        self.monitor.setAccessibleName(self.t("monitor_stop_name" if s.monitoring else "monitor_start_name"))
        set_visual_state(self.monitor, "active" if s.monitoring else "")
        can_monitor = not s.playing and not s.fault and bool(s.input_device and s.output_device)
        self.availability(self.monitor, s.monitoring or can_monitor, self.t("why_monitor"))
        self.record.setText(self.t("record_stop" if s.recording else "record_start"))
        self.record.setAccessibleName(self.record.text())
        set_visual_state(self.record, "recording" if s.recording else "")
        self.availability(self.record, s.recording or bool(s.input_device and not s.playing and not s.fault),
                          self.t("why_record"))
        can_play = bool(s.take and s.output_device and not s.recording and not s.fault)
        for button in (self.play_original, self.play_effect):
            self.availability(button, can_play, self.t("why_play"))
        self.availability(self.stop_play, bool(s.playing), self.t("why_stop_play"))
        self.availability(self.save, s.take and not s.recording, self.t("why_save"))
        self.stop.setEnabled(s.active)
        set_visual_state(
            self.stop,
            "fault" if s.fault else "active" if s.active else "",
        )
        self.duration.setText(f"{s.duration // 60:02d}:{s.duration % 60:02d}")
        if s.recording:
            status = self.t("take_recording")
        elif s.take:
            status = self.t("take_saved" if s.reviewed_save else "take_unsaved")
            if s.incomplete:
                status += self.t("take_incomplete")
        else:
            status = self.t("take_none")
        self.take_status.setText(status)
        set_visual_state(
            self.take_status,
            "recording" if s.recording else "fault" if s.fault else "",
        )
        self.play_hint.setText(
            self.t("playing_hint", track=Key(f"track_{s.playing}"), pos=s.position, dur=max(1, s.duration))
            if s.playing else self.t("play_rules")
        )
        for meter in (self.input_meter, self.effect_meter):
            meter.set_idle(self.t("level_idle"))
        self.activity.setText(self.t(
            "activity",
            capture=Key("state_demo" if s.capturing else "state_stopped"),
            monitor=Key("state_on" if s.monitoring else "state_off"),
            record=Key("state_demo" if s.recording else "state_stopped"),
            playback=Key(f"track_{s.playing}") if s.playing else Key("state_stopped"),
        ))
        set_visual_state(
            self.activity,
            "recording" if s.recording else
            "active" if s.monitoring else "fault" if s.fault else "",
        )
        if s.fault:
            self.notice.setText(self.t("fault_stopped", scenario=Key(f"fault_{s.fault}")))
            self.notice.show()
        else:
            key, values = self.notice_message
            self.notice.setText(self.t(key, **values))
            self.notice.setVisible(key != "initial_notice")
        warning = bool(s.fault) or self.notice_message[0].startswith("settings_")
        self.notice.setObjectName("warning" if warning else "muted")
        set_visual_state(self.notice, "fault" if warning else "")

    def closeEvent(self, event: QCloseEvent) -> None:
        self.state.stop_all()
        self.render()
        if self.resolve_unsaved():
            self.timer.stop()
            event.accept()
        else:
            event.ignore()


def main() -> int:
    parser = argparse.ArgumentParser(description="静音 UI 原型，不实现任何真实音频功能。 / Silent UI prototype; no real audio.")
    parser.add_argument("--prototype", action="store_true", help="明确启用模拟界面 / explicitly enable the simulated UI")
    args = parser.parse_args()
    if not args.prototype:
        parser.error("必须显式传入 --prototype；尚无真实音频运行模式。 / --prototype is required; there is no real audio mode yet.")
    app = QApplication(sys.argv[:1])
    app.setFont(QFont("Noto Sans CJK SC", 10))
    window = PrototypeWindow()
    screen = app.primaryScreen()
    if screen:
        available = screen.availableGeometry()
        window.resize(min(640, available.width()), min(440, available.height()))
    window.show()
    QTimer.singleShot(0, window.showMaximized)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
