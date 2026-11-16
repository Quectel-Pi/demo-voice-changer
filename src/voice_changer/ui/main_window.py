"""Opt-in desktop listening demo; all device/DSP work stays off the GUI thread."""

from html import escape
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dsp.parameters import EffectParameters

from PySide6.QtCore import QEvent, QSignalBlocker, QTimer, Qt, Slot
from PySide6.QtGui import QCloseEvent, QFontMetrics, QKeySequence, QTextOption
from PySide6.QtWidgets import (
    QComboBox,
    QBoxLayout,
    QFormLayout,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QApplication,
    QLabel,
    QLineEdit,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..app.engine_controller import EngineController
from ..app.device_selection import DeviceKind, DeviceSelectionError
from ..audio.device_discovery import DeviceInfo
from ..audio.device_types import DeviceRef, StreamConfig
from ..audio.engine import EngineError, EngineState, OperationFailure
from ..audio.media_status import MediaStatus, RecordingState, Track
from ..audio import virtual_mic
from ..app.preferences import Preferences, preferences_path
from .audio_wave import AudioWave, WaveComparison
from .theme import SPACING, SIZES, set_visual_state, stylesheet
from .layout import footer_status, new_panel as _new_panel, page_heading
from .i18n import LanguageFilter, load_language, save_language, settings_path
from .interactions import (
    FormInteractions, PageScrollDoubleSpinBox, PageScrollSlider,
    parameter_slider, sync_parameter_slider,
)


_PROCESSED_PEAK_TOOLTIP = (
    "软件处理后的峰值，不代表耳机或扬声器实测值 / "
    "Processed software peak, not a physical output measurement"
)

_LANGUAGE_ISSUES = {
    "settings_read_failed": (
        "无法读取语言设置（{path}）：{error}；已使用中文。",
        "Could not read language setting ({path}): {error}; using Chinese.",
    ),
    "settings_invalid": (
        "语言设置无效（{path}），已使用中文；切换语言后将覆盖。",
        "Invalid language setting ({path}); using Chinese. Switching language overwrites it.",
    ),
    "settings_write_failed": (
        "界面已切换，但语言设置未能保存到 {path}：{error}",
        "Language switched, but the setting could not be saved to {path}: {error}",
    ),
}


class MainWindow(QMainWindow):
    # The DSP demo deliberately supports one fixed format family; keep listen and
    # record from handing other formats to the native DSP and failing deep inside start.
    _DSP_SAMPLERATES = (48000,)
    _DSP_BLOCKSIZES = (256, 512, 1024)
    _DSP_CHANNELS = (1, 2)

    def __init__(self, controller: EngineController | None = None) -> None:
        super().__init__()
        self.controller = controller if controller is not None else EngineController(
            parent=self, preferences_file=preferences_path(),
        )
        self._preferences_applied = False
        self._save_dialog_pending = False
        self._closing_requested = False
        self._allow_close = False
        self._refresh_pending = False
        self._startup_refresh_due = True
        self._refresh_failed = False
        self._previewed_config: StreamConfig | None = None
        self._preparing_config: StreamConfig | None = None
        self._prepared_config: StreamConfig | None = None
        self._displayed_state = self.controller.state
        self._pending_media_action: str | None = None
        self._pending_media_phase: str | None = None
        self._wave_source: tuple[str, int] | None = None
        self._last_output_level_name: str | None = None
        self._last_capture_levels: tuple[float | None, float | None] | None = None
        self._last_playback_peak_sequence: int | None = None
        self._last_analysis_sequence: int | None = None
        self._last_analysis_drops = 0
        self._base_preset = "original"
        self._small_layout: bool | None = None
        self._monitor_in_priority_slot = False
        self._language, language_issue, language_detail = load_language()
        self._language_issue = (
            (language_issue, language_detail) if language_issue else None
        )
        self.setWindowTitle("声场 / Soundstage")
        self.resize(640, 440)
        self.setMinimumSize(480, 400)
        self.setStyleSheet(stylesheet())

        content = QWidget(self)
        content.setObjectName("appRoot")
        outer = QVBoxLayout(content)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        heading = QLabel("声场 / Soundstage")
        heading.setObjectName("appTitle")
        header = QHBoxLayout()
        self.header_layout = header
        header.setContentsMargins(12, 6, 12, 4)
        brand = QVBoxLayout()
        brand.setSpacing(0)
        brand.addWidget(heading)
        header.addLayout(brand)
        self.app_title = heading
        self.app_subtitle = QLabel("Pi M2 声音工作台 / Pi M2 audio workspace")
        self.app_subtitle.setProperty("role", "muted")
        brand.addWidget(self.app_subtitle)
        header.addStretch()
        self._mode = "dsp"
        self.navigation = QWidget()
        self.navigation.setObjectName("workspaceNavigation")
        self.navigation_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, self.navigation)
        self.navigation_layout.setContentsMargins(12, 16, 12, 16)
        self.navigation_layout.setSpacing(6)
        self.navigation_title = QLabel("工作区 / Workspace")
        self.navigation_title.setProperty("role", "muted")
        self.navigation_layout.addWidget(self.navigation_title)
        self.mode_buttons: dict[str, QPushButton] = {}
        self._mode_labels = {
            "dsp": ("DSP 变声 / DSP voice", "DSP 变声 / DSP"),
            "tts": ("AI 朗读 / AI read-aloud", "AI 朗读 / TTS"),
            "vc": ("AI 变声 / AI voice", "AI 变声 / AI VC"),
        }
        for number, (key, (text, _short)) in enumerate(self._mode_labels.items(), start=1):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setProperty("role", "navigation")
            button.setProperty("i18nPreserveShortcut", True)
            button.setShortcut(QKeySequence(f"Ctrl+{number}"))
            button.setToolTip(f"{text} (Ctrl+{number})")
            button.setAccessibleName(f"模式 / Mode: {text}")
            button.clicked.connect(lambda _checked=False, mode=key: self._select_mode(mode))
            self.navigation_layout.addWidget(button)
            self.mode_buttons[key] = button
        self.navigation_layout.addStretch()
        self.navigation_note = QLabel("本机声音工作台 / On-device audio workspace")
        self.navigation_note.setWordWrap(True)
        self.navigation_note.setProperty("role", "muted")
        self.navigation_layout.addWidget(self.navigation_note)
        self.language_button = QPushButton()
        self.language_button.clicked.connect(self._switch_language)
        header.addWidget(self.language_button)
        self.close_button = QPushButton("退出 / Exit")
        self.close_button.clicked.connect(self.close)
        header.addWidget(self.close_button)
        header_widget = QWidget()
        header_widget.setObjectName("workspaceHeader")
        header_widget.setLayout(header)
        self.header_widget = header_widget
        header_widget.setFixedHeight(68)
        outer.addWidget(header_widget)
        self.workspace_shell = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.workspace_shell.setContentsMargins(0, 0, 0, 0)
        self.workspace_shell.setSpacing(0)
        self.workspace_shell.addWidget(self.navigation)
        self.workspace = QWidget()
        workspace_layout = QVBoxLayout(self.workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)
        self.workspace_shell.addWidget(self.workspace, 1)
        outer.addLayout(self.workspace_shell, 1)
        self.mode_stack = QStackedWidget()
        self.mode_stack.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.dsp_page = QWidget()
        dsp_layout = QVBoxLayout(self.dsp_page)
        dsp_layout.setContentsMargins(0, 0, 0, 0)
        dsp_layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.form_interactions = FormInteractions(self.scroll_area)
        body = QWidget()
        body.setObjectName("canvas")
        layout = QVBoxLayout(body)
        self.body_layout = layout
        layout.setContentsMargins(SPACING["lg"], SPACING["lg"], SPACING["lg"], SPACING["lg"])
        layout.setSpacing(SPACING["sm"])
        page_heading(
            layout, "DSP 变声 / DSP voice",
            "选择设备与声音，再开始监听或录音。 / Choose devices and a voice, then monitor or record.",
        )
        self.language_notice = QLabel()
        self.language_notice.setWordWrap(True)
        self.language_notice.setAccessibleName("语言设置状态 / Language settings status")
        set_visual_state(self.language_notice, "warning")
        self.language_notice.hide()
        layout.removeWidget(self.language_notice)
        self.language_notice.setParent(content)
        outer.insertWidget(1, self.language_notice)

        self.audio_status = QLabel(
            "未采集 / No capture"
        )
        self.audio_status.setWordWrap(True)
        self.audio_status.setAccessibleName("采集状态 / Capture status")

        self.virtual_mic_status = QLabel()
        self.virtual_mic_status.setTextFormat(Qt.TextFormat.PlainText)
        self.virtual_mic_status.setWordWrap(True)
        self.virtual_mic_status.setAccessibleName(
            "虚拟麦克风状态 / Virtual microphone status"
        )
        self.monitor_panel, self.monitor_layout = _new_panel()
        monitor_heading = QHBoxLayout()
        monitor_heading.addWidget(QLabel("实时监听 / Monitor"))
        monitor_heading.addStretch()
        self.analysis_toggle = QPushButton()
        self.analysis_toggle.setCheckable(True)
        self.analysis_toggle.toggled.connect(
            lambda expanded: self._set_compact_disclosure(
                self.analysis_toggle,
                (self.input_level.analysis_canvas, self.output_level.analysis_canvas),
                expanded,
                "分析 / Analysis",
                "隐藏分析 / Hide analysis",
            ),
        )
        monitor_heading.addWidget(self.analysis_toggle)
        self.wave_layout_button = QPushButton()
        self.wave_layout_button.setCheckable(True)
        self.wave_layout_button.toggled.connect(self._set_wave_layout)
        monitor_heading.addWidget(self.wave_layout_button)
        self.listen_button = QPushButton("开启监听 / &Listen")
        self.listen_button.setProperty("role", "primary")
        self.listen_button.clicked.connect(self._request_listen)
        self.monitor_layout.addLayout(monitor_heading)
        self.monitor_layout.addWidget(self.audio_status)

        panel = QWidget()
        self.device_panel = panel
        panel.setObjectName("panel")
        panel.setProperty("role", "section")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 10, 12, 10)
        panel_layout.setSpacing(8)
        self.device_status = QLabel()
        self.device_status.setTextFormat(Qt.TextFormat.PlainText)
        self.device_status.setWordWrap(True)
        self.device_status.setProperty("role", "muted")
        device_heading = QHBoxLayout()
        device_heading.setSpacing(8)
        device_heading.addWidget(QLabel("音频设备 / Audio devices"))
        self.device_toggle = QPushButton()
        self.device_toggle.setCheckable(True)
        self.device_toggle.toggled.connect(
            lambda expanded: self._set_compact_disclosure(
                self.device_toggle, self.device_selector_container, expanded,
                "设备 / Devices",
                "隐藏设备 / Hide devices",
            ),
        )
        device_heading.addWidget(self.device_toggle)
        self.refresh_button = QPushButton("刷新设备 / &Refresh devices")
        self.refresh_button.clicked.connect(self._request_refresh)
        device_heading.addStretch()
        device_heading.addWidget(self.refresh_button)
        panel_layout.addLayout(device_heading)

        self.input_combo = QComboBox()
        self.input_combo.setObjectName("inputDeviceCombo")
        self.output_combo = QComboBox()
        self.output_combo.setObjectName("outputDeviceCombo")
        for combo, name in (
            (self.input_combo, "输入设备 / Input device"),
            (self.output_combo, "输出设备 / Output device"),
        ):
            combo.setAccessibleName(name)
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon,
            )
            combo.setMinimumContentsLength(8)
        self.input_label = QLabel("麦克风 / &Mic")
        self.output_label = QLabel("输出 / &Output")
        self.input_label.setBuddy(self.input_combo)
        self.output_label.setBuddy(self.output_combo)
        device_selectors = QGridLayout()
        self.device_selectors = device_selectors
        device_selectors.setHorizontalSpacing(8)
        device_selectors.setVerticalSpacing(4)
        device_selectors.addWidget(self.input_label, 0, 0)
        device_selectors.addWidget(self.input_combo, 0, 1)
        device_selectors.addWidget(self.output_label, 0, 2)
        device_selectors.addWidget(self.output_combo, 0, 3)
        device_selectors.setColumnStretch(1, 1)
        device_selectors.setColumnStretch(3, 1)
        self._device_grid_compact: bool | None = None
        self.device_selector_container = QWidget()
        self.device_selector_container.setLayout(device_selectors)
        panel_layout.addWidget(self.device_selector_container)
        panel_layout.addWidget(self.device_status)
        self.input_combo.activated.connect(lambda index: self._select_device("input", index))
        self.output_combo.activated.connect(lambda index: self._select_device("output", index))
        self.device_details = self._readonly_details("所选设备完整信息 / Selected device details")
        layout.addWidget(panel)
        self.main_flow = QVBoxLayout()
        self.main_flow.setSpacing(8)
        layout.addLayout(self.main_flow)

        self.input_level = AudioWave("麦克风 / Microphone")
        self.output_level = AudioWave("处理后 / Processed")
        self.input_level_label = self.input_level.name_label
        self.output_level_label = self.output_level.name_label
        self.input_level.setToolTip("麦克风输入峰值 / Captured microphone peak")
        self.output_level.setToolTip(_PROCESSED_PEAK_TOOLTIP)
        self.wave_comparison = WaveComparison(self.input_level, self.output_level)
        self.wave_comparison.setObjectName("meterSection")
        self.wave_comparison.setProperty("role", "section")
        self.monitor_layout.addWidget(self.wave_comparison)
        self._set_wave_layout(False)

        self.diagnostics_panel = QWidget()
        self.diagnostics_panel.setObjectName("diagnosticsPanel")
        diagnostics_layout = QVBoxLayout(self.diagnostics_panel)
        diagnostics_layout.setContentsMargins(8, 6, 8, 6)
        diagnostics_layout.setSpacing(6)

        self.device_details_heading = QLabel("设备详情 / Device details")
        self.device_details_heading.setProperty("role", "muted")
        diagnostics_layout.addWidget(self.device_details_heading)
        diagnostics_layout.addWidget(self.device_details)

        self.config_notice = QLabel(
            "采样率、块长等格式字段；校验仅预览，不开启音频。\n"
            "Stream format fields. Validation previews only; no audio starts."
        )
        self.config_notice.setWordWrap(True)
        self.config_notice.setProperty("role", "muted")
        diagnostics_layout.addWidget(self.config_notice)
        self.config_preview = self._readonly_details("未应用配置预览 / Configuration preview; not applied")
        diagnostics_layout.addWidget(self.config_preview)
        config_form = QVBoxLayout()
        config_form.setSpacing(6)
        self.samplerate_input = QLineEdit()
        self.blocksize_input = QLineEdit()
        self.channels_input = QLineEdit()
        self.output_ceiling_input = QLineEdit()
        self.config_inputs = (
            self.samplerate_input, self.blocksize_input, self.channels_input,
            self.output_ceiling_input,
        )
        for editor, name, value in zip(self.config_inputs, (
            "采样率 (Hz) / &Sample rate (Hz)",
            "块长 (帧) / &Block size (frames)",
            "双端通道数 / &Channels (input/output)",
            "输出峰值上限 (0, 1] / Output &ceiling (0, 1]",
        ), ("48000", "1024", "1", "0.05")):
            label = QLabel(name)
            label.setWordWrap(True)
            label.setBuddy(editor)
            editor.setAccessibleName(name.replace("&", ""))
            editor.setPlaceholderText("必填 / Required")
            editor.setText(value)
            editor.textChanged.connect(self._clear_config_preview)
            config_form.addWidget(label)
            config_form.addWidget(editor)
        diagnostics_layout.addLayout(config_form)
        self.validate_button = QPushButton("校验配置（不应用） / &Validate (not apply)")
        self.validate_button.clicked.connect(self._request_stream_config)
        diagnostics_layout.addWidget(self.validate_button)
        self.prepare_backend_button = QPushButton(
            "诊断：准备后端（不开流） / Diagnostic: &prepare only"
        )
        self.prepare_backend_button.setEnabled(False)
        self.prepare_backend_button.clicked.connect(self._request_backend_prepare)
        diagnostics_layout.addWidget(
            self.prepare_backend_button, alignment=Qt.AlignmentFlag.AlignRight,
        )
        self.backend_prepare_status = QLabel()
        self.backend_prepare_status.setTextFormat(Qt.TextFormat.PlainText)
        self.backend_prepare_status.setWordWrap(True)
        self.backend_prepare_status.setAccessibleName(
            "后端准备状态 / Backend preparation status"
        )
        self.backend_prepare_status.hide()
        diagnostics_layout.addWidget(self.backend_prepare_status)
        diagnostics_layout.addWidget(self.virtual_mic_status)
        self.preset_combo = QComboBox()
        for label, key in (
            ("原声 / Original", "original"),
            ("男变女 / Higher voice", "male_to_female"),
            ("女变男 / Lower voice", "female_to_male"),
            ("机器人 / Robot", "robot"),
            ("回声 / Echo", "echo"),
            ("短混响 / Short reverb", "reverb"),
            ("萝莉 / Childlike", "loli"),
            ("大叔 / Deep voice", "uncle"),
            ("怪兽 / Monster", "monster"),
            ("自定义（基于原声） / Custom (Original)", "custom"),
        ):
            self.preset_combo.addItem(label, key)
        self.preset_combo.setParent(content)
        self.preset_combo.setAccessibleName("选择声音预设 / Choose a voice preset")
        self.preset_combo.hide()
        self.preset_panel, preset_layout = _new_panel()
        preset_heading = QLabel("声音预设 / Voice presets")
        preset_heading.setObjectName("sectionTitle")
        preset_header = QHBoxLayout()
        preset_header.addWidget(preset_heading)
        preset_header.addStretch()
        self.preset_summary = QLabel()
        self.preset_summary.setProperty("role", "muted")
        self.preset_summary.setAccessibleName("当前声音预设 / Current voice preset")
        self.preset_toggle = QPushButton()
        self.preset_toggle.setCheckable(True)
        self.preset_toggle.toggled.connect(self._update_preset_disclosure)
        preset_header.addWidget(self.preset_toggle)
        preset_layout.addLayout(preset_header)
        preset_layout.addWidget(self.preset_summary)
        preset_grid = QGridLayout()
        self.preset_grid = preset_grid
        self._preset_columns: int | None = None
        preset_grid.setHorizontalSpacing(7)
        preset_grid.setVerticalSpacing(7)
        self.preset_buttons: dict[str, QPushButton] = {}
        for index, (key, title) in enumerate((
            ("original", "原声 / Original"),
            ("male_to_female", "男变女 / Higher"),
            ("female_to_male", "女变男 / Lower"),
            ("robot", "机器人 / Robot"),
            ("echo", "回声 / Echo"),
            ("reverb", "短混响 / Reverb"),
            ("loli", "萝莉 / Childlike"),
            ("uncle", "大叔 / Deep voice"),
            ("monster", "怪兽 / Monster"),
            ("custom", "自定义 / Custom"),
        )):
            button = QPushButton(title)
            button.setCheckable(True)
            button.setProperty("role", "preset")
            button.setAccessibleName(f"选择音色 / Select voice: {title}")
            button.clicked.connect(
                lambda checked=False, preset=key: self._choose_preset_button(preset),
            )
            preset_grid.addWidget(button, index // 2, index % 2)
            self.preset_buttons[key] = button
        preset_grid.setColumnStretch(0, 1)
        preset_grid.setColumnStretch(1, 1)
        self.preset_options = QWidget()
        self.preset_options.setLayout(preset_grid)
        preset_layout.addWidget(self.preset_options)
        self.preset_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum,
        )

        self.pitch_input = PageScrollDoubleSpinBox()
        self.pitch_input.setRange(-12, 12)
        self.pitch_input.setDecimals(3)
        self.pitch_input.setSingleStep(0.1)
        self.pitch_input.setKeyboardTracking(False)
        self.pitch_input.setSuffix(" st")
        self.pitch_input.setAccessibleName("音高 / Pitch")
        self.formant_input = PageScrollDoubleSpinBox()
        self.formant_input.setRange(0.8, 1.25)
        self.formant_input.setDecimals(2)
        self.formant_input.setSingleStep(0.01)
        self.formant_input.setValue(1.0)
        self.formant_input.setSuffix(" ×")
        self.robot_input = PageScrollDoubleSpinBox()
        self.echo_input = PageScrollDoubleSpinBox()
        self.reverb_input = PageScrollDoubleSpinBox()
        for control, maximum in (
            (self.robot_input, 100), (self.echo_input, 40), (self.reverb_input, 40),
        ):
            control.setRange(0, maximum)
            control.setDecimals(0)
            control.setSingleStep(1)
            control.setSuffix(" %")
        self.input_gain_input = PageScrollDoubleSpinBox()
        self.output_gain_input = PageScrollDoubleSpinBox()
        self._gain_inputs = (self.input_gain_input, self.output_gain_input)
        for control in self._gain_inputs:
            control.setRange(-24, 12)
            control.setDecimals(1)
            control.setSingleStep(0.5)
            control.setSuffix(" dB")
        self._sound_inputs = (
            self.pitch_input, self.formant_input, self.robot_input,
            self.echo_input, self.reverb_input,
        )
        self.denoise_input = PageScrollDoubleSpinBox()
        self.denoise_input.setRange(0, 100)
        self.denoise_input.setDecimals(0)
        self.denoise_input.setSingleStep(1)
        self.denoise_input.setSuffix(" %")
        self.denoise_input.setAccessibleName("降噪强度 / Noise reduction strength")
        self.parameter_sliders: dict[PageScrollDoubleSpinBox, PageScrollSlider] = {}
        self._slider_scales = {
            self.pitch_input: 10, self.formant_input: 100,
            self.robot_input: 1, self.echo_input: 1, self.reverb_input: 1,
            self.input_gain_input: 10, self.output_gain_input: 10,
            self.denoise_input: 1,
        }

        def slider_row(editor: PageScrollDoubleSpinBox) -> QWidget:
            slider = parameter_slider(
                editor, self._slider_scales[editor],
                5 if editor in self._gain_inputs else 1,
            )
            self.parameter_sliders[editor] = slider
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(slider, 1)
            layout.addWidget(editor)
            return row

        effects_heading = QLabel("自定义声音 / Custom voice")
        effects_heading.setObjectName("sectionTitle")
        self.preset_label = QLabel("音色 / Voice")
        self.preset_label.setBuddy(self.preset_combo)
        effects_form = QFormLayout()
        effects_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        effects_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        effects_form.setVerticalSpacing(6)
        pitch_row = slider_row(self.pitch_input)
        self.pitch_slider = self.parameter_sliders[self.pitch_input]
        pitch_label = QLabel("音高 / &Pitch")
        pitch_label.setBuddy(self.pitch_slider)
        effects_form.addRow(pitch_label, pitch_row)
        for title, control in (
            ("共振峰比例 / &Formant scale", self.formant_input),
            ("机器人强度（80Hz） / Robot mix", self.robot_input),
            ("回声混合（200ms） / Echo mix", self.echo_input),
            ("短混响混合 / Reverb mix", self.reverb_input),
        ):
            label = QLabel(title)
            label.setWordWrap(True)
            control.setAccessibleName(title.replace("&", ""))
            row = slider_row(control)
            label.setBuddy(self.parameter_sliders[control])
            effects_form.addRow(label, row)
        gain_form = QFormLayout()
        gain_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        gain_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        gain_form.setVerticalSpacing(6)
        for title, control in (
            ("输入音量 / Input level", self.input_gain_input),
            ("输出音量 / Output level", self.output_gain_input),
        ):
            label = QLabel(title)
            control.setAccessibleName(title)
            row = slider_row(control)
            label.setBuddy(self.parameter_sliders[control])
            gain_form.addRow(label, row)
        denoise_form = QFormLayout()
        denoise_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        denoise_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        denoise_form.setVerticalSpacing(6)
        denoise_label = QLabel("降噪强度 / &Noise reduction")
        denoise_row = slider_row(self.denoise_input)
        denoise_label.setBuddy(self.parameter_sliders[self.denoise_input])
        denoise_form.addRow(denoise_label, denoise_row)
        self.parameter_forms = (effects_form, gain_form, denoise_form)
        for form in self.parameter_forms:
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.tuning_panel, tuning_layout = _new_panel()
        self.tuning_panel.setObjectName("panel")
        tuning_layout.addWidget(effects_heading)
        self.tuning_controls = QWidget()
        controls_layout = QVBoxLayout(self.tuning_controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(7)
        controls_layout.addLayout(effects_form)
        controls_layout.addLayout(gain_form)
        tuning_layout.addWidget(self.tuning_controls)
        tuning_layout.addWidget(self.diagnostics_panel)
        denoise_heading = QLabel("降噪 / Noise reduction")
        denoise_heading.setObjectName("sectionTitle")
        self.denoise_panel, denoise_layout = _new_panel()
        self.denoise_panel.setObjectName("panel")
        denoise_layout.addWidget(denoise_heading)
        denoise_layout.addLayout(denoise_form)
        denoise_hint = QLabel(
            "前置频谱门限降噪，所有音色均可用；0% 关闭。录音保留未降噪原声；"
            "启用后会增加约 5 ms 处理延迟。\n"
            "Pre-DSP spectral gate for every voice; 0% is off. Recordings keep the "
            "unprocessed original; enabling it adds about 5 ms of latency."
        )
        denoise_hint.setWordWrap(True)
        denoise_hint.setProperty("role", "muted")
        denoise_layout.addWidget(denoise_hint)
        self.preset_combo.activated.connect(self._select_preset)
        for control in self._sound_inputs:
            control.valueChanged.connect(self._edit_effect)
        for control in self._gain_inputs:
            control.valueChanged.connect(self._submit_effects)
        self.denoise_input.valueChanged.connect(self._submit_effects)
        self.reset_effect_button = QPushButton("重置效果 / Rese&t effect")
        self.reset_effect_button.setAccessibleName("重置基准效果，保留增益 / Reset base effect; keep gains")
        self.reset_effect_button.setToolTip(
            "恢复基准预设，不改变增益、设备或活动。\nRestore the base preset; keep gains, devices and activity."
        )
        self.reset_effect_button.clicked.connect(self._reset_effect)
        controls_layout.addWidget(self.reset_effect_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.effect_status = QLabel("尚未应用；启动采集后生效。\nNot applied; start capture to apply.")
        self.effect_status.setWordWrap(True)
        controls_layout.addWidget(self.effect_status)
        self.notice = QLabel(
            "默认原声；监听与录音独立，点击相应操作后才采集。\n"
            "Starts in Original; click Listen or Start recording to begin capture."
        )
        self.notice.setWordWrap(True)
        self.notice.setProperty("role", "muted")
        self.monitor_layout.addWidget(self.notice)

        self.activity_status = QLabel("已停止 / Stopped")
        self.activity_status.setWordWrap(True)
        recording_heading = QLabel("录音与试听 / Recording")
        recording_heading.setObjectName("sectionTitle")
        self.recording_heading = recording_heading
        self.recording_panel, recording_layout = _new_panel()
        recording_header = QHBoxLayout()
        recording_header.addWidget(recording_heading)
        recording_header.addStretch()
        self.recording_toggle = QPushButton()
        self.recording_toggle.setCheckable(True)
        self.recording_toggle.toggled.connect(
            lambda expanded: self._set_compact_disclosure(
                self.recording_toggle, self.recording_content, expanded,
                "录音/试听 / Recording",
                "隐藏录音 / Hide recording",
            ),
        )
        recording_header.addWidget(self.recording_toggle)
        self.record_button = QPushButton("开始录音 / &Start recording")
        self.record_button.clicked.connect(self._request_record)
        recording_layout.addLayout(recording_header)
        self.recording_content = QWidget()
        recording_content_layout = QVBoxLayout(self.recording_content)
        recording_content_layout.setContentsMargins(0, 0, 0, 0)
        self.recording_details = QLabel("未录音 / No recording")
        self.recording_details.setWordWrap(True)
        self.recording_details.setTextFormat(Qt.TextFormat.PlainText)
        recording_content_layout.addWidget(self.recording_details)
        playback_actions = QHBoxLayout()
        self.play_original_button = QPushButton("试听原声 / Play &original")
        self.play_effect_button = QPushButton("试听效果 / Play &effect")
        self.play_original_button.clicked.connect(lambda: self._request_play("original"))
        self.play_effect_button.clicked.connect(lambda: self._request_play("effect"))
        playback_actions.addWidget(self.play_original_button)
        playback_actions.addWidget(self.play_effect_button)
        recording_content_layout.addLayout(playback_actions)
        file_actions = QHBoxLayout()
        self.save_recording_button = QPushButton("保存录音组 / &Save group")
        self.discard_recording_button = QPushButton("丢弃未保存录音 / Discard")
        self.save_recording_button.clicked.connect(self._save_recording)
        self.discard_recording_button.clicked.connect(lambda: self._guard_recording("discard"))
        file_actions.addWidget(self.save_recording_button)
        file_actions.addWidget(self.discard_recording_button)
        recording_content_layout.addLayout(file_actions)
        recovery_actions = QHBoxLayout()
        self.recover_recording_button = QPushButton("恢复遗留录音 / Recover")
        self.delete_recovery_button = QPushButton("删除遗留录音 / Delete leftover")
        self.recover_recording_button.clicked.connect(lambda: self._choose_recovery(False))
        self.delete_recovery_button.clicked.connect(lambda: self._choose_recovery(True))
        recovery_actions.addWidget(self.recover_recording_button)
        recovery_actions.addWidget(self.delete_recovery_button)
        recording_content_layout.addLayout(recovery_actions)
        recording_layout.addWidget(self.recording_content)
        for control in (
            self.play_original_button, self.play_effect_button,
            self.save_recording_button, self.discard_recording_button,
            self.recover_recording_button, self.delete_recovery_button,
            self.recording_details,
        ):
            control.hide()
        self.error_notice = QLabel()
        self.error_notice.setTextFormat(Qt.TextFormat.PlainText)
        self.error_notice.setWordWrap(True)
        self.error_notice.setMinimumWidth(0)
        self.error_notice.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred,
        )
        self.error_notice.setAccessibleName("操作错误 / Operation error")
        self.error_notice.hide()
        layout.insertWidget(0, self.error_notice)
        for section in (
            self.preset_panel, self.denoise_panel, self.monitor_panel,
            self.tuning_panel, self.recording_panel,
        ):
            section.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
            self.main_flow.addWidget(section)
        self.monitor_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred,
        )
        layout.addStretch(1)
        self.scroll_area.setWidget(body)
        dsp_layout.addWidget(self.scroll_area, 1)

        footer = QWidget()
        footer.setObjectName("fixedFooter")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(16, 8, 16, 8)
        footer_layout.setSpacing(4)
        self.activity_status.setObjectName("activityStatus")
        self.recording_status = QLabel()
        self.recording_status.setObjectName("recordingStatus")
        self.recording_status.setWordWrap(True)
        self.recording_status.setTextFormat(Qt.TextFormat.PlainText)
        self.recording_status.setAccessibleName("录音与回放状态 / Recording and playback status")
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        status_row.addWidget(self.activity_status, 1)
        status_row.addWidget(self.recording_status, 1)
        status_layout = footer_status(footer_layout)
        status_layout.addLayout(status_row)
        self.listen_guidance = QLabel()
        self.listen_guidance.setTextFormat(Qt.TextFormat.PlainText)
        self.listen_guidance.setWordWrap(True)
        self.listen_guidance.setAccessibleName("监听操作提示 / Listening steps")
        self.listen_guidance.hide()
        status_layout.addWidget(self.listen_guidance)
        audio_actions = QHBoxLayout()
        audio_actions.setSpacing(8)
        audio_actions.addWidget(self.listen_button, 1)
        audio_actions.addWidget(self.record_button, 1)
        self.stop_button = QPushButton("停止 / &Stop")
        self.stop_button.setMinimumWidth(104)
        self.stop_button.clicked.connect(self._request_stop)
        audio_actions.addStretch()
        audio_actions.addWidget(self.stop_button)
        footer_layout.addLayout(audio_actions)
        footer.setMinimumHeight(SIZES["footerMinHeight"])
        self.retry_button = QPushButton("重试退出 / Retry exit")
        self.retry_button.clicked.connect(self._begin_shutdown)
        self.retry_button.hide()
        header.addWidget(self.retry_button)
        dsp_layout.addWidget(footer)
        dsp_layout.removeWidget(footer)
        footer.setParent(None)
        self.footer_stack = QStackedWidget()
        self.footer_stack.setMinimumHeight(SIZES["footerMinHeight"])
        self.footer_stack.addWidget(footer)
        self._mode_footers = {"dsp": footer}
        # AI pages are built on first use so the P0 DSP startup never imports AI code.
        self.tts_page = None
        self.vc_page = None
        self.mode_stack.addWidget(self.dsp_page)
        workspace_layout.addWidget(self.mode_stack, 1)
        workspace_layout.addWidget(self.footer_stack)
        self._sync_mode_buttons()
        self.setCentralWidget(content)
        self._language_filter = LanguageFilter(self, lambda: self._language)
        QApplication.instance().installEventFilter(self._language_filter)

        self.controller.stateChanged.connect(self._on_state)
        self.controller.errorOccurred.connect(self._on_error)
        self.controller.shutdownFailed.connect(self._on_shutdown_failed)
        self.controller.shutdownComplete.connect(self._on_shutdown_complete)
        self.controller.deviceSelectionChanged.connect(self._update_device_controls)
        self.controller.streamConfigReady.connect(self._on_stream_config_ready)
        self.controller.backendPrepared.connect(self._on_backend_prepared)
        self.controller.operationFinished.connect(self._on_operation_finished)
        self.controller.ready.connect(self._maybe_startup_refresh)
        self.controller.liveStatus.connect(self._on_live_status)
        self.controller.mediaStatus.connect(self._on_media_status)
        self.controller.preferencesChanged.connect(self._on_preferences)
        self.controller.saveDirectoryReady.connect(self._on_save_directory_ready)
        self._feedback_scroll_timer = QTimer(self)
        self._feedback_scroll_timer.setSingleShot(True)
        self._feedback_scroll_timer.setInterval(50)
        self._feedback_scroll_timer.timeout.connect(self._reveal_feedback)
        self._on_state(self.controller.state)
        self._update_device_controls()
        self._on_media_status(self.controller.media_status)
        self._sync_preset_buttons()
        self._update_responsive_layout()
        self._apply_language()
        self._refresh_language_notice()

    @property
    def mode(self) -> str:
        return self._mode

    def _sync_mode_buttons(self) -> None:
        for key, button in self.mode_buttons.items():
            button.setChecked(key == self._mode)

    def _switch_language(self) -> None:
        self._language = "en" if self._language == "zh" else "zh"
        self._apply_language()
        try:
            save_language(self._language)
        except OSError as error:
            self._language_issue = ("settings_write_failed", error.strerror or str(error))
        else:
            self._language_issue = None
        self._refresh_language_notice()

    def _apply_language(self) -> None:
        target = "English" if self._language == "zh" else "中文"
        if self._language == "zh":
            name = "切换界面语言为英文"
            current = "当前界面语言：中文"
        else:
            name = "Switch interface language to Chinese"
            current = "Current interface language: English"
        self.language_button.setText(target)
        self.language_button.setAccessibleName(name)
        self.language_button.setAccessibleDescription(current)
        self.language_button.setToolTip(f"{name} · {current}")
        for meter in (self.input_level, self.output_level):
            meter.analysis_canvas.set_language(self._language)
        self._language_filter.localize_tree()
        self._render_device_details()
        if self._previewed_config is not None:
            self._render_config_preview(self._previewed_config)
        self._set_navigation_widths(self.width() < SIZES["sidebarBreakpoint"])
        self._refresh_language_notice()

    def _refresh_language_notice(self) -> None:
        if self._language_issue is None:
            self.language_notice.clear()
            self.language_notice.hide()
            return
        key, detail = self._language_issue
        template = _LANGUAGE_ISSUES[key][0 if self._language == "zh" else 1]
        self.language_notice.setText(template.format(path=str(settings_path()), error=detail))
        self.language_notice.show()
        self._language_filter.localize_widget(self.language_notice)

    def _set_navigation_widths(self, compact: bool) -> None:
        for key, button in self.mode_buttons.items():
            if not compact:
                button.setMinimumWidth(0)
                button.setMaximumWidth(16777215)
                continue
            source = self._mode_labels[key][1 if compact else 0]
            local, translated = source.split(" / ", 1)
            metrics = QFontMetrics(button.font())
            button.setFixedWidth(max(
                metrics.horizontalAdvance(local), metrics.horizontalAdvance(translated),
            ) + 24)
        metrics = QFontMetrics(self.language_button.font())
        self.language_button.setFixedWidth(
            max(metrics.horizontalAdvance("English"), metrics.horizontalAdvance("中文")) + 24
        )
        metrics = QFontMetrics(self.close_button.font())
        self.close_button.setFixedWidth(
            max(metrics.horizontalAdvance("退出"), metrics.horizontalAdvance("Exit")) + 24
        )

    def _dsp_audio_active(self) -> bool:
        controller = self.controller
        return (
            (self.tts_page is not None and self.tts_page.busy)
            or controller.busy or controller.owns_devices
            or controller.state in (EngineState.RUNNING, EngineState.STARTING, EngineState.STOPPING)
        )

    def _select_mode(self, mode: str) -> bool:
        if mode == self._mode or self._closing_requested:
            self._sync_mode_buttons()
            return mode == self._mode
        if self._dsp_audio_active():
            self._sync_mode_buttons()
            if self.tts_page is not None and self.tts_page.busy:
                self._on_error(EngineError(
                    "请先停止朗读生成或等待保存完成，再切换模式 / "
                    "Stop speech generation or wait for saving before switching modes"
                ))
                return False
            if self.controller.busy and not self.controller.owns_devices:
                self._on_error(EngineError(
                    "正在处理设备或文件操作，请稍候再切换模式 / "
                    "A device or file operation is in progress; switch modes when it finishes."
                ))
                return False
            self._on_error(EngineError(
                "请先停止 DSP / AI 监听、录音或回放，再切换模式；不会自动停止。 / "
                "Stop DSP / AI listening, recording or playback before switching modes; nothing is stopped automatically."
            ))
            return False
        if mode != "dsp":
            self._ensure_ai_pages()
        self._mode = mode
        self.mode_stack.setCurrentWidget(
            {"dsp": self.dsp_page, "tts": self.tts_page, "vc": self.vc_page}[mode],
        )
        self.footer_stack.setCurrentWidget(self._mode_footers[mode])
        self._sync_mode_buttons()
        self._language_filter.localize_tree()
        return True

    def _ensure_ai_pages(self) -> None:
        if self.tts_page is not None:
            return
        from .ai_pages import AiTtsPage, AiVcPage

        self.tts_page = AiTtsPage(
            controller=self.controller,
            output_ceiling=lambda: float(self.output_ceiling_input.text()),
        )
        self._tts_close_pending = False
        self.tts_page.idle.connect(self._tts_idle)
        self.vc_page = AiVcPage(self.controller)
        for page in (self.tts_page, self.vc_page):
            page_footer = page.findChild(QWidget, "fixedFooter")
            if page_footer is not None:
                page.layout().removeWidget(page_footer)
                page_footer.setParent(None)
                page_footer.setMinimumHeight(SIZES["footerMinHeight"])
                self.footer_stack.addWidget(page_footer)
                self._mode_footers["tts" if page is self.tts_page else "vc"] = page_footer
        self.tts_page.devicesRequested.connect(self._select_audio_devices)
        self.vc_page.devicesRequested.connect(self._select_audio_devices)
        self.mode_stack.addWidget(self.tts_page)
        self.mode_stack.addWidget(self.vc_page)
        self._language_filter.localize_tree()

    @Slot()
    def _select_audio_devices(self) -> None:
        if not self._select_mode("dsp"):
            return
        if self._small_layout and not self.device_toggle.isChecked():
            self.device_toggle.click()
        QTimer.singleShot(0, lambda: self.scroll_area.ensureWidgetVisible(self.input_combo))
        self.input_combo.setFocus(Qt.FocusReason.OtherFocusReason)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._update_responsive_layout()
        QTimer.singleShot(0, self._maybe_startup_refresh)

    def _set_wave_layout(self, side_by_side: bool) -> None:
        self.wave_comparison.set_side_by_side(side_by_side)
        full_action = "切换上下 / Stack" if side_by_side else "切换左右 / Side by side"
        action = (
            ("上下 / Stack" if side_by_side else "左右 / Side")
            if self._small_layout else full_action
        )
        current = "当前左右排列 / Currently side by side" if side_by_side else "当前上下排列 / Currently stacked"
        self.wave_layout_button.setText(action)
        self.wave_layout_button.setAccessibleName(full_action)
        self.wave_layout_button.setAccessibleDescription(current)
        self.wave_layout_button.setToolTip(f"{current} · {action}")

    @Slot()
    def _maybe_startup_refresh(self) -> None:
        if (
            self._startup_refresh_due and self.isVisible()
            and self.controller.is_ready and not self.controller.busy
            and not self._closing_requested and not self.controller.device_selection.locked
        ):
            self._request_refresh(clear_error=False)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_responsive_layout()

    def _update_responsive_layout(self) -> None:
        sidebar = self.width() >= SIZES["sidebarBreakpoint"]
        self.workspace_shell.setDirection(
            QBoxLayout.Direction.LeftToRight if sidebar else QBoxLayout.Direction.TopToBottom,
        )
        self.navigation_layout.setDirection(
            QBoxLayout.Direction.TopToBottom if sidebar else QBoxLayout.Direction.LeftToRight,
        )
        if sidebar and self.workspace_shell.indexOf(self.navigation) < 0:
            self.header_layout.removeWidget(self.navigation)
            self.workspace_shell.insertWidget(0, self.navigation)
        elif not sidebar and self.header_layout.indexOf(self.navigation) < 0:
            self.workspace_shell.removeWidget(self.navigation)
            self.header_layout.insertWidget(1, self.navigation, 1)
        self.navigation.setMinimumWidth(SIZES["sidebarWidth"] if sidebar else 0)
        self.navigation.setMaximumWidth(SIZES["sidebarWidth"] if sidebar else 16777215)
        self.navigation.setMinimumHeight(0)
        self.navigation.setMaximumHeight(16777215 if sidebar else 52)
        self.navigation_layout.setContentsMargins(
            12 if sidebar else 0, 16 if sidebar else 0,
            12 if sidebar else 0, 16 if sidebar else 0,
        )
        self.navigation_title.setVisible(sidebar)
        self.navigation_note.setVisible(sidebar)
        self.app_subtitle.setVisible(sidebar)
        self.app_title.setVisible(sidebar)
        self.header_widget.setFixedHeight(68 if sidebar else 52)
        self.footer_stack.setFixedHeight(
            SIZES["footerHeight"] if sidebar else SIZES["compactFooterHeight"],
        )
        self._set_navigation_widths(not sidebar)
        for number, (key, button) in enumerate(self.mode_buttons.items(), start=1):
            full, short = self._mode_labels[key]
            button.setText(full if sidebar else short)
            button.setShortcut(QKeySequence(f"Ctrl+{number}"))
        content_width = self.width() - (SIZES["sidebarWidth"] if sidebar else 0)
        inline_parameters = content_width >= 600
        for form in self.parameter_forms:
            form.setRowWrapPolicy(
                QFormLayout.RowWrapPolicy.DontWrapRows if inline_parameters
                else QFormLayout.RowWrapPolicy.WrapAllRows,
            )
            for index in range(form.rowCount()):
                label = form.itemAt(index, QFormLayout.ItemRole.LabelRole).widget()
                label.setFixedWidth(160 if inline_parameters else 0)
                if not inline_parameters:
                    label.setMaximumWidth(16777215)
        preset_columns = 5 if content_width >= 900 else 3 if content_width >= 600 else 2
        if preset_columns != self._preset_columns:
            while self.preset_grid.count():
                self.preset_grid.takeAt(0)
            for index, button in enumerate(self.preset_buttons.values()):
                self.preset_grid.addWidget(
                    button, index // preset_columns, index % preset_columns,
                )
            for column in range(5):
                self.preset_grid.setColumnStretch(column, int(column < preset_columns))
            self._preset_columns = preset_columns

        device_compact = content_width < 520
        if device_compact != self._device_grid_compact:
            while self.device_selectors.count():
                self.device_selectors.takeAt(0)
            for column in range(4):
                self.device_selectors.setColumnStretch(column, 0)
            if device_compact:
                self.device_selectors.addWidget(self.input_label, 0, 0)
                self.device_selectors.addWidget(self.input_combo, 0, 1)
                self.device_selectors.addWidget(self.output_label, 1, 0)
                self.device_selectors.addWidget(self.output_combo, 1, 1)
                self.device_selectors.setColumnStretch(1, 1)
            else:
                self.device_selectors.addWidget(self.input_label, 0, 0)
                self.device_selectors.addWidget(self.input_combo, 0, 1)
                self.device_selectors.addWidget(self.output_label, 0, 2)
                self.device_selectors.addWidget(self.output_combo, 0, 3)
                self.device_selectors.setColumnStretch(1, 1)
                self.device_selectors.setColumnStretch(3, 1)
            self._device_grid_compact = device_compact

        small_layout = self.isVisible() and (
            not self.isMaximized() and (self.width() < 800 or self.height() < 600)
        )
        if small_layout == self._small_layout:
            return
        self._small_layout = small_layout
        self.body_layout.setContentsMargins(16, 2 if small_layout else 16, 16, 8)
        self.dsp_page.findChild(QWidget, "pageHeading").setVisible(not small_layout)
        self.monitor_layout.setContentsMargins(12, 8 if small_layout else 12, 12, 8 if small_layout else 12)
        inset = SPACING["md"] if small_layout else SPACING["lg"]
        self.preset_panel.layout().setContentsMargins(inset, 12, inset, 12)
        self._set_navigation_widths(not sidebar)
        self._set_compact_disclosure(
            self.analysis_toggle,
            (self.input_level.analysis_canvas, self.output_level.analysis_canvas),
            not small_layout,
            "分析 / Analysis",
            "隐藏分析 / Hide analysis",
            visible=small_layout,
        )
        self._set_compact_disclosure(
            self.device_toggle, self.device_selector_container, not small_layout,
            "设备 / Devices",
            "隐藏设备 / Hide devices",
            visible=small_layout,
        )
        self._set_compact_disclosure(
            self.recording_toggle, self.recording_content, not small_layout,
            "录音/试听 / Recording",
            "隐藏录音 / Hide recording",
            visible=small_layout,
        )
        self._update_preset_disclosure(not small_layout, visible=small_layout)
        if small_layout and not self._monitor_in_priority_slot:
            self.main_flow.removeWidget(self.monitor_panel)
            self.body_layout.insertWidget(
                self.body_layout.indexOf(self.device_panel), self.monitor_panel,
            )
            self._monitor_in_priority_slot = True
        elif not small_layout and self._monitor_in_priority_slot:
            self.body_layout.removeWidget(self.monitor_panel)
            # The denoise panel sits between the preset and monitor panels, so the
            # monitor must return to index 2 rather than the pre-denoise slot.
            self.main_flow.insertWidget(2, self.monitor_panel)
            self._monitor_in_priority_slot = False
        refresh_text = (
            "刷新 / Refresh" if small_layout
            else "刷新设备 / &Refresh devices"
        )
        self.refresh_button.setText(refresh_text)
        self.refresh_button.setAccessibleName("刷新设备 / Refresh devices")
        self._set_wave_layout(self.wave_comparison.side_by_side)

    def _set_compact_disclosure(
        self, button: QPushButton, content: QWidget | tuple[QWidget, ...], expanded: bool,
        collapsed_text: str, expanded_text: str, *, visible: bool = True,
    ) -> None:
        with QSignalBlocker(button):
            button.setChecked(expanded)
        button.setText(expanded_text if expanded else collapsed_text)
        button.setAccessibleName(button.text())
        button.setAccessibleDescription(
            "已展开 / Expanded" if expanded else "已收起 / Collapsed"
        )
        button.setVisible(visible)
        widgets = content if isinstance(content, tuple) else (content,)
        for widget in widgets:
            widget.setVisible(expanded)

    def _update_preset_disclosure(
        self, expanded: bool, *, visible: bool | None = None,
    ) -> None:
        compact = self._small_layout if visible is None else visible
        with QSignalBlocker(self.preset_toggle):
            self.preset_toggle.setChecked(expanded)
        self.preset_toggle.setText(
            "隐藏预设 / Hide presets" if expanded else "预设 / Presets"
        )
        self.preset_toggle.setAccessibleName(self.preset_toggle.text())
        self.preset_toggle.setAccessibleDescription(
            "已展开 / Expanded" if expanded else "已收起 / Collapsed"
        )
        self.preset_toggle.setVisible(compact)
        self.preset_options.setVisible(expanded)
        self.preset_summary.setVisible(compact and not expanded)

    @staticmethod
    def _readonly_details(accessible_name: str) -> QPlainTextEdit:
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard,
        )
        editor.setTabChangesFocus(True)
        editor.setAccessibleName(accessible_name)
        editor.setMinimumHeight(72)
        editor.setMaximumHeight(96)
        editor.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        option = editor.document().defaultTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        editor.document().setDefaultTextOption(option)
        editor.hide()
        return editor

    @Slot()
    def _clear_config_preview(self) -> None:
        self._previewed_config = None
        self._clear_backend_prepare_status()
        self.config_preview.clear()
        self.config_preview.hide()
        self.prepare_backend_button.setEnabled(False)
        self._update_live_controls()

    def _clear_backend_prepare_status(self) -> None:
        self._preparing_config = None
        self._prepared_config = None
        self.backend_prepare_status.clear()
        self.backend_prepare_status.hide()

    def _config_is_current(self, config: StreamConfig) -> bool:
        selection = self.controller.device_selection
        if (
            self._closing_requested or not self.controller.is_ready
            or self.controller.is_closed or selection.locked
        ):
            return False
        try:
            current = selection.build_stream_config(
                samplerate=config.samplerate, blocksize=config.blocksize,
                channels=config.channels, output_ceiling=config.output_ceiling,
            )
        except (ValueError, DeviceSelectionError):
            return False
        return current == config

    def _fill_combo(
        self, combo: QComboBox, kind: DeviceKind, devices: tuple[DeviceInfo, ...],
        selected: DeviceRef | None,
    ) -> int:
        count = 0
        with QSignalBlocker(combo):
            combo.clear()
            combo.addItem("未选择 / Not selected", None)
            for info in devices:
                channels = (
                    info.max_input_channels if kind == "input" else info.max_output_channels
                )
                if channels == 0:
                    continue
                text = f"{info.ref.name} [#{info.ref.index}, {info.hostapi_name}]"
                combo.addItem(text, info.ref)
                combo.setItemData(combo.count() - 1, f"<qt>{escape(text)}</qt>",
                                  Qt.ItemDataRole.ToolTipRole)
                count += 1
                if info.ref == selected:
                    combo.setCurrentIndex(combo.count() - 1)
        return count

    @Slot()
    def _update_device_controls(self) -> None:
        selection = self.controller.device_selection
        locked = (
            selection.locked or self._closing_requested or self.controller.is_closed
            or not self.controller.is_ready
        )
        if (
            self._previewed_config is not None and not selection.locked
            and not self._config_is_current(self._previewed_config)
        ):
            self._clear_config_preview()
        inputs = self._fill_combo(
            self.input_combo, "input", selection.devices, selection.input_device,
        )
        outputs = self._fill_combo(
            self.output_combo, "output", selection.devices, selection.output_device,
        )
        self.refresh_button.setEnabled(not locked)
        self.input_combo.setEnabled(not locked and inputs > 0)
        self.output_combo.setEnabled(not locked and outputs > 0)
        for editor in self.config_inputs:
            editor.setEnabled(not locked)
        self.validate_button.setEnabled(
            not locked and selection.input_device is not None and selection.output_device is not None,
        )
        self.prepare_backend_button.setEnabled(
            not locked and self._previewed_config is not None
            and self._preparing_config is None and self._prepared_config is None,
        )
        if self._closing_requested or self.controller.closing or self.controller.is_closed:
            message = "正在安全退出 / Closing."
        elif self._refresh_pending and self.controller.busy:
            message = "正在查找设备 / Refreshing devices."
        elif not self.controller.is_ready:
            message = "正在准备 / Preparing."
        elif locked:
            message = "正在使用设备；停止后可切换 / In use; stop to change."
        elif self._refresh_failed:
            message = "查找失败，请重试刷新 / Discovery failed; refresh to retry."
        elif not self.controller.devices_loaded:
            message = "请刷新设备列表 / Refresh the device list."
        elif inputs == 0 and outputs == 0:
            message = "没有可用设备 / No usable devices found."
        else:
            message = (
                f"设备列表：找到 {inputs} 个输入、{outputs} 个输出 / "
                f"Last valid list: {inputs} input(s), {outputs} output(s)."
            )
        self.device_status.setText(message)
        self._render_device_details()
        self._update_virtual_mic_status(selection.output_device)
        self._update_live_controls()

    def _render_device_details(self) -> None:
        selection = self.controller.device_selection
        is_zh = self._language == "zh"
        details = []
        for ref, zh_label, en_label in (
            (selection.input_device, "输入", "Input"),
            (selection.output_device, "输出", "Output"),
        ):
            if ref is not None:
                details.append(
                    f"{zh_label if is_zh else en_label}: {ref.name}\n"
                    f"{'索引' if is_zh else 'Index'}: {ref.index}; "
                    f"{'主机 API' if is_zh else 'Host API'}: {ref.hostapi}"
                )
        text = "\n\n".join(details)
        self._set_details_text(self.device_details, text)
        self.device_details.setVisible(bool(text))
        self.device_details_heading.setVisible(bool(text))

    @staticmethod
    def _set_details_text(editor: QPlainTextEdit, text: str) -> None:
        if editor.toPlainText() == text:
            return
        cursor = editor.textCursor()
        position, anchor = cursor.position(), cursor.anchor()
        vertical = editor.verticalScrollBar().value()
        horizontal = editor.horizontalScrollBar().value()
        with QSignalBlocker(editor):
            editor.setPlainText(text)
            maximum = max(0, editor.document().characterCount() - 1)
            cursor = editor.textCursor()
            cursor.setPosition(min(anchor, maximum))
            cursor.setPosition(min(position, maximum), cursor.MoveMode.KeepAnchor)
            editor.setTextCursor(cursor)
            editor.verticalScrollBar().setValue(vertical)
            editor.horizontalScrollBar().setValue(horizontal)

    def _update_virtual_mic_status(self, output: DeviceRef | None) -> None:
        capabilities = virtual_mic.capabilities_or_none()
        if capabilities is None:
            text = (
                "虚拟麦克风：未初始化\nVirtual microphone: not initialized"
            )
        elif not capabilities.available:
            text = (
                "虚拟麦克风不可用 / Virtual microphone unavailable:\n"
                f"{capabilities.reason}\n"
                "仍可使用真实输出设备监听；虚拟麦克风条目不会出现。\n"
                "Real outputs still work; no virtual device is offered."
            )
        elif output is not None and output.name == virtual_mic.PCM_NAME:
            text = (
                f"虚拟麦克风：{virtual_mic.DESCRIPTION} / "
                f"Virtual microphone: {virtual_mic.DESCRIPTION}\n"
                "监听开始时创建；其他应用请选择该麦克风。\n"
                "Created while listening; select it in the other application."
            )
        else:
            text = (
                f"虚拟麦克风已就绪：{virtual_mic.DESCRIPTION} / "
                f"Virtual microphone ready: {virtual_mic.DESCRIPTION}\n"
                "在输出设备中选择它，即可把变声结果送给其他应用。\n"
                "Choose it as the output device to feed other applications."
            )
        if self.virtual_mic_status.text() != text:
            self.virtual_mic_status.setText(text)

    @Slot()
    def _update_live_controls(self) -> None:
        idle = (
            self.controller.is_ready and not self.controller.busy
            and not self._closing_requested
        )
        stopped = idle and self.controller.state is EngineState.STOPPED
        selection = self.controller.device_selection
        media = self.controller.media_status
        self.listen_button.setEnabled(
            (
                stopped and not selection.locked and selection.input_device is not None
                or idle and media.recording and not media.monitoring
            ) and selection.output_device is not None and media.playback is None,
        )
        self.stop_button.setEnabled(
            idle and (
                self.controller.state is EngineState.FAULT
                or self.controller.state is EngineState.RUNNING
                and (not media.recording or media.monitoring or media.playback is not None)
            ),
        )
        self._language_filter.set_text(self.stop_button,
            "停止回放 / &Stop" if media.playback is not None
            else "停止监听 / &Stop" if self.controller.state is EngineState.RUNNING
            and media.monitoring else "停止 / &Stop"
        )
        self.stop_button.setAccessibleName(self.stop_button.text())
        set_visual_state(
            self.stop_button,
            "active" if self.stop_button.isEnabled() and self.controller.state is EngineState.RUNNING
            else "fault" if self.controller.state is EngineState.FAULT else "",
        )
        for control in (
            self.preset_combo, *self.parameter_sliders.values(), *self.preset_buttons.values(),
            *self._sound_inputs, *self._gain_inputs, self.denoise_input,
            self.reset_effect_button,
        ):
            control.setEnabled(
                idle and self.controller.state in (EngineState.STOPPED, EngineState.RUNNING)
                and media.playback is None
            )
        self._language_filter.set_text(self.record_button,
            "停止录音 / &Stop" if media.recording else "开始录音 / &Start recording"
        )
        self.record_button.setAccessibleName(
            "停止录音 / Stop recording" if media.recording
            else "开始录音；只需输入设备 / Start recording; input device only"
        )
        set_visual_state(self.record_button, "recording" if media.recording else "")
        can_record = (
            idle and media.playback is None
            and self.controller.state in (EngineState.STOPPED, EngineState.RUNNING)
        )
        self.record_button.setEnabled(can_record and (media.recording or selection.input_device is not None))
        playable = (
            idle and media.group is not None and media.group.frames > 0 and not media.recording
            and self.controller.state in (EngineState.STOPPED, EngineState.RUNNING)
            and selection.output_device is not None
        )
        for button in (self.play_original_button, self.play_effect_button):
            button.setEnabled(playable)
            button.setToolTip("先停止录音并选择输出；每次从录音起点播放，不叠加当前效果。\nStop recording and select output; playback starts at the beginning.")
        unsaved = media.group is not None and not media.group.saved
        self.save_recording_button.setEnabled(idle and unsaved and not media.recording and media.playback is None)
        self.discard_recording_button.setEnabled(idle and unsaved and not media.recording)
        recoverable = bool(media.recoverable) and stopped and not self.controller.owns_devices
        self.recover_recording_button.setEnabled(recoverable and not unsaved)
        self.delete_recovery_button.setEnabled(recoverable)
        if not idle:
            guidance = "正在处理，请稍候 / Working; please wait."
        elif media.playback is not None:
            guidance = "回放中；停止后可监听或录音 / Playing; stop before capture."
        elif media.recording:
            guidance = "录音独立运行；可单独停止监听 / Recording continues if monitoring stops."
        elif self.controller.state is EngineState.RUNNING:
            guidance = "监听中；按“停止”结束 / Listening; press Stop to finish."
        elif self.controller.state is EngineState.FAULT:
            guidance = "已停止，请检查提示 / Stopped; see message."
        elif selection.input_device is None:
            guidance = "先选择麦克风 / Choose a microphone first."
        elif selection.output_device is None:
            guidance = "录音只需麦克风；监听还需选择输出 / Recording needs a mic; listening also needs output."
        elif self._previewed_config is None:
            guidance = "开始监听前会检查设备 / Devices are checked before listening."
        else:
            guidance = "设备已就绪 / Devices ready."
        self._language_filter.set_text(self.listen_guidance, guidance)
        self.listen_guidance.show()
        self.listen_button.setAccessibleDescription(guidance)

    def _effect_parameters(self) -> "EffectParameters":
        from ..dsp.parameters import EffectParameters, PRESETS
        key = self.preset_combo.currentData()
        strength = self.denoise_input.value() / 100
        if key in PRESETS:
            return replace(
                PRESETS[key], input_gain_db=self.input_gain_input.value(),
                output_gain_db=self.output_gain_input.value(),
                denoise_strength=strength,
            )
        return EffectParameters(
            self.pitch_input.value(), self.formant_input.value(), False,
            input_gain_db=self.input_gain_input.value(),
            output_gain_db=self.output_gain_input.value(),
            robot_mix=self.robot_input.value() / 100,
            echo_mix=self.echo_input.value() / 100,
            reverb_mix=self.reverb_input.value() / 100,
            denoise_strength=strength,
        )

    @Slot(int)
    def _select_preset(self, index: int) -> None:
        from ..dsp.parameters import PRESETS
        key = self.preset_combo.itemData(index)
        parameters = PRESETS.get(key)
        if parameters is not None:
            self._base_preset = key
            values = (
                parameters.pitch_semitones, parameters.formant_scale,
                parameters.robot_mix * 100, parameters.echo_mix * 100,
                parameters.reverb_mix * 100,
            )
            for control, value in zip(self._sound_inputs, values):
                with QSignalBlocker(control):
                    control.setValue(value)
            for editor, slider in self.parameter_sliders.items():
                sync_parameter_slider(slider, editor.value(), self._slider_scales[editor])
        self._sync_preset_buttons(key)
        self._submit_effects()

    def _choose_preset_button(self, key: str) -> None:
        index = self.preset_combo.findData(key)
        if index < 0:
            return
        with QSignalBlocker(self.preset_combo):
            self.preset_combo.setCurrentIndex(index)
        self._select_preset(index)

    def _sync_preset_buttons(self, selected: str | None = None) -> None:
        selected = selected or self.preset_combo.currentData()
        self._feedback_scroll_timer.stop()
        self.tuning_panel.setVisible(selected == "custom")
        self.preset_summary.setText(
            f"当前 / Current: {self.preset_combo.currentText()}"
        )
        for key, button in self.preset_buttons.items():
            with QSignalBlocker(button):
                button.setChecked(key == selected)

    @Slot()
    def _edit_effect(self) -> None:
        with QSignalBlocker(self.preset_combo):
            base = self.preset_combo.itemText(self.preset_combo.findData(self._base_preset))
            custom = self.preset_combo.findData("custom")
            self.preset_combo.setItemText(custom, f"自定义（基于{base.split(' / ')[0]}） / Custom ({self._base_preset})")
            self.preset_combo.setCurrentIndex(custom)
        self._sync_preset_buttons("custom")
        self._submit_effects()

    @Slot()
    def _reset_effect(self) -> None:
        index = self.preset_combo.findData(self._base_preset)
        with QSignalBlocker(self.preset_combo):
            self.preset_combo.setCurrentIndex(index)
        self._select_preset(index)

    def _submit_effects(self) -> None:
        if self.controller.state is EngineState.RUNNING:
            if self.controller.request_parameters(self._effect_parameters()):
                self._language_filter.set_text(self.effect_status, "参数待音频块应用 / Parameters pending at block boundary")
        else:
            self._language_filter.set_text(self.effect_status, "尚未应用；下次采集生效。\nNot applied; applies on next capture.")

    @Slot()
    def _request_listen(self) -> None:
        if self.controller.media_status.recording:
            ref = self.controller.device_selection.output_device
            if ref is None:
                self._on_error(EngineError("Select an output before starting a recording to enable monitoring later"))
                return
            group = self.controller.media_status.group
            if group is None:
                self._on_error(EngineError("Active recording has no format information"))
                return
            self.controller.request_monitor(output_confirmed=True)
            self._update_live_controls()
            return
        config = self._previewed_config
        if config is None or not self._config_is_current(config):
            self._request_stream_config()
            config = self._previewed_config
        if config is None:
            return
        if self._closing_requested or self.controller.is_closed:
            return
        if self._previewed_config != config or not self._config_is_current(config):
            self._clear_config_preview()
            self._on_error(EngineError("Device configuration changed before listening; try again"))
            return
        problem = self._dsp_format_problem(config.samplerate, config.blocksize, config.channels)
        if problem is not None:
            self._on_error(EngineError(problem))
            return
        self.error_notice.hide()
        self.controller.request_listen(
            config, self._effect_parameters(),
            output_confirmed=True,
        )
        self._update_live_controls()

    @Slot()
    def _request_record(self) -> None:
        if self.controller.media_status.recording:
            self.controller.request_stop_recording()
        else:
            self._guard_recording("record")
        self._update_live_controls()

    def _start_recording(self) -> None:
        try:
            samplerate, blocksize, channels, ceiling = self._configuration_values()
        except EngineError as error:
            self._on_error(error)
            return
        problem = self._dsp_format_problem(samplerate, blocksize, channels)
        if problem is not None:
            self._on_error(EngineError(problem))
            return
        self.error_notice.hide()
        self.controller.request_record(
            self._effect_parameters(), samplerate=samplerate, blocksize=blocksize,
            channels=channels, output_ceiling=ceiling,
        )

    def _guard_recording(self, action: str) -> None:
        media = self.controller.media_status
        if action == "close" and not media.recording and (media.group is None or media.group.saved):
            self._begin_shutdown_now()
            return
        if self.controller.busy:
            if action == "close":
                self._pending_media_action, self._pending_media_phase = action, "wait"
                self.activity_status.setText("等待当前操作后退出 / Waiting for the current operation before exit")
            return
        if (
            self.controller.state in (EngineState.RUNNING, EngineState.FAULT)
            and (action != "record" or media.group is not None and not media.group.saved)
        ):
            self._pending_media_action, self._pending_media_phase = action, "stop"
            if not self.controller.request_stop_all():
                self._pending_media_action = self._pending_media_phase = None
            return
        if media.group is not None and not media.group.saved:
            dialog = QMessageBox(self)
            dialog.setWindowTitle("未保存录音 / Unsaved recording")
            dialog.setTextFormat(Qt.TextFormat.PlainText)
            dialog.setText("保存、丢弃还是取消？未保存的双轨仍在本机临时目录。\nSave, discard or cancel? Temporary tracks remain on this computer.")
            dialog.setStandardButtons(
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            )
            dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
            answer = dialog.exec()
            dialog.deleteLater()
            if answer == QMessageBox.StandardButton.Save:
                self._pending_media_action, self._pending_media_phase = action, "save"
                if not self._save_recording():
                    self._pending_media_action = self._pending_media_phase = None
                return
            if answer == QMessageBox.StandardButton.Discard:
                self._pending_media_action, self._pending_media_phase = action, "discard"
                if not self.controller.request_discard_recording():
                    self._pending_media_action = self._pending_media_phase = None
                return
            self._pending_media_action = self._pending_media_phase = None
            return
        self._pending_media_action = self._pending_media_phase = None
        if action == "record":
            self._start_recording()
        elif action == "close":
            self._begin_shutdown_now()

    def _continue_media_action(self) -> None:
        action, phase = self._pending_media_action, self._pending_media_phase
        if action is None or self.controller.busy:
            return
        media = self.controller.media_status
        self._pending_media_action = self._pending_media_phase = None
        if phase == "save" and (media.group is None or not media.group.saved):
            return
        if phase == "discard" and media.group is not None:
            return
        self._guard_recording(action)

    def _request_play(self, track: Track) -> None:
        group = self.controller.media_status.group
        output = self.controller.device_selection.output_device
        if group is None or output is None:
            self._on_error(EngineError("Choose a finalized recording and an output device"))
            return
        try:
            blocksize = int(self.blocksize_input.text())
            ceiling = float(self.output_ceiling_input.text())
            config = self.controller.device_selection.build_output_config(
                samplerate=group.samplerate, blocksize=blocksize,
                channels=group.channels, output_ceiling=ceiling,
            )
        except (ValueError, DeviceSelectionError) as error:
            self._on_error(EngineError(f"Invalid playback configuration: {error}"))
            return
        if self._closing_requested or self.controller.media_status.group is not group:
            return
        try:
            current = self.controller.device_selection.build_output_config(
                samplerate=group.samplerate, blocksize=int(self.blocksize_input.text()),
                channels=group.channels, output_ceiling=float(self.output_ceiling_input.text()),
            )
        except (ValueError, DeviceSelectionError) as error:
            self._on_error(EngineError(f"Invalid playback configuration before playback: {error}"))
            return
        if current != config:
            self._on_error(EngineError("Output or playback configuration changed before playback"))
            return
        self.controller.request_play(
            track, blocksize=blocksize, output_ceiling=ceiling, output_confirmed=True,
        )
        self._update_live_controls()

    def _save_recording(self) -> bool:
        if self.controller.preferences_enabled:
            if self._save_dialog_pending:
                self._on_error(EngineError("Save directory check is already pending"))
                return False
            self._save_dialog_pending = True
            if not self.controller.request_save_directory():
                self._save_dialog_pending = False
                return False
            return True
        return self._show_save_dialog(str(Path.home()))

    @Slot(object)
    def _on_preferences(self, preferences: Preferences) -> None:
        if self._preferences_applied or self._closing_requested:
            return
        self._preferences_applied = True
        if preferences.audio is not None:
            audio = preferences.audio
            if (
                audio.samplerate in self._DSP_SAMPLERATES
                and audio.blocksize in self._DSP_BLOCKSIZES
                and audio.channels in self._DSP_CHANNELS
            ):
                for editor, value in zip(self.config_inputs, (
                    audio.samplerate, audio.blocksize, audio.channels, audio.output_ceiling,
                )):
                    editor.setText(str(value))
                self.notice.setText(
                    self.notice.text() + "\n已预填历史成功配置；自动查找后请核对设备，输出仍关闭。\n"
                    "History prefilled only; check devices after discovery. Output remains off."
                )
            else:
                self.notice.setText(
                    self.notice.text()
                    + f"\n历史成功配置为 {audio.samplerate} Hz、{audio.blocksize} 帧、"
                    f"{audio.channels} 声道；DSP 演示固定 48 kHz、256/512/1024 帧、1–2 声道，"
                    "未预填，需在「自定义声音」诊断面板中手动设置。\n"
                    "Saved format is outside the DSP demo's fixed format; not prefilled."
                )

    @Slot(str)
    def _on_save_directory_ready(self, directory: str) -> None:
        if not self._save_dialog_pending:
            return
        self._save_dialog_pending = False
        if self._closing_requested:
            return
        if not self.save_recording_button.isEnabled():
            self._pending_media_action = self._pending_media_phase = None
            self._on_error(EngineError("Audio state changed before choosing a save location; retry"))
            return
        if not self._show_save_dialog(directory):
            self._pending_media_action = self._pending_media_phase = None

    def _show_save_dialog(self, directory: str) -> bool:
        selected, _ = QFileDialog.getSaveFileName(
            self, "保存录音组：输入新目录名 / Save group: new folder name",
            str(Path(directory) / "voice-recording"), "录音组目录 / Recording group (*)",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not selected:
            return False
        path = Path(selected)
        self.error_notice.hide()
        return self.controller.request_save_recording(str(path.parent), path.name)

    def _choose_recovery(self, delete: bool) -> None:
        paths = self.controller.media_status.recoverable
        if not paths:
            self._on_error(EngineError("There are no listed recoverable recordings"))
            return
        selected, accepted = QInputDialog.getItem(
            self, "遗留录音 / Leftover recordings", "选择本机临时录音 / Choose a local temporary group",
            list(paths), 0, False,
        )
        if not accepted:
            return
        if delete:
            dialog = QMessageBox(self)
            dialog.setTextFormat(Qt.TextFormat.PlainText)
            dialog.setText(f"永久删除这组遗留临时录音？\nDelete this temporary group?\n{selected}")
            dialog.setStandardButtons(QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
            answer = dialog.exec()
            dialog.deleteLater()
            if answer == QMessageBox.StandardButton.Discard:
                self.controller.request_delete_recoverable(selected)
        else:
            self.controller.request_recover_recording(selected)

    @Slot(object)
    def _on_media_status(self, media: MediaStatus) -> None:
        if media.playback == "generated":
            track_name, track_english = "朗读结果", "Generated speech"
        elif media.playback == "original":
            track_name, track_english = "原声", "Original"
        else:
            track_name, track_english = "效果", "Effect"
        labels = {
            RecordingState.NONE: "未录音 / No recording",
            RecordingState.RECORDING: "录音中 / Recording",
            RecordingState.FINALIZING: "正在完成录音 / Finishing recording",
            RecordingState.UNSAVED: "录音已完成，可试听 / Ready to listen",
            RecordingState.SAVING: "正在保存 / Saving",
            RecordingState.SAVED: "已保存 / Saved",
            RecordingState.SAVE_FAILED: "保存失败，录音仍保留 / Save failed; recording retained",
            RecordingState.INCOMPLETE: "录音不完整 / Incomplete recording",
        }
        text = labels[media.recording_state]
        if media.recording and media.recording_state is RecordingState.RECORDING:
            text = f"录音中 {media.duration:.0f}s / Recording"
        elif media.playback is not None:
            text = f"正在播放{track_name} / Playing {track_english}"
        self._language_filter.set_text(self.recording_status, text)
        set_visual_state(
            self.recording_status,
            "recording" if media.recording else
            "fault" if media.recording_state in (
                RecordingState.SAVE_FAILED, RecordingState.INCOMPLETE,
            ) else "",
        )
        if media.group is not None:
            group = media.group
            space = f"\n可用空间 / Free: {media.free_bytes / 1048576:.1f} MiB" if media.free_bytes is not None else ""
            self.recording_details.setText(
                f"{labels[media.recording_state]}\n{group.samplerate} Hz; {group.channels} channel(s); "
                f"PCM16; {media.duration:.3f}s{space}\n{group.directory}\n"
                "原声为增益前采集，效果为限幅后DSP；对齐/参数变化见说明。\n"
                "Original: pre-gain input. Effect: limited DSP output. See metadata for alignment."
            )
        elif media.recoverable:
            self.recording_details.setText(
                f"发现 {len(media.recoverable)} 组遗留本机临时录音，可尝试恢复或明确删除。\n"
                "Local leftover recordings found; choose Recover or Delete. Nothing plays automatically."
            )
        else:
            self.recording_details.setText(
                "停止后可试听、保存或丢弃 / Stop to listen, save or discard."
            )
        has_group = media.group is not None
        review_ready = has_group and not media.recording
        unsaved = review_ready and not media.group.saved
        has_recovery = bool(media.recoverable)
        self.recording_heading.show()
        self.recording_details.setVisible(review_ready or has_recovery or media.recording)
        self.play_original_button.setVisible(review_ready)
        self.play_effect_button.setVisible(review_ready)
        self.save_recording_button.setVisible(unsaved)
        self.discard_recording_button.setVisible(unsaved)
        self.recover_recording_button.setVisible(has_recovery and not unsaved)
        self.delete_recovery_button.setVisible(has_recovery)
        if media.playback is not None:
            name = f"回放音量（{track_name}） / Playback level ({track_english})"
            self.output_level.setToolTip(
                f"{track_name}轨回放峰值，不代表耳机或扬声器实测值 / "
                f"{track_english} track playback peak, not a physical output measurement"
            )
            self._language_filter.set_text(
                self.effect_status,
                "播放朗读结果，不套用 DSP 效果 / Generated speech; DSP effects are not applied"
                if media.playback == "generated" else
                "播放已录轨道，不套用当前效果 / Recorded track; current effects are not applied",
            )
        else:
            name = "处理后 / Processed"
            self.output_level.setToolTip(_PROCESSED_PEAK_TOOLTIP)
        if name != self._last_output_level_name:
            self._last_output_level_name = name
            self.output_level.set_name(name)
        self._language_filter.localize_subtree(self.output_level)
        self._sync_wave_source(media)
        self._refresh_activity_status(self.controller.state, media)
        self._refresh_audio_status()
        self._update_live_controls()

    def _sync_wave_source(self, media: MediaStatus) -> None:
        source = None
        if (
            self.controller.state is EngineState.RUNNING
            and not self._closing_requested and media.meter_source_id > 0
        ):
            if media.playback is not None:
                source = ("playback", media.meter_source_id)
            elif media.monitoring or media.recording:
                source = ("capture", media.meter_source_id)
        if source != self._wave_source:
            self._wave_source = source
            self._last_capture_levels = None
            self._last_playback_peak_sequence = None
            self._last_analysis_sequence = None
            self._last_analysis_drops = 0
            self.input_level.set_idle()
            self.output_level.set_idle()
            if source is not None:
                if source[0] == "capture":
                    self.input_level.set_waiting()
                self.output_level.set_waiting()
        if source is not None and source[0] == "playback":
            if media.output_peak is None:
                if self._last_playback_peak_sequence is not None:
                    self._last_playback_peak_sequence = None
                    self.output_level.set_waiting()
            elif media.output_peak_sequence != self._last_playback_peak_sequence:
                self._last_playback_peak_sequence = media.output_peak_sequence
                self.output_level.set_peak(media.output_peak)
        analysis = media.analysis
        if (
            source is None or analysis is None
            or (
                self._last_analysis_sequence is not None
                and analysis.sequence <= self._last_analysis_sequence
            )
        ):
            return
        self._last_analysis_sequence = analysis.sequence
        if analysis.dropped_blocks > self._last_analysis_drops:
            if source[0] == "capture":
                self.input_level.set_analysis_waiting()
            self.output_level.set_analysis_waiting()
        self._last_analysis_drops = analysis.dropped_blocks
        lanes = (
            ((self.input_level, analysis.input), (self.output_level, analysis.output))
            if source[0] == "capture" else ((self.output_level, analysis.output),)
        )
        for meter, reading in lanes:
            meter.set_analysis_info(analysis.sample_rate, analysis.dropped_blocks)
            if reading is None:
                meter.set_analysis_waiting()
            else:
                meter.set_analysis(reading)

    @Slot()
    def _request_stop(self) -> None:
        self.controller.request_stop()
        self._update_live_controls()

    @Slot(object)
    def _on_live_status(
        self, status: tuple[tuple[float | None, float | None] | None, int, int],
    ) -> None:
        levels, applied, requested = status
        if (
            self.controller.state is not EngineState.RUNNING or self._closing_requested
            or self._wave_source is None or self._wave_source[0] != "capture"
        ):
            return
        if levels is None:
            if self._last_capture_levels is not None:
                self._last_capture_levels = None
                self.input_level.set_waiting()
                self.output_level.set_waiting()
        elif levels is not self._last_capture_levels:
            # Each capture callback publishes a new tuple, even for equal-valued blocks.
            self._last_capture_levels = levels
            self.input_level.set_peak(levels[0])
            self.output_level.set_peak(levels[1])
        self._language_filter.set_text(self.effect_status,
            "参数已在DSP输出块应用 / Parameters applied to DSP output blocks"
            if applied == requested else "参数待音频块应用 / Parameters pending at block boundary"
        )

    @Slot()
    def _request_refresh(self, *, clear_error: bool = True) -> None:
        if self._refresh_pending or not self.controller.is_ready or self.controller.busy:
            return
        self._startup_refresh_due = False
        self._clear_config_preview()
        self._refresh_pending = True
        self._refresh_failed = False
        if clear_error:
            self.error_notice.hide()
        if not self.controller.request_device_refresh():
            self._refresh_pending = False
            self._update_device_controls()

    def _select_device(self, kind: DeviceKind, index: int) -> None:
        combo = self.input_combo if kind == "input" else self.output_combo
        ref = combo.itemData(index)
        if ref is not None and not isinstance(ref, DeviceRef):
            self._on_error(EngineError("Invalid device item; refresh and reselect"))
            self._update_device_controls()
            return
        self.controller.select_device(kind, ref)
        self._update_device_controls()

    @Slot()
    def _on_operation_finished(self) -> None:
        self._refresh_pending = False
        self._update_device_controls()
        self._maybe_startup_refresh()
        if self._pending_media_action is not None:
            QTimer.singleShot(0, self._continue_media_action)
        if self._preparing_config is None:
            return
        requested = self._preparing_config
        prepared = self._prepared_config
        self._preparing_config = None
        if prepared != requested:
            self._clear_backend_prepare_status()
            return
        if not self._config_is_current(requested):
            self._clear_config_preview()
            self._on_error(EngineError("Prepared backend configuration is no longer current"))
            return
        self._prepared_config = requested
        self.backend_prepare_status.setText(
            "原声后端对象已准备，但输出仍未确认；没有打开音频流，也没有采集或播放。\n"
            "Original-audio backend object prepared; output remains unconfirmed. "
            "No audio stream was opened; nothing was captured or played."
        )
        self.backend_prepare_status.show()
        self.prepare_backend_button.setEnabled(False)
        self._feedback_scroll_timer.start()

    @Slot()
    def _request_backend_prepare(self) -> None:
        config = self._previewed_config
        if config is None:
            self._on_error(
                EngineError("Validate a current configuration before preparing the backend")
            )
            return
        if not self._config_is_current(config):
            self._clear_config_preview()
            self._on_error(EngineError("Configuration preview is stale; validate it again"))
            return
        self.error_notice.hide()
        self._preparing_config = config
        self._prepared_config = None
        self.backend_prepare_status.setText(
            "正在准备未确认的原声后端；不会打开音频流。\n"
            "Preparing the unconfirmed original-audio backend; no stream will be opened."
        )
        self.backend_prepare_status.show()
        self.prepare_backend_button.setEnabled(False)
        self._feedback_scroll_timer.start()
        if not self.controller.request_backend_prepare(config):
            self._clear_backend_prepare_status()
            self._update_device_controls()

    @Slot(object)
    def _on_backend_prepared(self, config: StreamConfig) -> None:
        if self._closing_requested or self.controller.is_closed:
            return
        if self._preparing_config is None:
            return
        if not isinstance(config, StreamConfig) or config != self._preparing_config:
            self._clear_backend_prepare_status()
            self._on_error(
                EngineError("Backend preparation result did not match the current request")
            )
            return
        self._prepared_config = config
        self.backend_prepare_status.setText(
            "后端对象已构造，正在复核配置；输出仍未确认，未打开音频流。\n"
            "Backend object constructed; verifying configuration. Output remains "
            "unconfirmed; no audio stream was opened."
        )
        self.backend_prepare_status.show()
        self._feedback_scroll_timer.start()

    @Slot()
    def _request_stream_config(self) -> None:
        self._clear_config_preview()
        self.error_notice.hide()
        try:
            samplerate, blocksize, channels, output_ceiling = self._configuration_values()
        except EngineError as error:
            self._on_error(error)
            return
        self.controller.request_stream_config(
            samplerate=samplerate, blocksize=blocksize, channels=channels,
            output_ceiling=output_ceiling,
        )

    @classmethod
    def _dsp_format_problem(cls, samplerate: int, blocksize: int, channels: int) -> str | None:
        if (
            samplerate in cls._DSP_SAMPLERATES
            and blocksize in cls._DSP_BLOCKSIZES
            and channels in cls._DSP_CHANNELS
        ):
            return None
        return (
            f"DSP 演示固定 48 kHz、块长 256/512/1024 帧、1–2 声道；当前为 {samplerate} Hz、"
            f"{blocksize} 帧、{channels} 声道，请在「自定义声音」诊断面板中改回演示格式后重新校验。\n"
            "The DSP demo supports only 48 kHz, 256/512/1024 frames and 1–2 channels; "
            f"current format is {samplerate} Hz / {blocksize} frames / {channels} channels."
        )

    def _configuration_values(self) -> tuple[int, int, int, float]:
        field = "samplerate"
        try:
            samplerate = int(self.samplerate_input.text())
            field = "blocksize"
            blocksize = int(self.blocksize_input.text())
            field = "channels"
            channels = int(self.channels_input.text())
            field = "output_ceiling"
            output_ceiling = float(self.output_ceiling_input.text())
        except ValueError as cause:
            error = EngineError("Invalid configuration input", (OperationFailure(field, cause),))
            raise error from cause
        return samplerate, blocksize, channels, output_ceiling

    @Slot(object)
    def _on_stream_config_ready(self, config: StreamConfig) -> None:
        self._clear_config_preview()
        selection = self.controller.device_selection
        if self._closing_requested or not self.controller.is_ready or selection.locked:
            return
        try:
            current = selection.build_stream_config(
                samplerate=config.samplerate, blocksize=config.blocksize,
                channels=config.channels, output_ceiling=config.output_ceiling,
            )
            if current != config:
                raise DeviceSelectionError("Configuration changed; validate again")
        except (ValueError, DeviceSelectionError) as cause:
            error = EngineError("Configuration preview is no longer current", (
                OperationFailure("preview", cause),
            ))
            error.__cause__ = cause
            self._on_error(error)
            return
        self._previewed_config = config
        self._render_config_preview(config)
        self.config_preview.show()
        self.config_preview.setFocus()
        self._update_device_controls()
        self._feedback_scroll_timer.start()

    def _render_config_preview(self, config: StreamConfig) -> None:
        is_zh = self._language == "zh"
        if is_zh:
            text = (
                "仅配置预览，未应用；不启动音频。\n\n"
                f"输入：{config.input_device.name} [索引 {config.input_device.index}，"
                f"主机 API {config.input_device.hostapi}]\n"
                f"输出：{config.output_device.name} [索引 {config.output_device.index}，"
                f"主机 API {config.output_device.hostapi}]\n"
                f"采样率：{config.samplerate} Hz\n块长：{config.blocksize} 帧\n"
                f"通道数（输入/输出）：{config.channels}\n"
                f"输出峰值上限：{config.output_ceiling!r}"
            )
        else:
            text = (
                "Configuration preview only; not applied. No audio starts.\n\n"
                f"Input: {config.input_device.name} [index {config.input_device.index}, "
                f"host API {config.input_device.hostapi}]\n"
                f"Output: {config.output_device.name} [index {config.output_device.index}, "
                f"host API {config.output_device.hostapi}]\n"
                f"Sample rate: {config.samplerate} Hz\nBlock size: {config.blocksize} frames\n"
                f"Channels (input/output): {config.channels}\n"
                f"Output ceiling: {config.output_ceiling!r}"
            )
        self._set_details_text(self.config_preview, text)

    @Slot()
    def _reveal_feedback(self) -> None:
        self._language_filter.localize_tree()
        QApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
        target = next((
            widget for widget in (
                self.error_notice, self.backend_prepare_status, self.config_preview,
            ) if widget.isVisible()
        ), None)
        if target is None:
            return
        if target is self.error_notice:
            self.scroll_area.verticalScrollBar().setValue(0)
            return
        content = self.scroll_area.widget()
        if content.layout() is not None:
            content.layout().activate()
        self.scroll_area.ensureWidgetVisible(target, 0, 0)
        if content.layout() is not None:
            content.layout().activate()
        top = target.mapTo(self.scroll_area.widget(), target.rect().topLeft()).y()
        bottom = top + target.height()
        scrollbar = self.scroll_area.verticalScrollBar()
        current = scrollbar.value()
        viewport_height = self.scroll_area.viewport().height()
        if top < current:
            current = top
        elif bottom > current + viewport_height:
            current = bottom - viewport_height
        scrollbar.setValue(max(0, current))

    @Slot(object)
    def _on_state(self, state: EngineState) -> None:
        if state is not self._displayed_state:
            self._clear_config_preview()
        self._displayed_state = state
        self._refresh_activity_status(state, self.controller.media_status)
        self._refresh_audio_status()
        if state is not EngineState.RUNNING:
            self._wave_source = None
            for meter in (self.input_level, self.output_level):
                if state is EngineState.STARTING:
                    meter.set_waiting()
                else:
                    meter.set_idle()
            self._language_filter.set_text(self.effect_status, "尚未应用；启动采集后生效。\nNot applied; start capture to apply.")
        self._update_live_controls()

    def _refresh_activity_status(self, state: EngineState, media: MediaStatus) -> None:
        if self._closing_requested:
            return
        if state is EngineState.RUNNING:
            if media.playback == "generated":
                message = "正在播放朗读结果 / Playing generated speech"
            elif media.playback is not None:
                message = "正在播放录音 / Playing recording"
            elif media.recording and not media.monitoring:
                message = "录音中，监听关闭 / Recording; monitoring off"
            elif media.recording:
                message = "监听并录音 / Listening and recording"
            else:
                message = "正在监听 / Listening"
        else:
            message = {
                EngineState.STOPPED: "已停止 / Stopped",
                EngineState.STARTING: "正在启动 / Starting",
                EngineState.STOPPING: "正在停止 / Stopping",
                EngineState.FAULT: "故障，已停止 / Fault; stopped",
            }[state]
        self._language_filter.set_text(self.activity_status, message)
        set_visual_state(
            self.activity_status,
            "active" if state is EngineState.RUNNING and media.playback is None else
            "fault" if state is EngineState.FAULT else "",
        )

    @Slot(object)
    def _on_error(self, error: EngineError) -> None:
        if self._refresh_pending and any(
            failure.operation in ("discover", "refresh") for failure in error.failures
        ):
            self._refresh_failed = True
        self._pending_media_action = self._pending_media_phase = None
        if self._preparing_config is not None:
            self._clear_backend_prepare_status()
        self.error_notice.setText(f"操作失败 / Operation failed:\n{error}")
        if self.preset_combo.currentData() != "custom" and any(
            failure.operation in ("samplerate", "blocksize", "channels", "output_ceiling")
            for failure in error.failures
        ):
            self.error_notice.setText(
                self.error_notice.text() + "\n请选择自定义声音修正配置。\n"
                "Choose Custom to correct the audio configuration."
            )
        set_visual_state(self.error_notice, "fault")
        self.error_notice.show()
        self._update_device_controls()
        self._feedback_scroll_timer.start()

    @Slot()
    def _begin_shutdown(self) -> None:
        if self._closing_requested:
            return
        if self.tts_page is not None and self.tts_page.busy:
            self._tts_close_pending = True
            self.tts_page.job.cancel()
            return
        self._guard_recording("close")

    def _tts_idle(self) -> None:
        if self._tts_close_pending:
            self._tts_close_pending = False
            QTimer.singleShot(0, self._begin_shutdown)

    def _refresh_audio_status(self) -> None:
        state = self.controller.state
        media = self.controller.media_status
        if state is EngineState.RUNNING:
            if media.recording_state is RecordingState.FINALIZING:
                message = "正在完成录音 / Finishing recording"
            elif media.playback == "generated":
                message = "正在播放朗读结果 / Playing generated speech"
            elif media.playback is not None:
                message = "正在播放录音 / Playing recording"
            elif media.recording and not media.monitoring:
                message = "录音中，监听关闭 / Recording; monitoring off"
            elif media.recording:
                message = "监听并录音 / Listening and recording"
            else:
                message = "正在监听 / Listening"
        elif state is EngineState.STOPPED and not self.controller.owns_devices:
            message = "未采集 / No capture"
        elif state is EngineState.FAULT:
            message = (
                "故障，正在释放资源 / Fault; releasing resources"
                if self.controller.owns_devices else "故障，采集已停止 / Fault; capture stopped"
            )
        else:
            message = "正在启停或清理 / Starting, stopping or cleaning up"
        self._language_filter.set_text(self.audio_status, message)
        self.input_level.set_activity_description(message)
        self.output_level.set_activity_description(message)
        media = self.controller.media_status
        set_visual_state(
            self.audio_status,
            "recording" if media.recording else
            "active" if state is EngineState.RUNNING and media.playback is None else
            "fault" if state is EngineState.FAULT else "",
        )

    def _begin_shutdown_now(self) -> None:
        if self._closing_requested:
            return
        if self.controller.is_closed:
            self._on_shutdown_complete()
            return
        self._closing_requested = True
        self._wave_source = None
        self.input_level.set_idle()
        self.output_level.set_idle()
        self._clear_config_preview()
        self.activity_status.setText("正在安全退出，请稍候 / Closing; waiting for cleanup")
        self.close_button.setEnabled(False)
        self.retry_button.hide()
        self._update_device_controls()
        if not self.controller.closing and not self.controller.request_shutdown():
            self._on_shutdown_failed()

    @Slot()
    def _on_shutdown_failed(self) -> None:
        self._closing_requested = False
        self.activity_status.setText("退出未完成，请重试 / Shutdown incomplete; retry")
        self.retry_button.show()
        self.close_button.setEnabled(True)
        self.retry_button.setFocus()
        self._update_device_controls()

    @Slot()
    def _on_shutdown_complete(self) -> None:
        if self.tts_page is not None:
            self.tts_page.job.dispose()
        self._allow_close = True
        self.controller.deleteLater()
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._allow_close:
            event.accept()
        else:
            event.ignore()
            self._begin_shutdown()
