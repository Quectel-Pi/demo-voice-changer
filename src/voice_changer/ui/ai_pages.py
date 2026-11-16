"""AI read-aloud and AI voice-conversion pages.

No model is loaded on page creation. Without a configured backend every AI
action stays disabled with an explicit reason; nothing is simulated.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import importlib.util
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QSettings, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QGridLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..ai import beatrice, hevc, hevc_rknn, model_catalog
from ..ai.availability import Availability, availability
from ..ai.model_inspection import ModelInspection
from ..audio.engine import EngineState
from .layout import footer_status, new_panel, page_heading
from .theme import SPACING, set_visual_state

if TYPE_CHECKING:
    from ..app.engine_controller import EngineController


#: JVS candidate voices, derived from the public ``average_pitch`` metadata. The
#: corpus has about as many female as male speakers; the three female candidates
#: are the high-pitch speakers and the low-pitch JVS 6 stays as the male one.
#: The engine follows the source pitch, so female entries also carry a default
#: pitch lift and an external formant ratio: measured on the board, a 124 Hz male
#: source becomes about 232-259 Hz with +12 st pitch, and the Rubber Band stage
#: lifts the formants (net envelope x1.18) so the timbre reads as female instead
#: of "male voice with raised pitch". Labels are preview tags only: the metadata
#: was not acoustically verified, so nothing here claims a confirmed speaker
#: gender or identity.
_BEATRICE_VOICES = (
    ("女声候选·清亮（JVS 10） / female candidate (JVS 10)", 10, 12.0, 0.0, 1.18),
    ("女声候选·柔和（JVS 14） / female candidate (JVS 14)", 14, 12.0, 0.0, 1.18),
    ("女声候选·高音（JVS 93） / female candidate (JVS 93)", 93, 12.0, 0.0, 1.18),
    ("男声候选·低沉（JVS 6） / low male candidate (JVS 6)", 6, 0.0, 0.0, 1.0),
)
_BEATRICE_VOICE_PARAMS = {
    speaker: (pitch_shift, formant_shift, external_formant)
    for _text, speaker, pitch_shift, formant_shift, external_formant in _BEATRICE_VOICES
}
_BEATRICE_VOICE_TIP = (
    "JVS 语料共 100 个说话人，女声与男声各约一半。女声候选按公开的 average_pitch "
    "选在高音区（JVS 10/14/93，average_pitch 60.4–61.1，MIDI 音高），男声候选选在低音区"
    "（JVS 6，43.5）。引擎默认保留输入音高，因此女声候选会自动升调 12 个半音，并在输出后"
    "用 Rubber Band 做 ×1.18 的共振峰提升（模型自带的共振峰参数上限只有 2 个半音，实测"
    "不足以摆脱男声音色）；板端实测约 124 Hz 的男声源会落到约 232–259 Hz 的女声范围。"
    "名称只是试听标签，未做声纹鉴定。\n"
    "The JVS corpus has 100 speakers, about half female. The female candidates are "
    "the high average_pitch speakers (JVS 10/14/93); JVS 6 is the low-pitch male "
    "candidate. The engine keeps the source pitch, so the female candidates default "
    "to a +12 st pitch lift and a Rubber Band x1.18 formant lift after the model "
    "(the native formant parameter is capped at 2 semitones and measured too weak "
    "to move the timbre out of the male range). Measured: a 124 Hz male source "
    "lands at about 232-259 Hz. Labels are preview tags only."
)


def _label(text: str = "", *, role: str | None = None, name: str | None = None) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    if role:
        label.setProperty("role", role)
    if name:
        label.setAccessibleName(name)
    return label


class _Task(QObject):
    done = Signal(object, object)


def run_background(parent: QObject, work: Callable[[], object],
                   callback: Callable[[object, BaseException | None], None]) -> None:
    """Run file I/O off the GUI thread; ``callback`` runs on the GUI thread."""
    task = _Task(parent)

    def finished(result, error):
        task.deleteLater()
        callback(result, error)

    task.done.connect(finished)

    def target():
        try:
            result = work()
        except Exception as error:  # reported to the user, never swallowed
            task.done.emit(None, error)
        else:
            task.done.emit(result, None)

    threading.Thread(target=target, daemon=True, name="voice-profile-io").start()



def _scroll_page(page: QWidget) -> tuple[QVBoxLayout, QVBoxLayout]:
    """Scrollable body plus a fixed bottom row, mirroring the DSP page structure."""
    outer = QVBoxLayout(page)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    body = QWidget()
    body.setObjectName("canvas")
    layout = QVBoxLayout(body)
    layout.setContentsMargins(SPACING["lg"], SPACING["lg"], SPACING["lg"], SPACING["lg"])
    layout.setSpacing(SPACING["sm"])
    scroll.setWidget(body)
    outer.addWidget(scroll, 1)
    footer = QWidget()
    footer.setObjectName("fixedFooter")
    footer_layout = QVBoxLayout(footer)
    footer_layout.setContentsMargins(16, 8, 16, 8)
    outer.addWidget(footer)
    page.scroll_area = scroll
    return layout, footer_layout


def _availability_notice(state: Availability) -> QLabel:
    notice = _label(state.reason, name="功能可用性 / Feature availability")
    set_visual_state(notice, "" if state.available else "fault")
    notice.setVisible(not state.available)
    return notice


def _layout_voice_grid(grid: QGridLayout, buttons: dict[str, QPushButton], width: int) -> None:
    columns = 3 if width >= 700 else 2
    for index, button in enumerate(buttons.values()):
        grid.addWidget(button, index // columns, index % columns)
    for column in range(3):
        grid.setColumnStretch(column, 1 if column < columns else 0)


class AiTtsPage(QWidget):
    devicesRequested = Signal()
    idle = Signal()

    def __init__(self, parent: QWidget | None = None,
                 *, controller: EngineController | None = None,
                 output_ceiling: Callable[[], float] = lambda: 0.05) -> None:
        super().__init__(parent)
        from .tts_job import TtsJob

        self.controller = controller
        self.output_ceiling = output_ceiling
        self.job = TtsJob(self)
        self.job.changed.connect(self._job_changed)
        self._saving = False
        self._message = ""
        self.selected_voice: str | None = None
        self.voice_buttons: dict[str, QPushButton] = {}
        kokoro_path = os.environ.get("VOICE_CHANGER_KOKORO_MODEL")
        self.kokoro_model = Path(kokoro_path).expanduser() if kokoro_path else None
        self.kokoro_python = os.environ.get("VOICE_CHANGER_KOKORO_PYTHON", sys.executable)
        self.state = Availability("tts", False, "")
        layout, footer = _scroll_page(self)
        page_heading(
            layout, "AI 朗读 / AI read-aloud",
            "选择音色，输入文字，再生成并试听。 / Choose a voice, enter text, then generate and listen.",
        )
        self.availability_notice = _availability_notice(self.state)
        layout.addWidget(self.availability_notice)
        voice_panel, voice_layout = new_panel()
        voice_layout.addWidget(_label("朗读音色 / Read-aloud voice", role="sectionHeading"))
        self.voice_grid = QGridLayout()
        self.voice_grid.setSpacing(SPACING["sm"])
        voice_layout.addLayout(self.voice_grid)
        self.voice_group = QButtonGroup(self)
        self.voice_group.setExclusive(True)
        self.voice_hint = _label(role="muted")
        voice_layout.addWidget(self.voice_hint)
        if self.kokoro_model is not None:
            voice_layout.addWidget(_label(
                "Kokoro 预置音色：建议先输入短句，每次最多 200 字；生成可能需要数十秒。 / "
                "Kokoro presets: start with a short sentence, up to 200 characters; generation can take tens of seconds.",
                role="muted",
            ))
        layout.addWidget(voice_panel)
        text_panel, text_layout = new_panel()
        text_label = _label("朗读文字 / Text to read", role="sectionHeading")
        text_layout.addWidget(text_label)
        self.text_input = QPlainTextEdit()
        text_label.setBuddy(self.text_input)
        self.text_input.setAccessibleName("朗读文字 / Text to read")
        self.text_input.setPlaceholderText("输入要朗读的文字 / Enter text to read aloud")
        self.text_input.setPlainText("Hello, 欢迎使用语音工作台")
        self.text_input.setTabChangesFocus(True)
        self.text_input.textChanged.connect(self.update_controls)
        # Keep the editor at roughly four text lines instead of letting it absorb
        # every spare pixel; the generated-audio panel then stays visible.
        self._fit_text_input()
        self.text_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        text_layout.addWidget(self.text_input)
        layout.addWidget(text_panel)
        action_row = QHBoxLayout()
        self.generate_button = QPushButton("生成朗读 / Generate")
        self.generate_button.clicked.connect(self.generate)
        self.generate_button.setProperty("role", "primary")
        action_row.addWidget(self.generate_button, 1)
        self.status = _label(name="朗读状态 / Read-aloud status")
        footer_status(footer).addWidget(self.status)
        result_panel, result_layout = new_panel()
        result_layout.addWidget(_label("生成结果 / Generated audio", role="sectionHeading"))
        self.result = _label("尚无生成结果 / No generated audio", role="muted")
        result_layout.addWidget(self.result)
        result_layout.addWidget(_label(
            "播放使用所选输出设备，请使用耳机或降低音量。 / "
            "Playback uses the selected output; use headphones or lower volume.",
            role="muted",
        ))
        result_row = QHBoxLayout()
        self.play_button = QPushButton("播放 / Play")
        self.stop_button = QPushButton("停止 / Stop")
        self.save_button = QPushButton("保存音频 / Save audio")
        self.play_button.clicked.connect(self.play)
        self.stop_button.clicked.connect(self.stop)
        self.save_button.clicked.connect(self.save)
        self.devices_button = QPushButton("音频设备 / Audio devices")
        self.devices_button.clicked.connect(self.devicesRequested)
        for button in (self.play_button, self.stop_button, self.save_button):
            button.setEnabled(False)
        for button in (self.play_button, self.save_button):
            result_row.addWidget(button)
        result_row.addStretch()
        result_row.addWidget(self.devices_button)
        result_layout.addLayout(result_row)
        layout.addWidget(result_panel)
        layout.addStretch(1)
        action_row.addWidget(self.stop_button)
        footer.addLayout(action_row)
        if controller is not None:
            controller.operationFinished.connect(self.update_controls)
            controller.stateChanged.connect(self.update_controls)
            controller.mediaStatus.connect(self.update_controls)
            controller.errorOccurred.connect(self._audio_error)
        self.refresh_model()
        self.refresh_voices()

    @Slot()
    def refresh_voices(self) -> None:
        for button in self.voice_buttons.values():
            button.hide()
            self.voice_group.removeButton(button)
            self.voice_grid.removeWidget(button)
            button.deleteLater()
        self.voice_buttons.clear()
        if self.kokoro_model is not None:
            from ..ai.kokoro_tts import unavailable_reason
            reason = unavailable_reason(self.kokoro_model, self.kokoro_python)
            for voice, name in (
                ("female", "女声 / Female"),
                ("xiaoxiao", "晓晓 / Xiaoxiao"),
                ("male", "男声 / Male"),
            ):
                key = f"kokoro:{voice}"
                button = QPushButton(name)
                button.setProperty("role", "preset")
                button.setCheckable(True)
                button.setMinimumHeight(44)
                button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
                button.setEnabled(reason is None)
                button.setToolTip(reason or "Kokoro 中文预置音色 / Kokoro Chinese preset")
                button.clicked.connect(lambda checked=False, selected=key: self._select_voice(selected))
                self.voice_group.addButton(button)
                self.voice_buttons[key] = button
        if self.selected_voice not in self.voice_buttons or not self.voice_buttons[self.selected_voice].isEnabled():
            self.selected_voice = next((
                key for key, button in self.voice_buttons.items() if button.isEnabled()
            ), None)
        if self.selected_voice:
            self.voice_buttons[self.selected_voice].setChecked(True)
        self._layout_voice_buttons()
        self._set_text(self.voice_hint, (
            "Kokoro 预置音色无需参考录音。 / Kokoro presets need no reference recording."
            if self.kokoro_model is not None else
            "尚未安装真实音色资源。 / Real voice resources are not installed."
        ))
        self.refresh_model()
        self.update_controls()

    def _layout_voice_buttons(self) -> None:
        _layout_voice_grid(self.voice_grid, self.voice_buttons, self.width())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_voice_buttons()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Fonts are final only after the first polish, so size the editor here too.
        self._fit_text_input()

    def _fit_text_input(self) -> None:
        """Size the text editor for about four lines plus the frame/margins."""
        spacing = self.text_input.fontMetrics().lineSpacing()
        self.text_input.setFixedHeight(spacing * 4 + 28)

    def _select_voice(self, key: str) -> None:
        self.selected_voice = key
        self._message = ""
        self.refresh_model()
        self.update_controls()

    def _set_text(self, widget: QLabel, text: str) -> None:
        ancestor = self
        while ancestor is not None:
            language_filter = getattr(ancestor, "_language_filter", None)
            if language_filter is not None:
                language_filter.set_text(widget, text)
                return
            ancestor = ancestor.parentWidget()
        widget.setText(text)

    @property
    def busy(self) -> bool:
        return self.job.busy or self._saving

    def refresh_model(self) -> None:
        from ..ai.kokoro_tts import unavailable_reason
        reason = unavailable_reason(self.kokoro_model, self.kokoro_python)
        self.state = Availability("tts", reason is None, reason or "")
        self._set_text(self.availability_notice, (
            f"AI 朗读不可用 / AI read-aloud unavailable:\n{reason}" if reason else ""
        ))
        self.availability_notice.setVisible(reason is not None)

    @Slot()
    def update_controls(self, *_args) -> None:
        text = self.text_input.toPlainText().strip()
        controller = self.controller
        audio_idle = controller is None or (
            controller.is_ready and not controller.busy and not controller.owns_devices
            and controller.state is EngineState.STOPPED and not controller.media_status.recording
        )
        from ..ai.kokoro_tts import TEXT_LIMIT
        ready = (self.state.available and self.selected_voice is not None and bool(text)
                 and len(text) <= TEXT_LIMIT)
        self.generate_button.setEnabled(ready and not self.busy and audio_idle)
        self.play_button.setEnabled(bool(self.job.result) and not self.busy and audio_idle and controller is not None)
        playing = controller is not None and controller.media_status.playback == "generated"
        can_stop_audio = controller is not None and not controller.busy and (
            playing or controller.owns_devices or controller.state is EngineState.FAULT
        )
        self.stop_button.setEnabled(self.job.busy or can_stop_audio)
        self.save_button.setEnabled(bool(self.job.result) and not self.busy and audio_idle)
        for widget in (self.devices_button, self.text_input):
            widget.setEnabled(not self.busy and audio_idle)
        from ..ai.kokoro_tts import unavailable_reason
        usable = unavailable_reason(self.kokoro_model, self.kokoro_python) is None
        for button in self.voice_buttons.values():
            button.setEnabled(not self.busy and audio_idle and usable)
        if self.busy:
            message = self.job.message if self.job.busy else "正在保存… / Saving…"
        elif len(text) > TEXT_LIMIT:
            message = f"Kokoro 每次最多 {TEXT_LIMIT} 字 / Kokoro accepts up to {TEXT_LIMIT} characters"
        elif self._message:
            message = self._message
        elif not self.state.available:
            message = "状态：不可用（未接入模型） / Status: unavailable (no model)"
        elif self.selected_voice is None:
            message = "状态：请选择可用音色 / Status: choose a usable voice"
        elif not text:
            message = "状态：请输入文字 / Status: enter text"
        else:
            message = "状态：待机 / Status: idle"
        if playing:
            message = "正在播放朗读结果 / Playing generated speech"
        self._set_text(self.status, message)

    def generate(self) -> None:
        self.update_controls()
        if not self.generate_button.isEnabled():
            self._message = "请先选择音色和文字，并停止其他音频 / Choose a voice and text; stop other audio"
            self.update_controls()
            return
        self.refresh_model()
        if not self.state.available:
            self.update_controls()
            return
        self._message = ""
        try:
            if self.selected_voice is None or self.kokoro_model is None:
                raise RuntimeError("Kokoro voice or model is no longer available")
            self.job.start_kokoro(
                self.selected_voice.split(":", 1)[1], self.text_input.toPlainText(),
                self.kokoro_model, python=self.kokoro_python,
            )
        except (OSError, RuntimeError) as error:
            self._message = f"无法启动生成 / Cannot start generation: {error}"
            self.update_controls()

    def _job_changed(self) -> None:
        if not self.job.busy:
            self._message = self.job.message
            if self.job.result:
                self._set_text(self.result,
                               f"{self.job.frames / self.job.sample_rate:.2f}s · {self.job.sample_rate} Hz · WAV")
        self.update_controls()
        if not self.busy:
            self.idle.emit()

    def _audio_error(self, error) -> None:
        if self.isVisible():
            self._message = f"音频操作失败 / Audio operation failed: {error}"
            self.update_controls()

    def play(self) -> None:
        self.update_controls()
        if not self.play_button.isEnabled():
            self._message = "请先生成音频并停止其他音频 / Generate speech and stop other audio first"
            self.update_controls()
            return
        if self.controller is not None and self.job.result is not None:
            try:
                ceiling = self.output_ceiling()
            except ValueError as error:
                self._audio_error(error)
                return
            self.controller.request_play_generated(
                self.job.result, self.job.sample_rate, output_ceiling=ceiling, output_confirmed=True,
            )

    def stop(self) -> None:
        if self.job.busy:
            self.job.cancel()
        elif self.controller is not None and (
            self.controller.media_status.playback == "generated" or self.controller.owns_devices
            or self.controller.state is EngineState.FAULT
        ):
            self.controller.request_stop_all()

    def save(self) -> None:
        if not self.save_button.isEnabled():
            self._message = "当前不能保存音频 / Audio cannot be saved right now"
            self.update_controls()
            return
        destination, _ = QFileDialog.getSaveFileName(
            self, "保存朗读音频 / Save generated speech", "speech.wav", "WAV (*.wav)",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not destination:
            return
        path = Path(destination)
        if path.suffix.lower() != ".wav":
            path = path.with_suffix(".wav")
        source = self.job.result
        if source is None:
            self._message = "没有可保存的音频 / No generated audio to save"
            self.update_controls()
            return
        self._saving = True
        self.update_controls()

        def copy() -> Path:
            # Exclusive create never overwrites an existing recording or symlink.
            with path.open("xb") as target:
                try:
                    with source.open("rb") as reader:
                        shutil.copyfileobj(reader, target)
                    target.flush()
                    os.fsync(target.fileno())
                except OSError:
                    path.unlink()
                    raise
            return path

        run_background(self, copy, self._saved)

    def _saved(self, result, error) -> None:
        self._saving = False
        self._message = (
            f"保存失败，请选择新文件名 / Save failed; choose a new filename: {error}"
            if error else f"音频已保存 / Audio saved:\n{result}"
        )
        self.update_controls()
        self.idle.emit()


class AiVcPage(QWidget):
    devicesRequested = Signal()

    _MODEL_LABELS = {
        "llvc_onnx": "LLVC ONNX · CPU",
        "llvc_rknn": "LLVC RKNN / NPU · 未验收 / unaccepted",
        "hevc_onnx": "HEVoicechanger ONNX · 实验 / experimental · 非实时 / not realtime",
        "hevc_rknn": "HEVoicechanger RKNN / NPU · 实时（实验） / live (experimental)",
        "pytorch_checkpoint": "PyTorch .pth · 格式未验证 / format unverified",
        "rvc_onnx_synthesis": "RVC synthesis ONNX / CPU · 原型 / prototype",
        "rvc_onnx_content": "RVC content ONNX component · prototype",
        "rvc_onnx_pitch": "RVC pitch ONNX component · prototype",
        "rvc_rknn": "RVC RKNN / NPU · 未验收 / unaccepted",
    }

    def __init__(self, controller: "EngineController", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._last_error: str | None = None
        self._beatrice_reason: str | None = None
        self.state = availability("vc")
        from ..ai.llvc import configured_model

        configured = configured_model()
        self._configured_model = configured.resolve() if configured is not None else None
        self._model_inspections: dict[Path, ModelInspection] = {}
        self._model_check_pending = False
        self._selected_model: Path | None = self._configured_model
        self._model_buttons: dict[Path, QPushButton] = {}
        self._name_overrides, name_problem = model_catalog.load_name_overrides()
        if name_problem is not None:
            self._last_error = name_problem
        self._settings = QSettings("voice-changer", "vc-models")
        stored_paths = self._settings.value("paths", [], type=list)
        self._model_paths: list[str] = []
        if self._settings.status() != QSettings.Status.NoError:
            self._last_error = "无法读取模型列表 / Cannot read saved model list"
        layout, footer = _scroll_page(self)
        page_heading(
            layout, "AI 变声 / AI voice conversion",
            "检查模型和设备，再开始实时变声。 / Check the model and devices, then start voice conversion.",
        )
        self.availability_notice = _availability_notice(self.state)
        layout.addWidget(self.availability_notice)
        model_panel, model_layout = new_panel()
        model_layout.addWidget(_label("变声模型 / Voice conversion model", role="sectionHeading"))
        model_row = QHBoxLayout()
        model_row.addStretch()
        self.model_button = QPushButton("添加本地模型… / Add local model…")
        self.model_button.clicked.connect(self.choose_local_model)
        model_row.addWidget(self.model_button)
        self.rename_button = QPushButton("重命名… / Rename…")
        self.rename_button.clicked.connect(self.rename_model)
        model_row.addWidget(self.rename_button)
        self._preview_busy = False
        self._preview_note: str | None = None
        self.preview_button = QPushButton("试听（离线）… / Try offline…")
        self.preview_button.setToolTip(
            "录 5 秒，整段离线转换，播放并保存；不是实时变声。\n"
            "HEVC 实验模型或 Beatrice v2 的 JVS 候选声音都走这个离线试听。\n"
            "Records 5 s, converts the whole clip offline, plays and saves it; not live conversion.\n"
            "Used by the experimental HEVC models and the Beatrice v2 JVS candidates."
        )
        self.preview_button.clicked.connect(self.preview_offline)
        model_row.addWidget(self.preview_button)
        model_layout.addLayout(model_row)
        self.model_grid = QGridLayout()
        self.model_grid.setSpacing(SPACING["sm"])
        self.model_group = QButtonGroup(self)
        self.model_group.setExclusive(True)
        model_layout.addLayout(self.model_grid)
        if self._configured_model is not None:
            self._add_model_button(self._configured_model)
        for item in stored_paths:
            if isinstance(item, str) and item and Path(item).is_absolute():
                if item not in self._model_paths:
                    self._model_paths.append(item)
                self._add_model_button(Path(item))
            else:
                self._last_error = "已保存的模型路径无效 / Saved model path is invalid"
        self.model_hint = _label(
            "选择已添加的模型，或添加本地文件；请只使用可信模型。 / "
            "Choose an added model or add a local file; use trusted models only.",
            role="muted",
        )
        model_layout.addWidget(self.model_hint)
        self.model_status = _label(name="模型识别状态 / Model identification status")
        model_layout.addWidget(self.model_status)
        # The model panel stays compact: identification details are not shown as
        # small print under the buttons; the availability notice above the panel
        # carries the honest reason when the selected model cannot be used.
        self.model_hint.hide()
        self.model_status.hide()
        self.model_error = _label(name="模型识别错误 / Model identification error")
        self.model_error.setProperty("i18nPreserveText", True)
        self.model_error.hide()
        model_layout.addWidget(self.model_error)
        layout.addWidget(model_panel)
        engine_panel, engine_layout = new_panel()
        engine_layout.addWidget(_label("实验引擎 / Experimental engine", role="sectionHeading"))
        engine_row = QHBoxLayout()
        self.beatrice_checkbox = QCheckBox(
            "启用 Beatrice v2（实验） / Enable Beatrice v2 (experimental)",
        )
        self.beatrice_checkbox.toggled.connect(self._on_beatrice_toggled)
        engine_row.addWidget(self.beatrice_checkbox, 1)
        engine_layout.addLayout(engine_row)
        # Speaker selection is button-based, mirroring the model picker: each
        # candidate voice is one clickable button instead of a number entry.
        self._beatrice_speaker = 0  # zero-based JVS index used by the engine
        self.beatrice_voices: dict[int, QPushButton] = {}
        voice_grid = QGridLayout()
        voice_grid.setHorizontalSpacing(SPACING["sm"])
        voice_grid.setVerticalSpacing(SPACING["sm"])
        for index, (text, speaker, _pitch_shift, _formant_shift,
                  _external_formant) in enumerate(_BEATRICE_VOICES):
            voice_button = QPushButton(text)
            voice_button.setCheckable(True)
            voice_button.setProperty("role", "preset")
            voice_button.setMinimumHeight(44)
            voice_button.setToolTip(_BEATRICE_VOICE_TIP)
            voice_button.clicked.connect(partial(self._select_beatrice_voice, speaker))
            voice_grid.addWidget(voice_button, index // 2, index % 2)
            self.beatrice_voices[speaker] = voice_button
        self.beatrice_custom = QPushButton("自定义编号… / Custom index…")
        self.beatrice_custom.setProperty("role", "preset")
        self.beatrice_custom.setMinimumHeight(44)
        self.beatrice_custom.setToolTip(
            "手动选择 JVS 说话人序号 1–100；不是参考录音克隆。\n"
            "Choose a JVS speaker index 1–100 manually; not reference-recording cloning."
        )
        self.beatrice_custom.clicked.connect(self._choose_beatrice_custom)
        voice_grid.addWidget(
            self.beatrice_custom,
            len(_BEATRICE_VOICES) // 2,
            len(_BEATRICE_VOICES) % 2,
        )
        engine_layout.addLayout(voice_grid)
        self._select_beatrice_voice(_BEATRICE_VOICES[0][1], refresh=False)
        layout.addWidget(engine_panel)
        devices_panel, devices_layout = new_panel()
        device_heading = QHBoxLayout()
        device_heading.addWidget(_label("音频设备 / Audio devices", role="sectionHeading"))
        device_heading.addStretch()
        self.devices_button = QPushButton("选择设备 / Choose devices")
        self.devices_button.clicked.connect(self.devicesRequested)
        device_heading.addWidget(self.devices_button)
        devices_layout.addLayout(device_heading)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        self.input_label = _label(name="AI 变声输入设备 / AI conversion input")
        self.output_label = _label(name="AI 变声输出设备 / AI conversion output")
        form.addRow("输入 / Input", self.input_label)
        form.addRow("输出 / Output", self.output_label)
        devices_layout.addLayout(form)
        devices_layout.addWidget(_label("设备在“DSP 变声”页选择 / Choose devices on the DSP page", role="muted"))
        layout.addWidget(devices_panel)
        self.levels = _label(
            "电平：未运行，无读数 / Levels: not running, no reading",
            name="AI 变声电平 / AI conversion levels",
        )
        self.latency = _label(
            "端到端延迟：尚未测量；暂不设合格线，须在 Pi M2 实测\n"
            "End-to-end latency: not measured; no pass threshold yet, to be measured on Pi M2",
            role="muted", name="AI 变声延迟 / AI conversion latency",
        )
        self.status = _label(name="AI 变声状态 / AI conversion status")
        level_panel, level_layout = new_panel()
        level_layout.addWidget(_label("输入与输出 / Input and output", role="sectionHeading"))
        level_layout.addWidget(self.levels)
        level_layout.addWidget(self.latency)
        layout.addWidget(level_panel)
        footer_status(footer).addWidget(self.status)
        layout.addStretch(1)
        actions = QHBoxLayout()
        self.start_button = QPushButton("开启 AI 变声 / Start AI conversion")
        self.start_button.setProperty("role", "primary")
        self.stop_button = QPushButton("停止 / Stop")
        actions.addWidget(self.start_button, 1)
        actions.addStretch()
        actions.addWidget(self.stop_button)
        footer.addLayout(actions)
        self.start_button.clicked.connect(self.start_conversion)
        self.stop_button.clicked.connect(self.stop_conversion)
        controller.deviceSelectionChanged.connect(self.refresh_devices)
        controller.stateChanged.connect(self.update_controls)
        controller.operationFinished.connect(self.update_controls)
        controller.errorOccurred.connect(self.show_error)
        controller.liveStatus.connect(self.update_levels)
        self._update_model_status()
        self.refresh_devices()
        if self._configured_model is not None and self.state.available:
            self._inspect_configured_model()

    @Slot()
    def refresh_devices(self) -> None:
        selection = self.controller.device_selection
        for label, ref in (
            (self.input_label, selection.input_device),
            (self.output_label, selection.output_device),
        ):
            label.setProperty("i18nPreserveText", ref is not None)
            label.setText(ref.name if ref is not None else "未选择 / Not selected")
        self.update_controls()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_models()

    def _layout_models(self) -> None:
        columns = 3 if self.width() >= 700 else 2
        for index, button in enumerate(self._model_buttons.values()):
            self.model_grid.addWidget(button, index // columns, index % columns)
        for column in range(3):
            self.model_grid.setColumnStretch(column, int(column < columns))

    def _model_display_label(self, path: Path, info: ModelInspection | None = None) -> str:
        sha = info.sha256 if info is not None else None
        model_type = info.model_type if info is not None else None
        entry = model_catalog.find_entry(path, sha, model_type)
        if path == self._configured_model and entry is None:
            if sha is not None and f"sha:{sha}" in self._name_overrides:
                return self._name_overrides[f"sha:{sha}"]
            return model_catalog.configured_label(self._name_overrides)
        return model_catalog.display_label(path, sha, model_type, self._name_overrides)

    def _model_can_rename(self) -> bool:
        info = self._selected_model_info()
        if info is None or self._model_check_pending:
            return False
        model = self._selected_model_path()
        return (model_catalog.find_entry(model, info.sha256, info.model_type) is not None
                or bool(info.sha256))

    def _add_model_button(self, path: Path) -> None:
        if path in self._model_buttons:
            return
        text = self._model_display_label(path, self._model_inspections.get(path))
        button = QPushButton(text)
        button.setProperty("role", "preset")
        button.setCheckable(True)
        button.setMinimumHeight(44)
        button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        button.setToolTip(str(path))
        button.clicked.connect(lambda checked=False, model=path: self._inspect_model(model))
        self.model_group.addButton(button)
        self._model_buttons[path] = button
        button.setChecked(path == self._selected_model)
        self._layout_models()

    def _selected_model_path(self) -> Path | None:
        return self._selected_model

    def _selected_model_info(self) -> ModelInspection | None:
        model = self._selected_model_path()
        return self._model_inspections.get(model) if model is not None else None

    def _model_usable(self) -> bool:
        model = self._selected_model_path()
        info = self._selected_model_info()
        if (self._model_check_pending or model is None or info is None
                or info.model_type not in ("llvc_onnx", "hevc_rknn")
                or not model.is_file() or model.is_symlink()):
            return False
        if info.model_type == "hevc_rknn":
            return hevc_rknn.unavailable_reason(model) is None
        if model == self._configured_model:
            return self.state.available
        return importlib.util.find_spec("onnxruntime") is not None

    def _update_model_status(self, error: BaseException | None = None) -> None:
        model = self._selected_model_path()
        self.model_error.setText(str(error) if error is not None else "")
        self.model_error.setVisible(error is not None)
        if error is not None:
            self.model_status.setText(
                "模型识别失败 / Model identification failed"
            )
            set_visual_state(self.model_status, "fault")
        elif self._model_check_pending:
            self.model_status.setText(
                "正在后台识别文件格式与模型接口；不会运行推理 / "
                "Identifying file format and model interface in background; no inference"
            )
            set_visual_state(self.model_status, "")
        elif model is None:
            self.model_status.setText(
                "选择模型文件后将自动识别类型；未知或未验收的模型不会启动。\n"
                "Select a model file for automatic type detection; unknown or unaccepted models cannot start."
            )
            set_visual_state(self.model_status, "")
        else:
            info = self._selected_model_info()
            if info is None:
                self.model_status.setText(
                    "尚未识别该模型 / Model has not been identified"
                )
                set_visual_state(self.model_status, "fault")
            else:
                details = info.description
                if info.model_type == "llvc_onnx":
                    delay = 52 if info.frames == 832 else 104
                    details = (
                        f"已识别：{self._MODEL_LABELS[info.model_type]} · {delay} ms\n\n"
                        f"{details}"
                    )
                    if not self._model_usable():
                        details += "\nONNX Runtime 不可用或模型文件已改变 / Runtime unavailable or file changed."
                    set_visual_state(self.model_status, "" if self._model_usable() else "fault")
                elif info.model_type == "hevc_rknn":
                    reason = hevc_rknn.unavailable_reason(model)
                    details = (
                        f"已识别：{self._MODEL_LABELS[info.model_type]} · "
                        f"{hevc_rknn.WINDOW_SAMPLES} 样本窗口 / sample window\n{details}"
                    )
                    if reason is None:
                        details += f"\n\n{hevc_rknn.LIVE_NOTE}"
                        set_visual_state(self.model_status, "")
                    else:
                        details += f"\n\n{reason}"
                        set_visual_state(self.model_status, "fault")
                else:
                    details = (
                        f"已识别：{self._MODEL_LABELS.get(info.model_type, info.model_type)}\n"
                        f"{details}\n仅识别类型，不会加载或推理 / Identified only; not loaded or executed."
                    )
                    if info.model_type == "hevc_onnx":
                        entry = model_catalog.find_entry(model, info.sha256, info.model_type)
                        if entry is not None and entry.live_reason:
                            details = f"{details}\n\n{entry.live_reason}"
                    set_visual_state(self.model_status, "fault")
                self.model_status.setText(details)
        self.model_status.setToolTip(str(model) if model is not None else "")

    @Slot()
    def choose_local_model(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件 / Choose model file",
            "",
            "Supported models (*.onnx *.rknn *.pth);;All files (*)",
        )
        if not selected:
            return
        self._inspect_model(Path(selected).expanduser())

    def _inspect_model(self, model: Path, *, initial: bool = False) -> None:
        if self._model_check_pending or (
            not initial and (
                self.controller.busy or self.controller.owns_devices
                or self.controller.state is not EngineState.STOPPED
            )
        ):
            return
        self._selected_model = model
        self._model_inspections.pop(model, None)
        self._model_check_pending = True
        self._update_model_status()
        self.update_controls()
        from ..ai.model_inspection import inspect_model_file

        run_background(
            self,
            lambda: (model, inspect_model_file(model)),
            lambda result, error: self._local_model_checked(result, error),
        )

    def _local_model_checked(self, result, error: BaseException | None) -> None:
        self._model_check_pending = False
        if error is not None:
            self._update_model_status(error)
        else:
            model, inspection = result
            model = model.resolve()
            self._selected_model = model
            self._model_inspections[model] = inspection
            self._add_model_button(model)
            button = self._model_buttons.get(model)
            if button is not None:
                button.setText(self._model_display_label(model, inspection))
            if model != self._configured_model and str(model) not in self._model_paths:
                self._model_paths.append(str(model))
                self._settings.setValue("paths", self._model_paths)
                self._settings.sync()
                if self._settings.status() != QSettings.Status.NoError:
                    self._last_error = "模型列表保存失败 / Could not save model list"
            self._update_model_status()
        self.update_controls()

    @Slot()
    def rename_model(self) -> None:
        model = self._selected_model_path()
        info = self._selected_model_info()
        if model is None or info is None or not self._model_can_rename():
            return
        current = self._model_display_label(model, info)
        label, accepted = QInputDialog.getText(
            self, "重命名模型 / Rename model",
            "显示名称，留空恢复默认；名称仅为界面标签，不代表音色已验收。\n"
            "Display name; empty restores the default. This is a UI label only, "
            "not proof of the voice.",
            QLineEdit.EchoMode.Normal, current,
        )
        if not accepted:
            return
        try:
            new_label = model_catalog.rename(
                model, info.sha256, info.model_type, label, self._name_overrides,
            )
        except ValueError as error:
            self._last_error = str(error)
            self._update_model_status(error)
            self.update_controls()
            return
        button = self._model_buttons.get(model)
        if button is not None:
            button.setText(new_label)
        self.update_controls()

    @Slot()
    def preview_offline(self) -> None:
        """Record a short clip, convert it offline, play it once and save it.

        Two honest preview paths share this flow: an experimental HEVC ONNX
        model (identified but not realtime) and the Beatrice v2 engine, whose
        JVS speakers include female and male candidates. Neither path pretends
        to be the live conversion.
        """
        from ..ai import beatrice_preview, hevc_preview

        if self._preview_busy or self._model_check_pending:
            return
        if (self.controller.busy or self.controller.owns_devices
                or self.controller.state is not EngineState.STOPPED):
            return
        selection = self.controller.device_selection
        if selection.input_device is None or selection.output_device is None:
            self.status.setText(
                "状态：请先在 DSP 页选择输入和输出设备 / "
                "Status: choose input/output on the DSP page first"
            )
            return
        engine = self._beatrice_engine()
        model = self._selected_model_path()
        info = self._selected_model_info()
        engine_preview = engine is not None
        if engine_preview:
            voice = self._beatrice_voice_label()
            voice_en = self._beatrice_voice_label_en()
            title = "离线试听 / Offline preview"
            question = (
                f"将录制约 {beatrice_preview.PREVIEW_SECONDS} 秒（从所选麦克风），"
                f"用 Beatrice v2 的 {voice} 整段离线转换，"
                "播放一次并保存为 WAV。\n"
                "这是离线转换，不是实时变声；实时链路约 0.92 倍实时，可能中断。\n"
                "模型与 API 仅限非商业用途（JVS 条款）。\n"
                f"Records about {beatrice_preview.PREVIEW_SECONDS} s, converts the whole clip "
                f"offline with {voice_en}, plays it once and saves a WAV. "
                "This is offline conversion, not live conversion. The model and API are "
                "non-commercial only (JVS terms)."
            )
        elif model is not None and info is not None and info.model_type == "hevc_onnx":
            title = "离线试听 / Offline preview"
            question = (
                "将录制约 5 秒（从所选麦克风），整段转换后播放一次，并保存为 WAV。\n"
                "这是离线转换，不是实时变声；处理时间取决于时长，期间请等待。\n"
                "Records about 5 s from the selected mic, converts the whole clip, plays it once and "
                "saves a WAV. This is offline conversion, not live conversion."
            )
        else:
            return
        answer = QMessageBox.question(
            self, title, question,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        input_index = selection.input_device.index
        output_index = selection.output_device.index
        self._preview_busy = True
        self._preview_note = None
        self.update_controls()

        if engine_preview:
            engine = engine  # bind the selected speaker for the worker thread

            def work():
                samples = hevc_preview.record(
                    hevc_preview.PREVIEW_SECONDS, device=input_index,
                )
                converted = beatrice_preview.convert(
                    samples, beatrice_preview.HOST_RATE, engine,
                )
                saved = hevc_preview.save_wav(
                    beatrice_preview.preview_file_name(engine), converted,
                )
                hevc_preview.play(converted, device=output_index)
                return saved, converted.size / beatrice_preview.HOST_RATE

        else:

            def work():
                samples = hevc_preview.record(
                    hevc_preview.PREVIEW_SECONDS, device=input_index,
                )
                converted = hevc_preview.convert(samples, hevc_preview.HOST_RATE, model)
                saved = hevc_preview.save_wav(
                    hevc_preview.preview_file_name(model), converted,
                )
                hevc_preview.play(converted, device=output_index)
                return saved, converted.size / hevc_preview.HOST_RATE

        run_background(self, work, self._preview_done)

    def _preview_done(self, result, error: BaseException | None) -> None:
        self._preview_busy = False
        if error is not None:
            self._last_error = str(error)
            self.status.setText(f"离线试听失败 / Offline preview failed: {error}")
            set_visual_state(self.status, "fault")
        else:
            saved, seconds = result
            self._preview_note = (
                f"已保存离线试听（{seconds:.1f} 秒）：{saved} / "
                f"Offline preview saved ({seconds:.1f} s)"
            )
            QMessageBox.information(
                self, "离线试听完成 / Offline preview finished",
                "已播放一次，并保存为 WAV。这是离线转换，不是实时变声。\n"
                "Played once and saved as WAV. This is offline conversion, not live conversion.\n\n"
                f"已保存：{saved}\nSaved: {saved}",
            )
        self.update_controls()

    def _inspect_configured_model(self) -> None:
        self._inspect_model(self._configured_model, initial=True)

    def _unavailable_reason(self, info: ModelInspection | None) -> str:
        if info is not None and info.model_type == "hevc_rknn":
            return hevc_rknn.unavailable_reason(self._selected_model_path())
        if info is not None and info.model_type == "hevc_onnx":
            entry = model_catalog.find_entry(
                self._selected_model_path(), info.sha256, info.model_type,
            )
            if entry is not None and entry.live_reason:
                return entry.live_reason
            return hevc.LIVE_REASON
        return info.description if info is not None else self.state.reason

    @Slot()
    def update_controls(self) -> None:
        controller = self.controller
        selection = controller.device_selection
        stopped = controller.state is EngineState.STOPPED
        idle = controller.is_ready and not controller.busy
        info = self._selected_model_info()
        model_usable = not self._model_check_pending and self._model_usable()
        hevc_selected = info is not None and info.model_type == "hevc_onnx"
        self._beatrice_reason = beatrice.unavailable_reason()
        beatrice_ready = self._beatrice_reason is None
        if not beatrice_ready and self.beatrice_checkbox.isChecked():
            blocked = self.beatrice_checkbox.blockSignals(True)
            self.beatrice_checkbox.setChecked(False)
            self.beatrice_checkbox.blockSignals(blocked)
        beatrice_enabled = beatrice_ready and self.beatrice_checkbox.isChecked()
        devices_chosen = (
            selection.input_device is not None and selection.output_device is not None
        )
        ready = (model_usable or beatrice_enabled) and devices_chosen
        editable = (
            idle and stopped and not controller.owns_devices
            and not self._model_check_pending and not self._preview_busy
        )
        self.beatrice_checkbox.setEnabled(editable and beatrice_ready)
        for voice_button in self.beatrice_voices.values():
            voice_button.setEnabled(editable and beatrice_enabled)
        self.beatrice_custom.setEnabled(editable and beatrice_enabled)
        # The experimental engine panel deliberately keeps only the checkbox,
        # the voice buttons and the custom index: availability details live in
        # the checkbox tooltip instead of a small-print status paragraph.
        self.beatrice_checkbox.setToolTip(self._beatrice_reason or "")
        self.availability_notice.setVisible(not model_usable and not beatrice_enabled)
        if not model_usable and not beatrice_enabled:
            self.availability_notice.setText(self._unavailable_reason(info) or (
                "请选择模型文件以自动识别 / Select a model file for automatic identification"
            ))
        self.model_group.setExclusive(False)
        for path, button in self._model_buttons.items():
            button.setEnabled(editable and not beatrice_enabled)
            button.setChecked(path == self._selected_model)
        self.model_group.setExclusive(True)
        self.model_button.setEnabled(editable and not beatrice_enabled)
        self.rename_button.setEnabled(
            editable and not beatrice_enabled and self._model_can_rename()
        )
        self.preview_button.setEnabled(
            editable and devices_chosen and (beatrice_enabled or hevc_selected)
        )
        self.start_button.setEnabled(editable and ready)
        self.stop_button.setEnabled(
            idle and (controller.owns_devices or controller.state is EngineState.FAULT)
        )
        if beatrice_enabled:
            if self._preview_busy:
                text = (
                    "状态：离线试听中（录制 → 转换 → 播放），不是实时变声 / "
                    "Status: offline preview running (record → convert → play); not live conversion"
                )
            elif controller.busy:
                text = (
                    "状态：正在启动实验引擎（加载 Beatrice 模型），请稍候 / "
                    "Status: starting the experimental engine (loading the Beatrice model)…"
                )
            elif controller.state is EngineState.RUNNING:
                text = (
                    "状态：AI 实时变声运行中（Beatrice v2 实验，"
                    f"{self._beatrice_voice_label()}） / Status: AI conversion running "
                    f"(Beatrice v2 experimental, {self._beatrice_voice_label_en()})"
                )
            elif controller.state is EngineState.FAULT:
                text = (
                    "状态：已停止，请检查错误并点击停止重置 / Status: fault; Stop to reset\n"
                    f"{self._last_error or ''}"
                )
            elif not devices_chosen:
                text = "状态：请先在 DSP 页选择输入和输出设备 / Status: choose input/output on DSP page"
            else:
                text = (
                    "状态：未启动（Beatrice v2 实验） / "
                    "Status: not started (Beatrice v2 experimental)"
                )
                if self._preview_note:
                    text += f"\n{self._preview_note}"
        elif not model_usable:
            if info is not None and info.model_type == "hevc_rknn":
                text = (
                    "状态：NPU 实验链路不可用 / "
                    "Status: HEVC RKNN / NPU experimental path unavailable\n"
                    f"{self._unavailable_reason(info)}"
                )
            elif hevc_selected and self._preview_busy:
                text = (
                    "状态：离线试听中（录制 → 转换 → 播放），不是实时变声 / "
                    "Status: offline preview running (record → convert → play); not live conversion"
                )
            elif hevc_selected:
                text = (
                    "状态：实验模型·非实时，未开放启动；可试听（离线） / "
                    "Status: experimental, not realtime; start disabled, offline preview available"
                )
                if self._preview_note:
                    text += f"\n{self._preview_note}"
            elif info is not None and info.model_type != "llvc_onnx":
                text = (
                    "状态：已识别，但此模型类型尚未接入已验收实时链 / "
                    "Status: identified, but this model type is not accepted in the live path"
                )
            else:
                text = (
                    "状态：请选择可用的 LLVC ONNX 模型 / "
                    "Status: select a usable LLVC ONNX model"
                )
        elif controller.busy:
            text = "状态：正在加载模型或释放设备，请稍候 / Status: loading or releasing devices…"
        elif controller.state is EngineState.RUNNING:
            if info is not None and info.model_type == "hevc_rknn":
                text = (
                    "状态：NPU 实验实时变声运行中 / NPU live conversion running (experimental)"
                )
            else:
                text = "状态：AI 实时变声运行中 / Status: AI conversion running"
        elif controller.state is EngineState.FAULT:
            text = (
                "状态：已停止，请检查错误并点击停止重置 / Status: fault; Stop to reset\n"
                f"{self._last_error or ''}"
            )
        elif not ready:
            text = "状态：请先在 DSP 页选择输入和输出设备 / Status: choose input/output on DSP page"
        elif info is not None and info.model_type == "hevc_rknn":
            text = (
                "状态：未启动（NPU 实验） / Status: not started (NPU experimental)"
            )
        else:
            text = "状态：未启动 / Status: not started"
        if self._last_error and controller.state is not EngineState.FAULT:
            text += f"\n{self._last_error}"
        self.status.setText(text)
        set_visual_state(self.status, "fault" if controller.state is EngineState.FAULT else "")
        if controller.state is not EngineState.RUNNING:
            self.levels.setText("电平：未运行，无读数 / Levels: not running, no reading")

    def _beatrice_voice_params(self) -> tuple[float, float, float]:
        """Return the default (pitch shift, formant shift, external formant)."""
        return _BEATRICE_VOICE_PARAMS.get(
            self._beatrice_speaker + 1, (0.0, 0.0, 1.0))

    def _beatrice_voice_label(self) -> str:
        """Name the selected voice, including its default compensation."""
        speaker = self._beatrice_speaker + 1
        pitch_shift, formant_shift, external_formant = self._beatrice_voice_params()
        details = []
        if pitch_shift:
            details.append(f"升调 {pitch_shift:+.1f} 半音")
        if formant_shift:
            details.append(f"模型共振峰 {formant_shift:+.1f} 半音")
        if external_formant != 1.0:
            details.append(f"女声共振峰补偿 ×{external_formant:.2f}")
        if details:
            return f"JVS 说话人 {speaker}（{'、'.join(details)}）"
        return f"JVS 说话人 {speaker}"

    def _beatrice_voice_label_en(self) -> str:
        speaker = self._beatrice_speaker + 1
        pitch_shift, formant_shift, external_formant = self._beatrice_voice_params()
        details = []
        if pitch_shift:
            details.append(f"pitch {pitch_shift:+.1f} st")
        if formant_shift:
            details.append(f"model formant {formant_shift:+.1f} st")
        if external_formant != 1.0:
            details.append(f"female formant compensation x{external_formant:.2f}")
        if details:
            return f"JVS speaker {speaker} ({', '.join(details)})"
        return f"JVS speaker {speaker}"

    def _select_beatrice_voice(self, speaker: int, *, refresh: bool = True) -> None:
        """Select one JVS speaker; the candidate buttons behave like a group."""
        speaker = max(1, min(beatrice.SPEAKER_COUNT, int(speaker)))
        self._beatrice_speaker = speaker - 1
        for value, voice_button in self.beatrice_voices.items():
            voice_button.setChecked(value == speaker)
        if speaker in self.beatrice_voices:
            self.beatrice_custom.setText("自定义编号… / Custom index…")
        else:
            self.beatrice_custom.setText(
                f"自定义编号 JVS {speaker}… / Custom index JVS {speaker}…"
            )
        if refresh:
            self.update_controls()

    @Slot()
    def _choose_beatrice_custom(self) -> None:
        value, accepted = QInputDialog.getInt(
            self,
            "自定义说话人 / Custom speaker",
            f"JVS 说话人序号（1–{beatrice.SPEAKER_COUNT}）；不是参考录音克隆。\n"
            f"JVS speaker index (1–{beatrice.SPEAKER_COUNT}); not reference-recording cloning.",
            self._beatrice_speaker + 1,
            1,
            beatrice.SPEAKER_COUNT,
        )
        if accepted:
            self._select_beatrice_voice(value)

    @Slot(bool)
    def _on_beatrice_toggled(self, _checked: bool) -> None:
        self.update_controls()

    def _beatrice_engine(self) -> beatrice.BeatriceConfig | None:
        """Return the selected experimental engine configuration, or None."""
        if not self.beatrice_checkbox.isChecked():
            return None
        config = beatrice.configured()
        if config is None or beatrice.unavailable_reason(config) is not None:
            return None
        pitch_shift, formant_shift, external_formant = self._beatrice_voice_params()
        return config.with_voice(
            self._beatrice_speaker, pitch_shift, formant_shift, external_formant)

    def _build_stream_config(self):
        from ..app.device_selection import DeviceSelectionError

        try:
            return self.controller.device_selection.build_stream_config(
                samplerate=48000, blocksize=1024, channels=1, output_ceiling=0.5,
            )
        except (ValueError, DeviceSelectionError) as error:
            self.status.setText(f"设备配置失败 / Device configuration failed: {error}")
            return None

    @Slot()
    def stop_conversion(self) -> None:
        # A bool-returning slot connected straight to clicked(bool) segfaults in PySide's result conversion.
        self.controller.request_stop()

    @Slot()
    def start_conversion(self) -> None:
        from ..ai.llvc import LlvcProcessorFactory
        from ..dsp.parameters import EffectParameters

        model = self._selected_model_path()
        engine = self._beatrice_engine()
        if engine is None and (model is None or not self._model_usable()):
            self.status.setText(
                "请选择可用的 LLVC ONNX 或 HEVC RKNN 模型，或配置并启用 Beatrice 实验引擎 / "
                "Select a usable LLVC ONNX or HEVC RKNN model, or configure and enable Beatrice"
            )
            return
        info = self._selected_model_info()
        rknn_model = info is not None and info.model_type == "hevc_rknn"
        config = None
        if engine is not None:
            config = self._build_stream_config()
            if config is None:
                return
        if config is None:
            config = self._build_stream_config()
            if config is None:
                return
        self._last_error = None
        if engine is not None:
            factory = beatrice.BeatriceProcessorFactory(engine)
        elif rknn_model:
            factory = hevc_rknn.HevcRknnProcessorFactory(model)
        else:
            factory = LlvcProcessorFactory(
                model, allow_custom_model=model != self._configured_model,
            )
        self.controller.request_listen(
            config, EffectParameters(), output_confirmed=True, processor_factory=factory,
        )
        self.update_controls()

    @staticmethod
    def _friendly_error(text: str) -> str:
        """Explain the experimental engine's most common stop in plain words."""
        if "underrun" in text.lower():
            return (
                "Beatrice 实验引擎欠载：板端 CPU 转换速度不足以维持实时，已停止。"
                "请改用 LLVC 实时链路，或关闭实验引擎。\n"
                "Beatrice experimental engine underran (slower than real time on this board); "
                "stopped. Use LLVC for real-time conversion."
            )
        return text

    @Slot(object)
    def show_error(self, error) -> None:
        self._last_error = self._friendly_error(str(error))
        self.status.setText(f"AI 变声失败 / AI conversion failed: {self._last_error}")
        set_visual_state(self.status, "fault")

    @Slot(object)
    def update_levels(self, status) -> None:
        levels = status[0]
        if levels is not None:
            self.levels.setText(
                f"输入峰值 / Input: {levels[0]:.3f} · 输出峰值 / Output: {levels[1]:.3f}"
            )
