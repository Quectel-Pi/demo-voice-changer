"""Bounded audio comparison widgets shared by the desktop and silent prototype."""

from collections import deque
import math
from numbers import Real
from time import monotonic

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..audio.analysis_status import AnalysisReading
from .theme import COLORS, SIZES, set_visual_state


WINDOW_SECONDS = 3.0
MAX_POINTS = 60
MAX_PITCH_POINTS = 30
UPDATE_INTERVAL_MS = 50
REPAINT_INTERVAL_MS = 35
NO_CAPTURE = "未采集 / No capture"
WAITING = "等待音频 / Waiting for audio"
INVALID = "电平异常 / Invalid level"
ANALYSIS_INVALID = "分析数据异常 / Invalid analysis data"

_ANALYSIS_TEXT = {
    "bilingual": {
        "bands": "频段 RMS / Bands",
        "band_names": ("低 Low", "中 Mid", "高 High"),
        "pitch": "音高 / Pitch",
        "unvoiced": "音高未识别 / Pitch unvoiced",
        "silent": "音高未识别（静音）/ Pitch unvoiced (silence)",
        "failure": "分析异常 / Analysis failed",
        "dropped": "分析跳帧 {count} / Analysis skipped {count} blocks",
        "nyquist": "频率最高至 / Frequency limited to {hz} Hz",
    },
    "zh": {
        "bands": "频段 RMS",
        "band_names": ("低", "中", "高"),
        "pitch": "音高",
        "unvoiced": "音高未识别",
        "silent": "音高未识别（静音）",
        "failure": "分析异常",
        "dropped": "分析跳帧 {count}",
        "nyquist": "频率最高至 {hz} Hz",
    },
    "en": {
        "bands": "Band RMS",
        "band_names": ("Low", "Mid", "High"),
        "pitch": "Pitch",
        "unvoiced": "Pitch unvoiced",
        "silent": "Pitch unvoiced (silence)",
        "failure": "Analysis failed",
        "dropped": "Analysis skipped {count} blocks",
        "nyquist": "Frequency limited to {hz} Hz",
    },
}


def _height_for_peak(peak: float) -> float:
    if peak == 0:
        return 0.0
    return max(0.0, min(1.0, (20 * math.log10(peak) + 60) / 60))


class _EnvelopeCanvas(QWidget):
    def __init__(self, history: deque[tuple[float, float]], parent: QWidget) -> None:
        super().__init__(parent)
        self._history = history
        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor(COLORS["line"]), 1))
        painter.setBrush(QColor(COLORS["surfaceRaised"]))
        painter.drawRoundedRect(rect, SIZES["meterRadius"], SIZES["meterRadius"])
        plot = rect.adjusted(8, 5, -8, -5)
        if plot.width() <= 0 or plot.height() <= 0:
            return
        middle = plot.center().y()
        painter.setPen(QPen(QColor(COLORS["muted"]), 1))
        painter.drawLine(QPointF(plot.left(), middle), QPointF(plot.right(), middle))
        if not self._history:
            return

        latest = self._history[-1][0]
        points = [
            (
                plot.right() - min(WINDOW_SECONDS, latest - timestamp) * plot.width() / WINDOW_SECONDS,
                _height_for_peak(peak) * plot.height() / 2,
            )
            for timestamp, peak in self._history
        ]
        painter.setPen(QPen(QColor(COLORS["active"]), 2))
        for x, height in points:
            if height > 0:
                painter.drawLine(QPointF(x, middle - height), QPointF(x, middle + height))


class _AnalysisCanvas(QWidget):
    def __init__(
        self, history: deque[tuple[float, float | None]], language: str, parent: QWidget,
    ) -> None:
        super().__init__(parent)
        self._history = history
        self._language = language
        self.reading: AnalysisReading | None = None
        self._compact = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.set_compact(False)

    def set_compact(self, compact: bool) -> None:
        self._compact = compact
        font = self.font()
        font.setPointSizeF(9)
        line = QFontMetrics(font).height()
        self.setFixedHeight(line * 4 + 45 if compact else line * 3 + 35)
        self.update()

    def set_language(self, language: str) -> None:
        if language not in _ANALYSIS_TEXT:
            raise ValueError("Unsupported analysis display language")
        self._language = language
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = painter.font()
        font.setPointSizeF(9)
        painter.setFont(font)
        line = painter.fontMetrics().height()
        width = float(self.width())
        track_height = 16.0 if self._compact else 30.0
        if self._compact:
            bands = QRectF(0, 0, width, line * 3 + track_height + 3)
            pitch = QRectF(0, bands.bottom() + 2, width, self.height() - bands.bottom() - 2)
        else:
            split = max(0.0, width / 2 - 4)
            bands = QRectF(0, 0, split, self.height())
            pitch = QRectF(split + 8, 0, max(0.0, width - split - 8), self.height())
        labels = _ANALYSIS_TEXT[self._language]
        painter.setPen(QColor(COLORS["text"]))
        painter.drawText(
            QRectF(bands.left(), bands.top(), bands.width(), line),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f'{labels["bands"]} · dBFS',
        )
        values = self.reading.bands if self.reading is not None else None
        column = bands.width() / 3
        track_top = bands.top() + line + 2
        for index, name in enumerate(labels["band_names"]):
            x = bands.left() + column * index
            track_width = min(25.0, max(0.0, column - 12))
            track = QRectF(x + (column - track_width) / 2, track_top, track_width, track_height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLORS["meterTrack"]))
            painter.drawRoundedRect(track, 2, 2)
            value = values[index] if values is not None else None
            if value is not None:
                height = _height_for_peak(value) * track_height
                if height > 0:
                    painter.setBrush(QColor(COLORS["active"]))
                    painter.drawRoundedRect(
                        QRectF(track.left(), track.bottom() - height, track.width(), height), 2, 2,
                    )
                text = f"{20 * math.log10(value):.1f}" if value > 0 else "−∞"
            else:
                text = "N/A" if values is not None else "—"
            painter.setPen(QColor(COLORS["muted"]))
            painter.drawText(
                QRectF(x, track.bottom() + 1, column, line),
                Qt.AlignmentFlag.AlignCenter, name,
            )
            painter.drawText(
                QRectF(x, track.bottom() + line + 1, column, line),
                Qt.AlignmentFlag.AlignCenter, text,
            )

        painter.setPen(QColor(COLORS["text"]))
        painter.drawText(
            QRectF(pitch.left(), pitch.top(), pitch.width(), line),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f'{labels["pitch"]} · 50–800 Hz',
        )
        plot = pitch.adjusted(0, line + 3, -1, -3)
        if plot.width() <= 0 or plot.height() <= 0:
            return
        painter.setPen(QPen(QColor(COLORS["line"]), 1))
        painter.setBrush(QColor(COLORS["surfaceRaised"]))
        painter.drawRoundedRect(plot, SIZES["meterRadius"], SIZES["meterRadius"])
        painter.setPen(QPen(QColor(COLORS["muted"]), 1))
        painter.drawLine(QPointF(plot.left() + 4, plot.center().y()), QPointF(plot.right() - 4, plot.center().y()))
        if not self._history:
            return
        area = plot.adjusted(4, 4, -4, -4)
        latest = self._history[-1][0]
        previous = None
        painter.setPen(QPen(QColor(COLORS["active"]), 2))
        for timestamp, pitch_hz in self._history:
            if pitch_hz is None:
                previous = None
                continue
            x = area.right() - min(WINDOW_SECONDS, latest - timestamp) * area.width() / WINDOW_SECONDS
            position = math.log2(pitch_hz / 50) / math.log2(800 / 50)
            point = QPointF(x, area.bottom() - position * area.height())
            painter.drawPoint(point)
            if previous is not None:
                painter.drawLine(previous, point)
            previous = point


class _RepaintGate:
    def __init__(self, owner: QWidget) -> None:
        self.owner = owner
        self._last_request: float | None = None
        self._timer = QTimer(owner)
        self._timer.setSingleShot(True)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._repaint)

    def request(self) -> None:
        if not self.owner.isVisible() or self._timer.isActive():
            return
        elapsed = (
            REPAINT_INTERVAL_MS / 1000 if self._last_request is None
            else monotonic() - self._last_request
        )
        self._timer.start(max(0, math.ceil(REPAINT_INTERVAL_MS - elapsed * 1000)))

    def _repaint(self) -> None:
        if self.owner.isVisible():
            self._last_request = monotonic()
            self.owner.update()

    def stop(self) -> None:
        self._timer.stop()
        self._last_request = None


class WaveComparison(QWidget):
    """Reorient the same two widgets without hiding or reparenting them."""

    def __init__(self, first: "AudioWave", second: "AudioWave") -> None:
        super().__init__()
        self._waves = (first, second)
        self._layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, self)
        self._layout.setContentsMargins(0, 4, 0, 4)
        self._layout.setSpacing(10)
        self._redraw_gate = _RepaintGate(self)
        for wave in self._waves:
            self._layout.addWidget(wave, 1)
            wave._redraw_gate = self._redraw_gate
        self.side_by_side = False

    def set_side_by_side(self, enabled: bool) -> None:
        if enabled == self.side_by_side:
            return
        self.side_by_side = enabled
        self._layout.setDirection(
            QBoxLayout.Direction.LeftToRight if enabled else QBoxLayout.Direction.TopToBottom,
        )
        for wave in self._waves:
            wave.set_compact(enabled)

    def hideEvent(self, event) -> None:
        self._redraw_gate.stop()
        super().hideEvent(event)


class _ReadingLabel(QLabel):
    """Reserve stable width for the range of finite floating-point dBFS readings."""

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        metrics = self.fontMetrics()
        margins = self.contentsMargins()
        numeric_width = (
            5 * max(metrics.horizontalAdvance(str(digit)) for digit in range(10))
            + metrics.horizontalAdvance("-. dBFS") + 8
        )
        width = max(
            numeric_width,
            *(metrics.horizontalAdvance(line) for line in self.text().splitlines()),
        )
        width += margins.left() + margins.right() + 2 * self.margin() + 1
        return QSize(max(hint.width(), width), hint.height())


class AudioWave(QWidget):
    """Display coalesced analysis and peaks; no audio samples or devices are owned."""

    def __init__(self, name: str, *, idle_text: str = NO_CAPTURE, language: str = "bilingual") -> None:
        super().__init__()
        if language not in _ANALYSIS_TEXT:
            raise ValueError("Unsupported analysis display language")
        self._language = language
        self._history: deque[tuple[float, float]] = deque(maxlen=MAX_POINTS)
        self._pitch_history: deque[tuple[float, float | None]] = deque(maxlen=MAX_PITCH_POINTS)
        self._peak: float | None = None
        self._peak_invalid = False
        self._pending_peak: tuple[float, float] | None = None
        self._pending_analysis: tuple[float, AnalysisReading] | None = None
        self._analysis: AnalysisReading | None = None
        self._analysis_state = idle_text
        self._analysis_notes: tuple[str, ...] = ()
        self._last_analysis_paint: float | None = None
        self._redraw_gate: _RepaintGate | None = None
        self._activity_description = ""
        self._idle_text = idle_text
        self.name_label = QLabel()
        self.name_label.setProperty("role", "muted")
        self.name_label.setWordWrap(True)
        self.reading_label = _ReadingLabel(idle_text)
        self.reading_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.reading_label.setWordWrap(True)
        self.canvas = _EnvelopeCanvas(self._history, self)
        self.analysis_canvas = _AnalysisCanvas(self._pitch_history, language, self)
        self.analysis_canvas.setToolTip(
            "低 20–250 Hz；中 250–2000 Hz；高 2000–8000 Hz。"
            "频段为 RMS，统一 −60 至 0 dBFS；音高为 50–800 Hz 对数尺度，"
            "未识别处留空，时间左旧右新。\n"
            "Low 20–250 Hz; mid 250–2000 Hz; high 2000–8000 Hz. "
            "Bands are RMS on the same −60 to 0 dBFS scale. Pitch uses a logarithmic "
            "50–800 Hz scale; unvoiced gaps are left blank. Oldest left, newest right."
        )
        self.analysis_status = QLabel(idle_text)
        self.analysis_status.setTextFormat(Qt.TextFormat.PlainText)
        self.analysis_status.setProperty("role", "muted")
        self.analysis_status.setWordWrap(True)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addWidget(self.name_label, 1)
        header.addWidget(self.reading_label)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addLayout(header)
        layout.addWidget(self.canvas)
        layout.addWidget(self.analysis_canvas)
        layout.addWidget(self.analysis_status)
        self._timer = QTimer(self)
        self._timer.setInterval(UPDATE_INTERVAL_MS)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._sample)
        self._analysis_timer = QTimer(self)
        self._analysis_timer.setSingleShot(True)
        self._analysis_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._analysis_timer.timeout.connect(self._sample_analysis)
        self.set_name(name)

    @property
    def peak(self) -> float | None:
        return self._peak

    @property
    def history(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._history)

    @property
    def pitch_history(self) -> tuple[tuple[float, float | None], ...]:
        return tuple(self._pitch_history)

    @property
    def display_text(self) -> str:
        return self.reading_label.text()

    def set_name(self, name: str) -> None:
        local, english = name.rsplit(" / ", 1) if " / " in name else (name, name)
        self.name_label.setText(name)
        self.setAccessibleName(name)
        self.canvas.setAccessibleName(name)
        self.reading_label.setAccessibleName(f"{local} · 电平 / {english} · Level")
        self.analysis_canvas.setAccessibleName(
            f"{local} · 频带与音高 / {english} · Bands and pitch"
        )
        self.analysis_status.setAccessibleName(
            f"{local} · 分析状态 / {english} · Analysis status"
        )
        self._update_accessibility()

    def set_compact(self, compact: bool) -> None:
        self.analysis_canvas.set_compact(compact)

    def set_activity_description(self, description: str) -> None:
        if description != self._activity_description:
            self._activity_description = description
            self._update_accessibility()

    def _update_accessibility(self) -> None:
        description = " · ".join(filter(None, (self._activity_description, self.display_text)))
        self.setAccessibleDescription(description)
        self.canvas.setAccessibleDescription(description)
        analysis = self._analysis
        band_text = ""
        if analysis is not None and analysis.bands is not None:
            parts = []
            for name, value in zip(_ANALYSIS_TEXT[self._language]["band_names"], analysis.bands):
                reading = "N/A" if value is None else (
                    f"{20 * math.log10(value):.1f} dBFS" if value > 0 else "−∞ dBFS"
                )
                parts.append(f"{name}: {reading}")
            band_text = "; ".join(parts)
        self.analysis_canvas.setAccessibleDescription(
            " · ".join(filter(None, (self._activity_description, band_text, self.analysis_status.text()))),
        )

    def _set_text(self, text: str) -> None:
        if self.reading_label.text() != text:
            self.reading_label.setText(text)
            self._update_accessibility()

    def _clear(self) -> None:
        self._timer.stop()
        self._peak = None
        self._pending_peak = None
        if self._history:
            self._history.clear()
            self.canvas.update()

    def _clear_analysis(self) -> None:
        self._analysis_timer.stop()
        self._pending_analysis = None
        self._analysis = None
        self._last_analysis_paint = None
        if self._pitch_history or self.analysis_canvas.reading is not None:
            self._pitch_history.clear()
            self.analysis_canvas.reading = None
            self.analysis_canvas.update()

    def _set_analysis_status(self, text: str, *, error: bool = False) -> None:
        self._analysis_state = text
        self.analysis_status.setText(" · ".join((text, *self._analysis_notes)))
        set_visual_state(self.analysis_status, "fault" if error else "")
        self._update_accessibility()

    def set_idle(self, text: str | None = None) -> None:
        self._clear()
        self._clear_analysis()
        self._peak_invalid = False
        self._analysis_notes = ()
        idle = self._idle_text if text is None else text
        self._set_text(idle)
        self._set_analysis_status(idle)

    def set_waiting(self) -> None:
        self._clear()
        self._clear_analysis()
        self._peak_invalid = False
        self._analysis_notes = ()
        self._set_text(WAITING)
        self._set_analysis_status(WAITING)

    def set_analysis_waiting(self) -> None:
        if self._peak_invalid:
            return
        self._clear_analysis()
        self._set_analysis_status(WAITING)

    def set_analysis_info(self, sample_rate: int, dropped_blocks: int) -> None:
        labels = _ANALYSIS_TEXT[self._language]
        notes = []
        if dropped_blocks:
            notes.append(labels["dropped"].format(count=dropped_blocks))
        if sample_rate < 16000:
            notes.append(labels["nyquist"].format(hz=f"{sample_rate / 2:g}"))
        self._analysis_notes = tuple(notes)
        self._set_analysis_status(
            self._analysis_state,
            error=self._analysis is not None and self._analysis.error is not None,
        )

    def set_analysis(self, reading: AnalysisReading) -> None:
        if self._peak_invalid:
            return
        if reading.error is not None:
            self._clear_analysis()
            self._analysis = reading
            self._set_analysis_status(f'{_ANALYSIS_TEXT[self._language]["failure"]}: {reading.error}', error=True)
            return
        valid_bands = reading.bands is not None and len(reading.bands) == 3 and all(
            value is None or (
                not isinstance(value, bool) and isinstance(value, Real)
                and math.isfinite(value) and value >= 0
            )
            for value in reading.bands
        )
        valid_pitch = reading.pitch_hz is None or (
            not isinstance(reading.pitch_hz, bool) and isinstance(reading.pitch_hz, Real)
            and math.isfinite(reading.pitch_hz) and 50 <= reading.pitch_hz <= 800
        )
        if not valid_bands or not valid_pitch or type(reading.silent) is not bool:
            self._clear_analysis()
            self._analysis = AnalysisReading(None, None, error=ANALYSIS_INVALID)
            self._set_analysis_status(ANALYSIS_INVALID, error=True)
            return
        self._analysis = reading
        self._pending_analysis = (monotonic(), reading)
        labels = _ANALYSIS_TEXT[self._language]
        if reading.pitch_hz is not None:
            text = f'{labels["pitch"]}: {reading.pitch_hz:.1f} Hz'
        else:
            text = labels["silent"] if reading.silent else labels["unvoiced"]
        self._set_analysis_status(text)
        if self.isVisible() and not self._timer.isActive():
            self._sample()
            self._timer.start()
        elif self.isVisible():
            self._schedule_analysis()

    def set_peak(self, peak: float | None) -> None:
        if peak is None:
            self._clear()
            if self._peak_invalid:
                self._peak_invalid = False
                self.set_analysis_waiting()
            self._set_text(WAITING)
            return
        if isinstance(peak, bool) or not isinstance(peak, Real) or not math.isfinite(peak) or peak < 0:
            self._clear()
            self._clear_analysis()
            self._peak_invalid = True
            self._analysis = AnalysisReading(None, None, error=ANALYSIS_INVALID)
            self._set_analysis_status(ANALYSIS_INVALID, error=True)
            self._set_text(INVALID)
            return
        if self._peak_invalid:
            self._peak_invalid = False
            self.set_analysis_waiting()
        self._peak = float(peak)
        self._pending_peak = (monotonic(), self._peak)
        value = f"{20 * math.log10(self._peak):.1f}" if self._peak > 0 else "−∞"
        self._set_text(f"{value} dBFS")
        if self.isVisible() and not self._timer.isActive():
            self._sample()
            self._timer.start()

    def _sample(self) -> None:
        if self._pending_peak is not None:
            timestamp, peak = self._pending_peak
            self._pending_peak = None
            self._history.append((timestamp, peak))
            cutoff = timestamp - WINDOW_SECONDS
            while self._history and self._history[0][0] < cutoff:
                self._history.popleft()
            self._request_redraw()
        self._sample_analysis()

    def _request_redraw(self) -> None:
        if not self.isVisible():
            return
        if self._redraw_gate is None:
            self._redraw_gate = _RepaintGate(self)
        self._redraw_gate.request()

    def _schedule_analysis(self) -> None:
        if self._pending_analysis is None or self._analysis_timer.isActive():
            return
        elapsed = (
            0.1 if self._last_analysis_paint is None
            else monotonic() - self._last_analysis_paint
        )
        self._analysis_timer.start(max(0, math.ceil((0.1 - elapsed) * 1000)))

    def _sample_analysis(self) -> None:
        if self._pending_analysis is None:
            return
        now = monotonic()
        if self._last_analysis_paint is not None and now - self._last_analysis_paint < 0.1:
            if self.isVisible():
                self._schedule_analysis()
            return
        self._analysis_timer.stop()
        timestamp, reading = self._pending_analysis
        self._pending_analysis = None
        self._pitch_history.append((timestamp, reading.pitch_hz))
        cutoff = timestamp - WINDOW_SECONDS
        while self._pitch_history and self._pitch_history[0][0] < cutoff:
            self._pitch_history.popleft()
        self.analysis_canvas.reading = reading
        self._last_analysis_paint = now
        self._request_redraw()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if (self._peak is not None or self._pending_analysis is not None) and not self._timer.isActive():
            self._sample()
            self._timer.start()

    def hideEvent(self, event) -> None:
        self._timer.stop()
        self._analysis_timer.stop()
        if self._redraw_gate is not None and self._redraw_gate.owner is self:
            self._redraw_gate.stop()
        self._history.clear()
        self._pitch_history.clear()
        self.analysis_canvas.reading = None
        self._pending_peak = None
        self._pending_analysis = None
        self._last_analysis_paint = None
        super().hideEvent(event)
