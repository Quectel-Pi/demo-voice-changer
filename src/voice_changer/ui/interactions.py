"""Shared mouse-wheel ownership and keyboard-only focus reveal for Qt forms."""

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QSignalBlocker, QTimer, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QScrollArea, QSlider, QWidget,
)
from .theme import SIZES


class FormInteractions(QObject):
    def __init__(self, scroll: QScrollArea) -> None:
        super().__init__(scroll)
        self.scroll = scroll
        self.active: QWidget | None = None
        self._focus_target: QWidget | None = None
        self._focus_timer = QTimer(self)
        self._focus_timer.setSingleShot(True)
        self._focus_timer.timeout.connect(self._reveal_keyboard_focus)
        scroll.installEventFilter(self)

    def _parameter(self, widget: QObject) -> QWidget | None:
        while isinstance(widget, QWidget) and self.scroll.isAncestorOf(widget):
            if isinstance(widget, (PageScrollDoubleSpinBox, PageScrollSlider, PageScrollComboBox)):
                return widget
            widget = widget.parentWidget()
        return None

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        event_type = event.type()
        if watched is self.scroll and event_type in (QEvent.Type.Show, QEvent.Type.Hide):
            app = QApplication.instance()
            if app is not None:
                if event_type == QEvent.Type.Show:
                    app.installEventFilter(self)
                else:
                    app.removeEventFilter(self)
                    self.active = self._focus_target = None
                    self._focus_timer.stop()
            return super().eventFilter(watched, event)
        if event_type not in (
            QEvent.Type.MouseButtonPress, QEvent.Type.FocusIn,
            QEvent.Type.FocusOut, QEvent.Type.WindowDeactivate,
        ):
            return super().eventFilter(watched, event)
        if not isinstance(watched, QWidget):
            return super().eventFilter(watched, event)
        if event_type == QEvent.Type.MouseButtonPress:
            parameter = self._parameter(watched)
            # An ignored press propagates through the parameter's ancestors.
            if not (
                parameter is None and self.active is not None
                and watched.isAncestorOf(self.active)
                and self.active.rect().contains(
                    self.active.mapFromGlobal(event.globalPosition().toPoint()),
                )
            ):
                self.active = parameter
        elif event_type == QEvent.Type.FocusIn:
            self.active = self._parameter(watched) if event.reason() in (
                Qt.FocusReason.MouseFocusReason, Qt.FocusReason.TabFocusReason,
                Qt.FocusReason.BacktabFocusReason, Qt.FocusReason.PopupFocusReason,
            ) else None
            if (
                self.scroll.widget().isAncestorOf(watched)
                and event.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
            ):
                self._focus_target = watched
                self._focus_timer.start(0)
        elif event_type in (QEvent.Type.FocusOut, QEvent.Type.WindowDeactivate):
            if event_type == QEvent.Type.WindowDeactivate or (
                self._parameter(watched) is self.active
                and event.reason() != Qt.FocusReason.PopupFocusReason
            ):
                self.active = None
        return super().eventFilter(watched, event)

    def _reveal_keyboard_focus(self) -> None:
        widget = self._focus_target
        self._focus_target = None
        if widget is None or not widget.hasFocus() or not widget.isVisible():
            return
        self.scroll.ensureWidgetVisible(widget, 0, 0)
        viewport = self.scroll.viewport()
        bounds = widget.rect().translated(widget.mapTo(viewport, QPoint()))
        scrollbar = self.scroll.verticalScrollBar()
        if bounds.top() < 0:
            scrollbar.setValue(scrollbar.value() + bounds.top())
        elif bounds.bottom() >= viewport.height():
            scrollbar.setValue(scrollbar.value() + bounds.bottom() - viewport.height() + 1)


def _wheel_owner(control: QWidget, event: QWheelEvent) -> bool:
    parent = control.parentWidget()
    while parent is not None and not isinstance(parent, QScrollArea):
        parent = parent.parentWidget()
    if not isinstance(parent, QScrollArea):
        event.ignore()
        return False
    interactions = parent.findChild(FormInteractions)
    if (
        interactions is not None and interactions.active is control and control.hasFocus()
        and control.rect().contains(control.mapFromGlobal(event.globalPosition().toPoint()))
    ):
        return True
    viewport = parent.viewport()
    forwarded = QWheelEvent(
        QPointF(viewport.mapFromGlobal(event.globalPosition().toPoint())),
        event.globalPosition(), event.pixelDelta(), event.angleDelta(),
        event.buttons(), event.modifiers(), event.phase(), event.inverted(), event.source(),
    )
    QApplication.sendEvent(viewport, forwarded)
    event.accept()
    return False


class PageScrollDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # WheelFocus would synthesize MouseFocusReason before wheelEvent.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if _wheel_owner(self, event):
            super().wheelEvent(event)
            event.accept()


class PageScrollSlider(QSlider):
    def wheelEvent(self, event: QWheelEvent) -> None:
        if _wheel_owner(self, event):
            super().wheelEvent(event)
            event.accept()


def sync_parameter_slider(slider: PageScrollSlider, value: float, scale: int) -> None:
    with QSignalBlocker(slider):
        slider.setValue(round(value * scale))


def parameter_slider(
    editor: PageScrollDoubleSpinBox, scale: int, step: int = 1,
) -> PageScrollSlider:
    editor.setProperty("role", "parameterValue")
    editor.setFixedWidth(SIZES["parameterValueWidth"])
    slider = PageScrollSlider(Qt.Orientation.Horizontal)
    slider.setRange(round(editor.minimum() * scale), round(editor.maximum() * scale))
    slider.setSingleStep(step)
    slider.setPageStep(step * 10)
    slider.setAccessibleName(editor.accessibleName())
    sync_parameter_slider(slider, editor.value(), scale)
    slider.valueChanged.connect(lambda value: editor.setValue(value / scale))
    editor.valueChanged.connect(lambda value: sync_parameter_slider(slider, value, scale))
    return slider


class PageScrollComboBox(QComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if _wheel_owner(self, event):
            super().wheelEvent(event)
            event.accept()
