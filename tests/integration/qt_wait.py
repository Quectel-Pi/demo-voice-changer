"""Condition polling via a native Qt event loop so worker teardown can acquire the GIL."""

from collections.abc import Callable
import time

from PySide6.QtCore import QEventLoop, QTimer


def wait_until(condition: Callable[[], object], timeout: float = 2) -> bool:
    deadline = time.monotonic() + timeout
    loop = QEventLoop()
    wake = QTimer()
    wake.setSingleShot(True)
    wake.timeout.connect(loop.quit)
    while not condition() and time.monotonic() < deadline:
        wake.start(5)
        loop.exec()
    return bool(condition())


def ensure_visible_after_layout(area, widget, rounds: int = 20) -> None:
    """Scroll only after pending layout changes settle; slow boards may lag a fixed wait."""
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    def snapshot():
        return (area.widget().size(), area.verticalScrollBar().maximum(), widget.geometry())

    previous = None
    for _ in range(rounds):
        QApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
        QApplication.processEvents()
        current = snapshot()
        if current == previous:
            break
        previous = current
    area.ensureWidgetVisible(widget, 0, 0)
    QApplication.processEvents()
