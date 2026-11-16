#!/usr/bin/env python3
"""Opt-in Qt paint-event observer; it does not start or control audio."""

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import os
import sys
import time

from PySide6.QtCore import QEvent, QMetaObject, QObject, QPoint, QThread, QTimer, Qt, Signal, qVersion
from PySide6.QtGui import QKeySequence, QRegion, QShortcut
from PySide6.QtWidgets import QApplication, QProgressBar, QWidget


class MeasurementError(Exception):
    """The selected widgets cannot be measured safely."""


def _gui_application() -> QApplication:
    app = QApplication.instance()
    if not isinstance(app, QApplication) or QThread.currentThread() != app.thread():
        raise MeasurementError("The paint observer must be used on the Qt GUI thread")
    return app


def current_targets(window: object) -> dict[str, QWidget]:
    try:
        input_level = getattr(window, "input_level")
        output_level = getattr(window, "output_level")
        targets = {
            "input_envelope": input_level.canvas,
            "input_analysis": input_level.analysis_canvas,
            "output_envelope": output_level.canvas,
            "output_analysis": output_level.analysis_canvas,
        }
    except AttributeError as error:
        raise MeasurementError(
            "Current-version canvas widgets were not found; verify the application version"
        ) from error
    if any(not isinstance(widget, QWidget) for widget in targets.values()):
        raise MeasurementError("A selected current-version canvas is not a QWidget")
    return targets


def legacy_targets(window: object) -> dict[str, QWidget]:
    input_level = getattr(window, "input_level", None)
    output_level = getattr(window, "output_level", None)
    if not isinstance(input_level, QProgressBar) or not isinstance(output_level, QProgressBar):
        raise MeasurementError(
            "Legacy input_level and output_level must both be QProgressBar widgets"
        )
    if input_level is output_level:
        raise MeasurementError("Legacy input and output level widgets must be distinct")
    return {"input_level": input_level, "output_level": output_level}


def request_close_after_failure(window: object) -> str | None:
    app = QApplication.instance()
    is_window = isinstance(window, QWidget)
    target = window if is_window else app
    if target is None:
        return "Cannot close after observer failure: no Qt application is available"
    try:
        queued = QMetaObject.invokeMethod(
            target, "close" if is_window else "quit", Qt.ConnectionType.QueuedConnection,
        )
    except (RuntimeError, TypeError) as error:
        return f"Cannot queue close after observer failure: {error}"
    return None if queued else "Cannot queue close after observer failure"


class PaintCounter(QObject):
    failed = Signal(str)

    def __init__(
        self,
        targets: Mapping[str, QWidget],
        *,
        interval_ms: int = 1000,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_samples: int = 3600,
    ) -> None:
        app = _gui_application()
        if (
            not targets or any(not isinstance(name, str) or not name for name in targets)
            or any(not isinstance(widget, QWidget) for widget in targets.values())
            or len({id(widget) for widget in targets.values()}) != len(targets)
            or not 1 <= interval_ms <= 60_000 or not 1 <= max_samples <= 3600
        ):
            raise MeasurementError("Invalid paint targets, sampling interval, or sample limit")
        if any(widget.thread() != app.thread() for widget in targets.values()):
            raise MeasurementError("All paint targets must belong to the GUI thread")
        super().__init__(app)
        self._targets = dict(targets)
        self._names = {widget: name for name, widget in self._targets.items()}
        self._target_classes = {
            name: widget.metaObject().className() for name, widget in self._targets.items()
        }
        self._destroyed: set[str] = set()
        self._destroy_callbacks = {}
        self._clock_ns = clock_ns
        self._interval_ms = interval_ms
        self._max_samples = max_samples
        self._state = "idle"
        self._issues: list[str] = []
        self._samples: list[dict] = []
        self._counts = {name: 0 for name in self._targets}
        self._hidden: set[str] = set()
        self._geometry_changed: set[str] = set()
        self._started_ns: int | None = None
        self._last_ns: int | None = None
        self._last_visibility: dict[str, dict[str, bool | int]] = {}
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._sample)
        for name, widget in self._targets.items():
            widget.installEventFilter(self)
            callback = lambda _object=None, name=name: self._target_destroyed(name)
            self._destroy_callbacks[name] = callback
            widget.destroyed.connect(callback)
        app.aboutToQuit.connect(self._about_to_quit)

    def _visibility(self) -> dict[str, dict[str, bool | int]]:
        result = {}
        for name, widget in self._targets.items():
            if name in self._destroyed:
                raise MeasurementError(f"Paint target was destroyed: {name}")
            visible = widget.isVisible()
            window = widget.window()
            position = widget.mapTo(window, QPoint(0, 0))
            result[name] = {
                "visible": visible,
                "fully_visible_to_window": (
                    visible and not widget.rect().isEmpty()
                    and QRegion(widget.rect()).subtracted(widget.visibleRegion()).isEmpty()
                ),
                "x": position.x(),
                "y": position.y(),
                "width": widget.width(),
                "height": widget.height(),
                "window_width": window.width(),
                "window_height": window.height(),
            }
        return result

    def start(self) -> None:
        _gui_application()
        if self._state != "idle":
            raise MeasurementError(f"Cannot start paint measurement in state {self._state}")
        visibility = self._visibility()
        not_visible = [
            name for name, state in visibility.items() if not state["fully_visible_to_window"]
        ]
        if not_visible:
            raise MeasurementError(f"Paint targets are not fully visible: {', '.join(not_visible)}")
        self._started_ns = self._last_ns = self._clock_ns()
        self._last_visibility = visibility
        self._state = "running"
        self._timer.start()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if self._state == "running":
            name = self._names.get(watched)
            if name is not None:
                if event.type() == QEvent.Type.Paint:
                    self._counts[name] += 1
                elif event.type() == QEvent.Type.Hide:
                    self._hidden.add(name)
                elif event.type() in (QEvent.Type.Move, QEvent.Type.Resize):
                    self._geometry_changed.add(name)
        return False

    def _sample(self) -> None:
        if self._state != "running":
            return
        try:
            now = self._clock_ns()
            if self._last_ns is None or now <= self._last_ns:
                raise MeasurementError("The paint measurement clock did not advance")
            if len(self._samples) >= self._max_samples:
                raise MeasurementError("The paint measurement sample limit was reached")
            visibility = self._visibility()
        except MeasurementError as error:
            self._abort(str(error))
            return
        elapsed = (now - self._last_ns) / 1_000_000_000
        late = elapsed > self._interval_ms * 1.5 / 1000
        hidden = sorted(self._hidden)
        geometry_changed = sorted(self._geometry_changed)
        if late and "A paint sampling interval was late" not in self._issues:
            self._issues.append("A paint sampling interval was late")
            self.failed.emit("A paint sampling interval was late")
        not_visible = [
            name for name in self._targets
            if not self._last_visibility[name]["fully_visible_to_window"]
            or not visibility[name]["fully_visible_to_window"]
            or name in self._hidden
        ]
        for name in not_visible:
            issue = f"Paint target was hidden or clipped: {name}"
            if issue not in self._issues:
                self._issues.append(issue)
                self.failed.emit(issue)
        geometry_fields = ("x", "y", "width", "height", "window_width", "window_height")
        for name in self._targets:
            if name in self._geometry_changed or any(
                self._last_visibility[name][field] != visibility[name][field]
                for field in geometry_fields
            ):
                issue = f"Paint target geometry changed: {name}"
                if issue not in self._issues:
                    self._issues.append(issue)
                    self.failed.emit(issue)
        self._samples.append({
            "start_monotonic_ns": self._last_ns,
            "end_monotonic_ns": now,
            "wall_seconds": elapsed,
            "paint_events": dict(self._counts),
            "visible_start": self._last_visibility,
            "visible_end": visibility,
            "hidden_during_interval": hidden,
            "geometry_changed_during_interval": geometry_changed,
            "late": late,
        })
        self._last_ns = now
        self._last_visibility = visibility
        self._counts = {name: 0 for name in self._targets}
        self._hidden.clear()
        self._geometry_changed.clear()

    def stop(self) -> None:
        _gui_application()
        if self._state != "running":
            raise MeasurementError(f"Cannot stop paint measurement in state {self._state}")
        self._timer.stop()
        if self._last_ns is not None and self._clock_ns() > self._last_ns:
            self._sample()
        if self._state == "running":
            self._state = "stopped"

    def _abort(self, reason: str) -> None:
        self._timer.stop()
        if reason not in self._issues:
            self._issues.append(reason)
        self._state = "failed"
        self.failed.emit(reason)

    def _target_destroyed(self, name: str) -> None:
        self._destroyed.add(name)
        if self._state == "running":
            self._abort(f"Paint target was destroyed: {name}")

    def _about_to_quit(self) -> None:
        if self._state == "running":
            self._abort("Application exited before paint measurement was stopped")

    def report(self) -> dict:
        issues = list(self._issues)
        if self._state == "idle":
            issues.append("Paint measurement was not started")
        elif self._state == "running":
            issues.append("Paint measurement is still running")
        elif not self._samples:
            issues.append("No complete paint observation interval")
        return {
            "kind": "qt_paint_events",
            "status": "complete" if self._state == "stopped" and not issues else "incomplete",
            "issues": issues,
            "pid": os.getpid(),
            "qt_version": qVersion(),
            "targets": list(self._targets),
            "target_classes": dict(self._target_classes),
            "sampling_interval_seconds": self._interval_ms / 1000,
            "start_monotonic_ns": self._started_ns,
            "end_monotonic_ns": self._last_ns,
            "samples": list(self._samples),
            "total_paint_events": {
                name: sum(sample["paint_events"][name] for sample in self._samples)
                for name in self._targets
            } if self._samples else None,
            "visibility_limit": (
                "Visibility is sampled at interval boundaries; external covering "
                "and changes between samples may not be detected."
            ),
            "interpretation": (
                "Paint events include window-system redraws; they are not W04 timer "
                "requests, equivalent frame rates, or audio latency."
            ),
        }

    def detach(self) -> None:
        app = _gui_application()
        if self._state == "running":
            raise MeasurementError("Stop paint measurement before detaching")
        self._timer.stop()
        app.aboutToQuit.disconnect(self._about_to_quit)
        for name, widget in self._targets.items():
            if name not in self._destroyed:
                widget.removeEventFilter(self)
                widget.destroyed.disconnect(self._destroy_callbacks[name])
        self.deleteLater()


class _Shortcuts(QObject):
    def __init__(self, window: QWidget, counter: PaintCounter) -> None:
        super().__init__(window)
        self._counter = counter
        self._start = QShortcut(QKeySequence("Ctrl+Shift+F8"), window)
        self._stop = QShortcut(QKeySequence("Ctrl+Shift+F9"), window)
        self._start.activated.connect(self._start_measurement)
        self._stop.activated.connect(self._stop_measurement)
        counter.failed.connect(self._show_failure)

    def _start_measurement(self) -> None:
        try:
            self._counter.start()
        except MeasurementError as error:
            self._show_failure(str(error))
            return
        print("Paint measurement started.", file=sys.stderr, flush=True)

    def _stop_measurement(self) -> None:
        try:
            self._counter.stop()
        except MeasurementError as error:
            self._show_failure(str(error))
            return
        print("Paint measurement stopped; report will be printed on exit.", file=sys.stderr, flush=True)

    @staticmethod
    def _show_failure(message: str) -> None:
        print(f"Paint measurement error: {message}", file=sys.stderr, flush=True)


class PaintSession:
    def __init__(self, *, legacy: bool = False) -> None:
        self._legacy = legacy
        self._observers: list[PaintCounter] = []
        self._shortcuts: list[_Shortcuts] = []
        self._failures: list[str] = []

    def attach(self, window: object) -> None:
        if self._failures:
            return
        try:
            _gui_application()
            if self._observers:
                raise MeasurementError("The window-ready callback was invoked more than once")
            if not isinstance(window, QWidget):
                raise MeasurementError("The selected window is not a QWidget")
            targets = legacy_targets(window) if self._legacy else current_targets(window)
            counter = PaintCounter(targets)
            self._shortcuts.append(_Shortcuts(window, counter))
            self._observers.append(counter)
        except MeasurementError as error:
            self._failures.append(str(error))
            print(f"Paint observer setup failed: {error}", file=sys.stderr, flush=True)
            close_error = request_close_after_failure(window)
            if close_error:
                self._failures.append(close_error)
                print(close_error, file=sys.stderr, flush=True)
            return
        print(
            f"Paint observer ready. PID {os.getpid()}. Ctrl+Shift+F8 starts; Ctrl+Shift+F9 stops. "
            "Close the window to print the report.",
            file=sys.stderr, flush=True,
        )

    def report(self) -> dict:
        failures = list(self._failures)
        if not self._observers and not failures:
            failures.append("Application exited before the paint observer was installed")
        if failures:
            return {
                "kind": "qt_paint_events", "status": "incomplete",
                "pid": os.getpid(), "issues": failures,
            }
        return self._observers[0].report()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch-real-app", action="store_true",
        help=(
            "Launch the real desktop app, including device discovery and possible shared "
            "XDG_RUNTIME_DIR/voice-changer/asound.conf writes; separate authorization is required."
        ),
    )
    args = parser.parse_args(argv)
    if not args.launch_real_app:
        parser.error("--launch-real-app is required to launch the desktop app")

    print(
        "将启动正式窗口并发现设备，可能写共享 ALSA 配置；须事先获得明确授权。 / "
        "Launching the real window may discover devices and write shared ALSA configuration; "
        "separate authorization is required.",
        file=sys.stderr, flush=True,
    )
    from voice_changer.app.main import main as app_main

    session = PaintSession()
    result = app_main([], window_ready=session.attach)
    report = session.report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return result if result else (0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
