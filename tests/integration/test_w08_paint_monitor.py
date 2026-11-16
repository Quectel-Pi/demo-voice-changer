"""Qt paint observer checks using offscreen synthetic widgets and no audio."""

import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QCoreApplication, QEvent, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QProgressBar, QVBoxLayout, QWidget

from voice_changer.audio.analysis_status import AnalysisReading
from voice_changer.ui.audio_wave import AudioWave, WaveComparison


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "w08_paint_monitor", ROOT / "scripts" / "w08_paint_monitor.py",
)
monitor = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = monitor
spec.loader.exec_module(monitor)


class FakeClock:
    def __init__(self):
        self.now_ns = 0

    def __call__(self):
        return self.now_ns


class W08PaintMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def bars(self):
        window = QWidget()
        layout = QVBoxLayout(window)
        first, second = QProgressBar(), QProgressBar()
        layout.addWidget(first)
        layout.addWidget(second)
        window.resize(400, 180)
        window.show()
        self.addCleanup(window.close)
        QTest.qWait(20)
        return window, first, second

    def counter(self, targets, **kwargs):
        counter = monitor.PaintCounter(targets, **kwargs)

        def detach():
            counter.detach()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

        self.addCleanup(detach)
        return counter

    def legacy_fixture(self, directory, *, entry=None, window=None):
        root = Path(directory)
        source = root / "src"
        package = source / "voice_changer"
        runtime = root / "runtime"
        runtime.mkdir(mode=0o700)
        for relative in (
            "__init__.py", "app/__init__.py", "audio/__init__.py", "ui/__init__.py",
        ):
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        (package / "audio" / "virtual_mic.py").write_text(
            "calls = 0\n"
            "def prepare():\n"
            "    global calls\n"
            "    calls += 1\n",
            encoding="utf-8",
        )
        if entry is None:
            entry = r'''
import sys

def main(argv=None, *, window_ready=None):
    assert "PySide6" not in sys.modules
    assert "_w08_legacy_paint_observer" not in sys.modules
    from ..audio import virtual_mic
    virtual_mic.prepare()
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    from ..ui.main_window import MainWindow
    app = QApplication(["synthetic-legacy"])
    window = MainWindow()
    window.show()
    if window_ready is not None:
        window_ready(window)
    result = app.exec()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert virtual_mic.calls == 1
    return result
'''
        if window is None:
            window = r'''
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QProgressBar, QVBoxLayout, QWidget

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.input_level = QProgressBar()
        self.output_level = QProgressBar()
        layout = QVBoxLayout(self)
        layout.addWidget(self.input_level)
        layout.addWidget(self.output_level)
        self.resize(400, 180)
        QTimer.singleShot(100, self.begin)
        QTimer.singleShot(2000, QApplication.instance().quit)

    def begin(self):
        self.activateWindow()
        self.setFocus()
        QTest.keyClick(
            self, Qt.Key.Key_F8,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        self.input_level.setValue(20)
        self.output_level.setValue(75)
        QTimer.singleShot(150, self.finish)

    def finish(self):
        QTest.keyClick(
            self, Qt.Key.Key_F9,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        self.close()
'''
        (package / "app" / "main.py").write_text(entry, encoding="utf-8")
        (package / "ui" / "main_window.py").write_text(window, encoding="utf-8")
        return source, runtime

    def run_legacy_fixture(self, source, runtime):
        environment = dict(
            os.environ, XDG_RUNTIME_DIR=str(runtime), QT_QPA_PLATFORM="offscreen",
            PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT / "src"),
        )
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts" / "w08_legacy_paint_monitor.py"),
             "--app-src", str(source), "--launch-real-app"],
            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=8,
        )

    def test_controlled_counts_are_independent_and_do_not_consume_events(self):
        _, first, second = self.bars()
        clock = FakeClock()
        counter = self.counter({"input": first, "output": second}, clock_ns=clock)
        counter.start()
        paint = QEvent(QEvent.Type.Paint)
        self.assertFalse(counter.eventFilter(first, paint))
        self.assertFalse(counter.eventFilter(first, paint))
        self.assertFalse(counter.eventFilter(second, paint))
        clock.now_ns = 1_000_000_000
        counter._sample()
        self.assertFalse(counter.eventFilter(second, paint))
        clock.now_ns = 2_000_000_000
        counter._sample()
        counter.stop()
        report = counter.report()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["pid"], os.getpid())
        self.assertEqual(report["start_monotonic_ns"], 0)
        self.assertEqual(report["end_monotonic_ns"], 2_000_000_000)
        self.assertEqual(report["samples"][0]["paint_events"], {"input": 2, "output": 1})
        self.assertEqual(report["samples"][1]["paint_events"], {"input": 0, "output": 1})
        self.assertEqual(report["total_paint_events"], {"input": 2, "output": 2})
        self.assertIn("not W04", report["interpretation"])

    def test_actual_paint_events_for_legacy_style_stub_widgets(self):
        _, first, second = self.bars()
        counter = self.counter(monitor.legacy_targets(
            SimpleNamespace(input_level=first, output_level=second),
        ))
        counter.start()
        first.setValue(20)
        second.setValue(75)
        QTest.qWait(120)
        counter.stop()
        report = counter.report()
        self.assertEqual(report["status"], "complete", report["issues"])
        self.assertEqual(report["target_classes"], {
            "input_level": "QProgressBar", "output_level": "QProgressBar",
        })
        self.assertGreater(report["total_paint_events"]["input_level"], 0)
        self.assertGreater(report["total_paint_events"]["output_level"], 0)

    def test_legacy_targets_require_two_distinct_progress_bars(self):
        _, first, second = self.bars()
        self.assertEqual(
            monitor.legacy_targets(SimpleNamespace(input_level=first, output_level=second)),
            {"input_level": first, "output_level": second},
        )
        for window in (
            SimpleNamespace(input_level=first),
            SimpleNamespace(input_level=first, output_level=QWidget()),
            SimpleNamespace(input_level=None, output_level=second),
        ):
            with self.subTest(window=window):
                with self.assertRaisesRegex(monitor.MeasurementError, "both be QProgressBar"):
                    monitor.legacy_targets(window)
        with self.assertRaisesRegex(monitor.MeasurementError, "distinct"):
            monitor.legacy_targets(SimpleNamespace(input_level=first, output_level=first))

    def test_timer_samples_real_events_without_manual_sampling(self):
        window, first, second = self.bars()
        counter = self.counter({"input": first, "output": second}, interval_ms=250)
        generator = QTimer(window)
        generator.setInterval(40)
        value = [0]

        def update():
            value[0] += 1
            first.setValue(value[0] % 100)
            second.setValue((value[0] * 2) % 100)

        generator.timeout.connect(update)
        counter.start()
        generator.start()
        QTest.qWait(650)
        generator.stop()
        counter.stop()
        report = counter.report()
        self.assertEqual(report["status"], "complete", report["issues"])
        self.assertGreaterEqual(len(report["samples"]), 3)
        self.assertTrue(all(
            sample["paint_events"]["input"] > 0 and sample["paint_events"]["output"] > 0
            for sample in report["samples"][:2]
        ))

    def test_real_current_canvases_receive_separate_synthetic_updates(self):
        first = AudioWave("Synthetic input")
        second = AudioWave("Synthetic output")
        window = WaveComparison(first, second)
        window.resize(800, 500)
        window.show()
        self.addCleanup(window.close)
        QTest.qWait(30)
        targets = monitor.current_targets(SimpleNamespace(input_level=first, output_level=second))
        counter = self.counter(targets)
        counter.start()
        for index in range(3):
            for offset, wave in enumerate((first, second)):
                peak = 0.1 + index * 0.1 + offset * 0.05
                wave.set_peak(peak)
                wave.set_analysis(AnalysisReading((peak, peak, peak), 200 + index * 10))
            QTest.qWait(70)
        counter.stop()
        report = counter.report()
        self.assertEqual(report["status"], "complete", report["issues"])
        self.assertEqual(set(report["targets"]), {
            "input_envelope", "input_analysis", "output_envelope", "output_analysis",
        })
        self.assertTrue(all(value > 0 for value in report["total_paint_events"].values()))

    def test_not_visible_at_start_is_an_error_not_zero_data(self):
        _, first, second = self.bars()
        counter = self.counter({"input": first, "output": second})
        second.hide()
        with self.assertRaisesRegex(monitor.MeasurementError, "not fully visible"):
            counter.start()
        report = counter.report()
        self.assertEqual(report["status"], "incomplete")
        self.assertIsNone(report["total_paint_events"])
        self.assertIn("not started", report["issues"][0])

    def test_temporarily_hidden_target_invalidates_observation(self):
        _, first, second = self.bars()
        clock = FakeClock()
        counter = self.counter({"input": first, "output": second}, clock_ns=clock)
        counter.start()
        first.hide()
        first.show()
        self.app.processEvents()
        clock.now_ns = 1_000_000_000
        counter._sample()
        counter.stop()
        report = counter.report()
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("Paint target was hidden or clipped: input", report["issues"])
        self.assertIn("input", report["samples"][0]["hidden_during_interval"])

    def test_clipping_and_geometry_changes_are_reported(self):
        window = QWidget()
        window.resize(240, 100)
        first = QProgressBar(window)
        first.setGeometry(10, 10, 120, 35)
        window.show()
        self.addCleanup(window.close)
        QTest.qWait(20)
        clock = FakeClock()
        counter = self.counter({"input": first}, clock_ns=clock)
        counter.start()
        first.move(200, 10)
        self.app.processEvents()
        clock.now_ns = 1_000_000_000
        counter._sample()
        counter.stop()
        report = counter.report()
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("Paint target was hidden or clipped: input", report["issues"])
        self.assertIn("Paint target geometry changed: input", report["issues"])
        self.assertFalse(report["samples"][0]["visible_end"]["input"]["fully_visible_to_window"])
        self.assertEqual(report["samples"][0]["visible_start"]["input"]["window_width"], 240)

    def test_quitting_before_stop_is_incomplete(self):
        _, first, _ = self.bars()
        counter = self.counter({"input": first})
        counter.start()
        counter._about_to_quit()
        report = counter.report()
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("exited before", report["issues"][0])
        self.assertIsNone(report["total_paint_events"])

    def test_late_timer_and_clock_failure_are_explicit(self):
        _, first, second = self.bars()
        clock = FakeClock()
        late = self.counter({"input": first, "output": second}, clock_ns=clock)
        late.start()
        clock.now_ns = 2_000_000_000
        late._sample()
        late.stop()
        self.assertEqual(late.report()["status"], "incomplete")
        self.assertTrue(late.report()["samples"][0]["late"])

        stuck = self.counter({"input": first}, clock_ns=clock)
        stuck.start()
        stuck._sample()
        self.assertEqual(stuck.report()["status"], "incomplete")
        self.assertIn("clock did not advance", stuck.report()["issues"][0])
        self.assertIsNone(stuck.report()["total_paint_events"])

    def test_bounded_samples_and_destroyed_target_fail_visibly(self):
        _, first, second = self.bars()
        clock = FakeClock()
        bounded = self.counter({"input": first}, clock_ns=clock, max_samples=1)
        bounded.start()
        clock.now_ns = 1_000_000_000
        bounded._sample()
        clock.now_ns = 2_000_000_000
        bounded._sample()
        self.assertEqual(bounded.report()["status"], "incomplete")
        self.assertIn("sample limit", bounded.report()["issues"][0])
        self.assertEqual(len(bounded.report()["samples"]), 1)

        destroyed = self.counter({"output": second}, clock_ns=clock)
        destroyed.start()
        second.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertEqual(destroyed.report()["status"], "incomplete")
        self.assertIn("destroyed", destroyed.report()["issues"][0])

    def test_gui_thread_and_target_validation(self):
        _, first, second = self.bars()
        for targets in ({}, {"a": first, "b": first}, {"a": None}):
            with self.subTest(targets=targets):
                with self.assertRaises(monitor.MeasurementError):
                    monitor.PaintCounter(targets)
        with self.assertRaisesRegex(monitor.MeasurementError, "not found"):
            monitor.current_targets(SimpleNamespace(input_level=first, output_level=second))
        errors = []

        def from_worker():
            try:
                monitor.PaintCounter({"input": first})
            except monitor.MeasurementError as error:
                errors.append(str(error))

        worker = threading.Thread(target=from_worker)
        worker.start()
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn("GUI thread", errors[0])

    def test_cli_requires_explicit_real_app_flag(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts" / "w08_paint_monitor.py")],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--launch-real-app is required", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_legacy_cli_requires_explicit_flags_before_importing_an_app(self):
        script = str(ROOT / "scripts" / "w08_legacy_paint_monitor.py")
        for arguments, expected in (
            ([], "--launch-real-app is required"),
            (["--launch-real-app"], "--app-src is required"),
        ):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [sys.executable, "-B", script, *arguments],
                    cwd=ROOT, capture_output=True, text=True, timeout=5,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)
                self.assertEqual(result.stdout, "")

    def test_legacy_launcher_uses_only_fake_hooked_source_and_both_bars(self):
        with tempfile.TemporaryDirectory(prefix="voice-changer-legacy-fixture-") as directory:
            source, runtime = self.legacy_fixture(directory)
            result = self.run_legacy_fixture(source, runtime)
            self.assertFalse((runtime / "voice-changer" / "asound.conf").exists())
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "complete", report["issues"])
            self.assertEqual(report["application_source"], str(source.resolve()))
            self.assertEqual(
                report["application_entry"],
                str((source / "voice_changer" / "app" / "main.py").resolve()),
            )
            self.assertEqual(report["target_classes"], {
                "input_level": "QProgressBar", "output_level": "QProgressBar",
            })
            self.assertTrue(all(value > 0 for value in report["total_paint_events"].values()))
            self.assertIn("Paint measurement started.", result.stderr)
            self.assertIn("Paint measurement stopped", result.stderr)

    def test_legacy_launcher_rejects_unhooked_entries_before_calling_them(self):
        for signature, expected in (
            ("argv=None", "no explicit window_ready hook"),
            ("argv=None, **kwargs", "no explicit window_ready hook"),
            ("argv=None, window_ready=None, /", "no explicit window_ready hook"),
            ("argv=None, *, window_ready", "disabled by default"),
            ("argv=None, *, window_ready=True", "disabled by default"),
        ):
            with self.subTest(signature=signature):
                entry = (
                    "import os\nfrom pathlib import Path\n"
                    "Path(os.environ['XDG_RUNTIME_DIR'], 'imported').write_text('imported')\n"
                    f"def main({signature}):\n"
                    "    Path(os.environ['XDG_RUNTIME_DIR'], 'called').write_text('called')\n"
                    "    return 0\n"
                )
                with tempfile.TemporaryDirectory(prefix="voice-changer-legacy-fixture-") as directory:
                    source, runtime = self.legacy_fixture(directory, entry=entry)
                    result = self.run_legacy_fixture(source, runtime)
                    self.assertFalse((runtime / "imported").exists())
                    self.assertFalse((runtime / "called").exists())
                    self.assertFalse((runtime / "voice-changer" / "asound.conf").exists())
                    self.assertEqual(result.returncode, 1, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertEqual(report["status"], "incomplete")
                    self.assertEqual(report["application_source"], str(source))
                    self.assertIn(expected, report["issues"][0])
                    self.assertIn("setup failed", result.stderr)

    def test_legacy_launcher_rechecks_hook_after_module_import(self):
        entry = r'''
import os
from pathlib import Path

def enable_hook(function):
    function.__kwdefaults__["window_ready"] = True
    return function

@enable_hook
def main(argv=None, *, window_ready=None):
    Path(os.environ["XDG_RUNTIME_DIR"], "called").write_text("called")
    return 0
'''
        with tempfile.TemporaryDirectory(prefix="voice-changer-legacy-fixture-") as directory:
            source, runtime = self.legacy_fixture(directory, entry=entry)
            result = self.run_legacy_fixture(source, runtime)
            self.assertFalse((runtime / "called").exists())
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["application_source"], str(source))
            self.assertIn("disabled by default", report["issues"][0])

    def test_legacy_launcher_rejects_missing_source(self):
        with tempfile.TemporaryDirectory(prefix="voice-changer-legacy-fixture-") as directory:
            runtime = Path(directory) / "runtime"
            runtime.mkdir(mode=0o700)
            result = self.run_legacy_fixture(Path(directory) / "missing", runtime)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "incomplete")
            self.assertIsNone(report["application_source"])
            self.assertIn("Cannot resolve the selected application source", report["issues"][0])
            self.assertFalse((runtime / "voice-changer" / "asound.conf").exists())

    def test_legacy_launcher_rejects_invalid_entry_syntax(self):
        with tempfile.TemporaryDirectory(prefix="voice-changer-legacy-fixture-") as directory:
            source, runtime = self.legacy_fixture(directory, entry="def main(:\n    pass\n")
            result = self.run_legacy_fixture(source, runtime)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["application_source"], str(source))
            self.assertIn("Cannot inspect the selected application entry", report["issues"][0])
            self.assertFalse((runtime / "voice-changer" / "asound.conf").exists())

    def test_legacy_launcher_rejects_mixed_package_origins_before_start(self):
        entry = r'''
import os
import sys
from pathlib import Path
from types import ModuleType

foreign = ModuleType("voice_changer.foreign")
foreign.__file__ = os.__file__
sys.modules[foreign.__name__] = foreign

def main(argv=None, *, window_ready=None):
    Path(os.environ["XDG_RUNTIME_DIR"], "called").write_text("called")
    return 0
'''
        with tempfile.TemporaryDirectory(prefix="voice-changer-legacy-fixture-") as directory:
            source, runtime = self.legacy_fixture(directory, entry=entry)
            result = self.run_legacy_fixture(source, runtime)
            self.assertFalse((runtime / "called").exists())
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("outside the selected application source", report["issues"][0])

    def test_legacy_launcher_reports_invalid_widgets_and_closes(self):
        window = r'''
from PySide6.QtWidgets import QProgressBar, QVBoxLayout, QWidget

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.input_level = QWidget()
        self.output_level = QProgressBar()
        layout = QVBoxLayout(self)
        layout.addWidget(self.input_level)
        layout.addWidget(self.output_level)
'''
        with tempfile.TemporaryDirectory(prefix="voice-changer-legacy-fixture-") as directory:
            source, runtime = self.legacy_fixture(directory, window=window)
            result = self.run_legacy_fixture(source, runtime)
            self.assertFalse((runtime / "voice-changer" / "asound.conf").exists())
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("both be QProgressBar", report["issues"][0])
            self.assertIn("setup failed", result.stderr)

    def test_legacy_launcher_rejects_non_window_callback_and_quits(self):
        entry = r'''
def main(argv=None, *, window_ready=None):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    app = QApplication(["synthetic-legacy"])
    QTimer.singleShot(1500, app.quit)
    window_ready(None)
    return app.exec()
'''
        with tempfile.TemporaryDirectory(prefix="voice-changer-legacy-fixture-") as directory:
            source, runtime = self.legacy_fixture(directory, entry=entry)
            result = self.run_legacy_fixture(source, runtime)
            self.assertFalse((runtime / "voice-changer" / "asound.conf").exists())
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("not a QWidget", report["issues"][0])
            self.assertIn("setup failed", result.stderr)

    def test_legacy_launcher_reports_when_no_application_can_be_closed(self):
        entry = "def main(argv=None, *, window_ready=None):\n    window_ready(None)\n    return 0\n"
        with tempfile.TemporaryDirectory(prefix="voice-changer-legacy-fixture-") as directory:
            source, runtime = self.legacy_fixture(directory, entry=entry)
            result = self.run_legacy_fixture(source, runtime)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("not a QWidget", report["issues"][0])
            self.assertIn("no Qt application", report["issues"][1])
            self.assertIn("no Qt application", result.stderr)

    def test_legacy_launcher_rejects_foreign_window_module_and_closes(self):
        window = r'''
import os
from PySide6.QtWidgets import QProgressBar, QVBoxLayout, QWidget

__file__ = os.__file__

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.input_level, self.output_level = QProgressBar(), QProgressBar()
        layout = QVBoxLayout(self)
        layout.addWidget(self.input_level)
        layout.addWidget(self.output_level)
'''
        with tempfile.TemporaryDirectory(prefix="voice-changer-legacy-fixture-") as directory:
            source, runtime = self.legacy_fixture(directory, window=window)
            result = self.run_legacy_fixture(source, runtime)
            self.assertFalse((runtime / "voice-changer" / "asound.conf").exists())
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("outside the selected application source", report["issues"][0])
            self.assertIn("setup failed", result.stderr)

    def test_legacy_launcher_rejects_callback_from_worker_thread(self):
        entry = r'''
from threading import Thread

def main(argv=None, *, window_ready=None):
    from ..audio import virtual_mic
    virtual_mic.prepare()
    from PySide6.QtWidgets import QApplication
    from ..ui.main_window import MainWindow
    app = QApplication(["synthetic-legacy"])
    window = MainWindow()
    window.show()
    worker = Thread(target=window_ready, args=(window,))
    worker.start()
    worker.join(timeout=3)
    if worker.is_alive():
        raise RuntimeError("synthetic callback did not finish")
    return app.exec()
'''
        with tempfile.TemporaryDirectory(prefix="voice-changer-legacy-fixture-") as directory:
            source, runtime = self.legacy_fixture(directory, entry=entry)
            result = self.run_legacy_fixture(source, runtime)
            self.assertFalse((runtime / "voice-changer" / "asound.conf").exists())
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("Qt GUI thread", report["issues"][0])
            self.assertIn("setup failed", result.stderr)

    def test_legacy_launcher_reports_duplicate_callback_and_closes(self):
        entry = r'''
def main(argv=None, *, window_ready=None):
    from ..audio import virtual_mic
    virtual_mic.prepare()
    from PySide6.QtWidgets import QApplication
    from ..ui.main_window import MainWindow
    app = QApplication(["synthetic-legacy"])
    window = MainWindow()
    window.show()
    window_ready(window)
    for _ in range(100):
        window_ready(window)
    return app.exec()
'''
        with tempfile.TemporaryDirectory(prefix="voice-changer-legacy-fixture-") as directory:
            source, runtime = self.legacy_fixture(directory, entry=entry)
            result = self.run_legacy_fixture(source, runtime)
            self.assertFalse((runtime / "voice-changer" / "asound.conf").exists())
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(len(report["issues"]), 1)
            self.assertIn("callback was invoked more than once", report["issues"][0])
            self.assertEqual(result.stderr.count("setup failed"), 1)

    def test_legacy_launcher_reports_ignored_callback(self):
        entry = "def main(argv=None, *, window_ready=None):\n    return 0\n"
        with tempfile.TemporaryDirectory(prefix="voice-changer-legacy-fixture-") as directory:
            source, runtime = self.legacy_fixture(directory, entry=entry)
            result = self.run_legacy_fixture(source, runtime)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("before invoking", report["issues"][0])

    def test_entry_hook_with_synthetic_window_does_not_prepare_audio(self):
        code = r'''
import importlib.util
import sys
from unittest.mock import patch
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

spec = importlib.util.spec_from_file_location("probe_for_child", sys.argv[1])
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)

class Lane(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.canvas = QWidget()
        self.analysis_canvas = QWidget()
        for widget in (self.canvas, self.analysis_canvas):
            widget.setMinimumSize(40, 30)
            layout.addWidget(widget)

class SyntheticWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.input_level, self.output_level = Lane(), Lane()
        layout = QVBoxLayout(self)
        layout.addWidget(self.input_level)
        layout.addWidget(self.output_level)
        QTimer.singleShot(100, self.begin)
        QTimer.singleShot(2000, QApplication.instance().quit)

    def begin(self):
        self.counter = QApplication.instance().findChild(probe.PaintCounter)
        self.activateWindow()
        self.setFocus()
        QTest.keyClick(
            self, Qt.Key.Key_F8,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        for widget in probe.current_targets(self).values():
            widget.update()
        QTimer.singleShot(150, self.finish)

    def finish(self):
        QTest.keyClick(
            self, Qt.Key.Key_F9,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        self.close()

from voice_changer.audio import virtual_mic
import voice_changer.ui.main_window as real_window
with patch.object(virtual_mic, "prepare") as prepare, \
     patch.object(real_window, "MainWindow", SyntheticWindow):
    result = probe.main(["--launch-real-app"])
    assert prepare.call_count == 1
raise SystemExit(result)
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        environment["QT_QPA_PLATFORM"] = "offscreen"
        with tempfile.TemporaryDirectory(prefix="voice-changer-w08-runtime-") as runtime:
            environment["XDG_RUNTIME_DIR"] = runtime
            result = subprocess.run(
                [sys.executable, "-B", "-c", code, str(ROOT / "scripts" / "w08_paint_monitor.py")],
                cwd=ROOT, env=environment, capture_output=True, text=True, timeout=8,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("may discover devices and write shared ALSA", result.stderr)
        self.assertIn("Paint measurement started.", result.stderr)
        self.assertIn("Paint measurement stopped", result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "complete", report["issues"])
        self.assertEqual(len(report["targets"]), 4)
        self.assertTrue(all(count > 0 for count in report["total_paint_events"].values()))

    def test_entry_reports_unknown_window_and_closes_without_audio(self):
        code = r'''
import importlib.util
import sys
from unittest.mock import patch
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QWidget

spec = importlib.util.spec_from_file_location("probe_for_child", sys.argv[1])
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)

class UnknownWindow(QWidget):
    def __init__(self):
        super().__init__()
        QTimer.singleShot(2000, QApplication.instance().quit)

from voice_changer.audio import virtual_mic
import voice_changer.ui.main_window as real_window
with patch.object(virtual_mic, "prepare") as prepare, \
     patch.object(real_window, "MainWindow", UnknownWindow):
    result = probe.main(["--launch-real-app"])
    assert prepare.call_count == 1
raise SystemExit(result)
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        environment["QT_QPA_PLATFORM"] = "offscreen"
        with tempfile.TemporaryDirectory(prefix="voice-changer-w08-runtime-") as runtime:
            environment["XDG_RUNTIME_DIR"] = runtime
            result = subprocess.run(
                [sys.executable, "-B", "-c", code, str(ROOT / "scripts" / "w08_paint_monitor.py")],
                cwd=ROOT, env=environment, capture_output=True, text=True, timeout=8,
            )
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("not found", report["issues"][0])
        self.assertIn("setup failed", result.stderr)

    def test_cpu_and_paints_share_the_same_synthetic_window_interval(self):
        code = r'''
import importlib.util
import os
import sys
from unittest.mock import patch
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

spec = importlib.util.spec_from_file_location("probe_for_child", sys.argv[1])
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)
from voice_changer.audio.analysis_status import AnalysisReading
from voice_changer.ui.audio_wave import AudioWave, WaveComparison

class SyntheticWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.input_level, self.output_level = AudioWave("Synthetic input"), AudioWave("Synthetic output")
        layout = QVBoxLayout(self)
        layout.addWidget(WaveComparison(self.input_level, self.output_level))
        self.resize(800, 500)
        self.index = 0
        self.updater = QTimer(self)
        self.updater.setInterval(50)
        self.updater.timeout.connect(self.update_waves)
        QTimer.singleShot(150, self.begin)
        QTimer.singleShot(4000, QApplication.instance().quit)

    def begin(self):
        self.counter = QApplication.instance().findChild(probe.PaintCounter)
        self.counter.start()
        self.updater.start()
        os.write(int(sys.argv[2]), b"ready")
        os.close(int(sys.argv[2]))
        QTimer.singleShot(1700, self.finish)

    def update_waves(self):
        self.index += 1
        for offset, wave in enumerate((self.input_level, self.output_level)):
            peak = 0.1 + (self.index % 10) * 0.02 + offset * 0.05
            wave.set_peak(peak)
            wave.set_analysis(AnalysisReading((peak, peak, peak), 200 + self.index))

    def finish(self):
        self.updater.stop()
        self.counter.stop()
        self.close()

from voice_changer.audio import virtual_mic
import voice_changer.ui.main_window as real_window
with patch.object(virtual_mic, "prepare"), patch.object(real_window, "MainWindow", SyntheticWindow):
    result = probe.main(["--launch-real-app"])
raise SystemExit(result)
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        environment["QT_QPA_PLATFORM"] = "offscreen"
        with tempfile.TemporaryDirectory(prefix="voice-changer-w08-runtime-") as runtime:
            environment["XDG_RUNTIME_DIR"] = runtime
            read_fd, write_fd = os.pipe()
            try:
                with subprocess.Popen(
                    [sys.executable, "-B", "-c", code,
                     str(ROOT / "scripts" / "w08_paint_monitor.py"), str(write_fd)],
                    cwd=ROOT, env=environment, pass_fds=(write_fd,),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                ) as child:
                    os.close(write_fd)
                    write_fd = -1
                    ready, _, _ = select.select([read_fd], [], [], 5)
                    self.assertTrue(ready, "Synthetic window did not start observation")
                    self.assertEqual(os.read(read_fd, 5), b"ready")
                    cpu = subprocess.run(
                        [sys.executable, "-B", str(ROOT / "scripts" / "w08_cpu_monitor.py"),
                         "--pid", str(child.pid), "--seconds", "0.8", "--interval", "0.4"],
                        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=5,
                    )
                    stdout, stderr = child.communicate(timeout=5)
                    self.assertEqual(child.returncode, 0, stderr)
                    self.assertIn(f"PID {child.pid}", stderr)
            finally:
                os.close(read_fd)
                if write_fd >= 0:
                    os.close(write_fd)
        self.assertEqual(cpu.returncode, 0, cpu.stderr)
        cpu_report, paint_report = json.loads(cpu.stdout), json.loads(stdout)
        self.assertEqual(cpu_report["status"], "complete")
        self.assertEqual(paint_report["status"], "complete", paint_report["issues"])
        self.assertEqual(cpu_report["target_pids"], [paint_report["pid"]])
        self.assertGreaterEqual(
            cpu_report["start_monotonic_ns"], paint_report["start_monotonic_ns"],
        )
        self.assertLessEqual(cpu_report["end_monotonic_ns"], paint_report["end_monotonic_ns"])
        self.assertTrue(all(count > 0 for count in paint_report["total_paint_events"].values()))


if __name__ == "__main__":
    unittest.main()
