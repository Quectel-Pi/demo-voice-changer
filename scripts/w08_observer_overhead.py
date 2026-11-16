#!/usr/bin/env python3
"""Compare paint-observer CPU cost using only offscreen synthetic widgets."""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence


ROOT = Path(__file__).resolve().parents[1]
INTERVAL_MS = 50
WARMUP_UPDATES = 10
LAYOUTS = ("legacy", "current")
TARGET_NAMES = {
    "legacy": {"input_level", "output_level"},
    "current": {"input_envelope", "input_analysis", "output_envelope", "output_analysis"},
}


class OverheadError(Exception):
    """The synthetic comparison is incomplete or not comparable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OverheadError(message)


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _affinity_count() -> int | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return len(os.sched_getaffinity(0))
    except OSError as error:
        raise OverheadError(f"Cannot read process CPU affinity: {error}") from error


def _validate_trial(trial: dict, layout: str, enabled: bool, updates: int) -> None:
    label = f"{layout} observer {'on' if enabled else 'off'}"
    try:
        _require(trial["status"] == "complete", f"{label} did not complete: {trial['issues']}")
        _require(
            trial["layout"] == layout and trial["observer_enabled"] is enabled,
            f"{label} returned the wrong trial identity",
        )
        _require(trial["qt_platform"] == "offscreen", f"{label} did not use Qt offscreen")
        _require(trial["measurement_updates"] == updates, f"{label} had missing updates")
        wall, cpu, percent = trial["wall_seconds"], trial["cpu_seconds"], trial["cpu_percent"]
        _require(
            all(_number(value) for value in (wall, cpu, percent)) and wall >= 3 and cpu >= 0,
            f"{label} has invalid time or CPU data",
        )
        _require(
            math.isclose(percent, 100 * cpu / wall, rel_tol=1e-9, abs_tol=1e-9),
            f"{label} has inconsistent CPU data",
        )
        rate = updates / wall
        _require(18 <= rate <= 22, f"{label} update rate is not close to 20 Hz: {rate:.2f}")
        before, after = trial["targets_start"], trial["targets_end"]
        _require(
            isinstance(before, dict) and isinstance(after, dict)
            and set(before) == TARGET_NAMES[layout] and before == after
            and all(target["fully_visible"] for target in before.values()),
            f"{label} targets were hidden, clipped, or changed geometry",
        )
        counts = trial["paint_events"]
        if enabled:
            _require(trial["observer_shortcuts"] == 2, f"{label} is missing observer shortcuts")
            _require(
                trial["observer_status"] == "complete",
                f"{label} observer was incomplete: {trial['observer_issues']}",
            )
            _require(
                isinstance(counts, dict) and set(counts) == TARGET_NAMES[layout]
                and all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                        for value in counts.values()),
                f"{label} is missing target paint events",
            )
        else:
            _require(
                trial["observer_status"] is None and counts is None
                and trial["observer_shortcuts"] == 0,
                f"{label} unexpectedly installed the observer",
            )
    except (KeyError, TypeError, ValueError) as error:
        raise OverheadError(f"Malformed {label} trial: {error}") from error


def summarize(trials: list[dict], pairs: int, updates: int) -> dict:
    _require(len(trials) == len(LAYOUTS) * pairs * 2, "The comparison is missing trials")
    comparison = {}
    versions = None
    for layout in LAYOUTS:
        pair_results = []
        geometry = None
        for pair in range(1, pairs + 1):
            selected = [trial for trial in trials if trial.get("layout") == layout
                        and trial.get("pair") == pair]
            _require(len(selected) == 2, f"{layout} pair {pair} is missing a trial")
            off = next((trial for trial in selected if trial.get("observer_enabled") is False), None)
            on = next((trial for trial in selected if trial.get("observer_enabled") is True), None)
            _require(off is not None and on is not None, f"{layout} pair {pair} is not on/off")
            _validate_trial(off, layout, False, updates)
            _validate_trial(on, layout, True, updates)
            _require(
                off["targets_start"] == on["targets_start"],
                f"{layout} pair {pair} used different target geometry",
            )
            if geometry is None:
                geometry = off["targets_start"]
            _require(off["targets_start"] == geometry, f"{layout} geometry changed between pairs")
            for trial in (off, on):
                try:
                    actual_versions = (trial["python"], trial["pyside"], trial["qt"])
                except KeyError as error:
                    raise OverheadError(f"{layout} trial is missing version information: {error}") from error
                if versions is None:
                    versions = actual_versions
                _require(actual_versions == versions, "Python or Qt versions changed between trials")
            pair_results.append({
                "pair": pair,
                "off_cpu_percent": off["cpu_percent"],
                "on_cpu_percent": on["cpu_percent"],
                "difference_percentage_points": on["cpu_percent"] - off["cpu_percent"],
            })
        differences = [item["difference_percentage_points"] for item in pair_results]
        comparison[layout] = {
            "pairs": pair_results,
            "mean_difference_percentage_points": statistics.mean(differences),
            "median_difference_percentage_points": statistics.median(differences),
            "min_difference_percentage_points": min(differences),
            "max_difference_percentage_points": max(differences),
        }
    return comparison


def _worker(layout_name: str, enabled: bool, updates: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="w08-overhead-runtime-") as runtime:
        os.environ["XDG_RUNTIME_DIR"] = runtime
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        report = _run_worker(layout_name, enabled, updates)
    try:
        _validate_trial(report, layout_name, enabled, updates)
    except OverheadError as error:
        issues = report.get("issues")
        if not isinstance(issues, list):
            issues = ["Worker returned an invalid issue list"]
        report["status"] = "incomplete"
        report["issues"] = [*issues, str(error)]
    return report


def _run_worker(layout_name: str, enabled: bool, updates: int) -> dict:
    from PySide6 import __version__ as pyside_version
    from PySide6.QtCore import QPoint, QTimer, Qt, qVersion
    from PySide6.QtGui import QRegion, QShortcut
    from PySide6.QtWidgets import QApplication, QProgressBar, QVBoxLayout, QWidget

    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
    from voice_changer.audio.analysis_status import AnalysisReading
    from voice_changer.ui.audio_wave import AudioWave, WaveComparison
    import voice_changer.ui.audio_wave as wave_module
    import w08_paint_monitor as observer_module
    from w08_paint_monitor import (
        MeasurementError, PaintCounter, PaintSession, current_targets, legacy_targets,
    )

    if (
        Path(wave_module.__file__).resolve() != ROOT / "src/voice_changer/ui/audio_wave.py"
        or Path(observer_module.__file__).resolve() != ROOT / "scripts/w08_paint_monitor.py"
    ):
        raise OverheadError("Synthetic widgets or observer came from an unexpected source")

    app = QApplication(["w08-synthetic-overhead"])
    window = QWidget()
    box = QVBoxLayout(window)
    if layout_name == "legacy":
        window.input_level, window.output_level = QProgressBar(), QProgressBar()
        for bar in (window.input_level, window.output_level):
            bar.setRange(0, 1000)
            bar.setFixedHeight(24)
            box.addWidget(bar)
        box.addStretch()
        targets = legacy_targets(window)
    else:
        window.input_level, window.output_level = AudioWave("Synthetic input"), AudioWave("Synthetic output")
        box.addWidget(WaveComparison(window.input_level, window.output_level))
        targets = current_targets(window)
    window.resize(800, 500)
    window.show()

    def snapshot() -> dict:
        result = {}
        for name, widget in targets.items():
            point = widget.mapTo(window, QPoint(0, 0))
            rect = widget.rect()
            result[name] = {
                "x": point.x(), "y": point.y(), "width": widget.width(), "height": widget.height(),
                "class": widget.metaObject().className(),
                "fully_visible": (
                    widget.isVisible() and not rect.isEmpty()
                    and QRegion(rect).subtracted(widget.visibleRegion()).isEmpty()
                ),
            }
        return result

    class Runner:
        def __init__(self) -> None:
            self.warmup = 0
            self.measured = 0
            self.session: PaintSession | None = None
            self.counter: PaintCounter | None = None
            self.started = False
            self.result: dict | None = None
            self.timer = QTimer(window)
            self.timer.setTimerType(Qt.TimerType.PreciseTimer)
            self.timer.timeout.connect(self.tick)
            self.timer.start(INTERVAL_MS)

        def update_widgets(self, index: int) -> None:
            if layout_name == "legacy":
                window.input_level.setValue((index * 173) % 1001)
                window.output_level.setValue(((index + 4) * 173) % 1001)
            else:
                for offset, wave in enumerate((window.input_level, window.output_level)):
                    peak = 0.1 + ((index * 3 + offset * 7) % 20) * 0.01
                    wave.set_peak(peak)
                    wave.set_analysis(AnalysisReading((peak, peak / 2, peak / 4), 180 + (index + offset * 5) % 80))

        def fail(self, reason: str) -> None:
            self.timer.stop()
            self.result = {"status": "incomplete", "issues": [reason]}
            window.close()
            app.quit()

        def tick(self) -> None:
            if self.warmup < WARMUP_UPDATES:
                self.update_widgets(self.warmup)
                self.warmup += 1
                return
            if not self.started:
                self.targets_start = snapshot()
                try:
                    if enabled:
                        self.session = PaintSession(legacy=layout_name == "legacy")
                        self.session.attach(window)
                        counters = app.findChildren(PaintCounter)
                        if len(counters) != 1:
                            self.fail(
                                f"Expected one paint counter, found {len(counters)}: "
                                f"{self.session.report()['issues']}"
                            )
                            return
                        self.counter = counters[0]
                    self.load_start = os.getloadavg()
                    self.wall_start = time.monotonic_ns()
                    self.cpu_start = time.process_time_ns()
                    if self.counter is not None:
                        self.counter.start()
                except MeasurementError as error:
                    self.fail(str(error))
                    return
                except OSError as error:
                    self.fail(f"Cannot read system load: {error}")
                    return
                self.started = True
                return
            if self.measured < updates:
                self.update_widgets(WARMUP_UPDATES + self.measured)
                self.measured += 1
                return
            self.timer.stop()
            try:
                if self.counter is not None:
                    self.counter.stop()
            except MeasurementError as error:
                self.fail(str(error))
                return
            cpu_end = time.process_time_ns()
            wall_end = time.monotonic_ns()
            wall = (wall_end - self.wall_start) / 1_000_000_000
            cpu = (cpu_end - self.cpu_start) / 1_000_000_000
            if wall <= 0 or cpu < 0:
                self.fail("The synthetic measurement clock did not advance")
                return
            try:
                load_end = os.getloadavg()
            except OSError as error:
                self.fail(f"Cannot read system load: {error}")
                return
            paint = self.session.report() if self.session is not None else None
            self.result = {
                "status": "complete",
                "issues": [],
                "layout": layout_name,
                "observer_enabled": enabled,
                "pid": os.getpid(),
                "python": sys.version.split()[0],
                "pyside": pyside_version,
                "qt": qVersion(),
                "qt_platform": app.platformName(),
                "measurement_updates": self.measured,
                "wall_seconds": wall,
                "cpu_seconds": cpu,
                "cpu_percent": 100 * cpu / wall,
                "load_average_start": list(self.load_start),
                "load_average_end": list(load_end),
                "targets_start": self.targets_start,
                "targets_end": snapshot(),
                "observer_shortcuts": len(window.findChildren(QShortcut)),
                "observer_status": paint["status"] if paint is not None else None,
                "observer_issues": paint["issues"] if paint is not None else None,
                "paint_events": paint["total_paint_events"] if paint is not None else None,
            }
            if self.counter is not None:
                self.counter.detach()
            window.close()
            app.quit()

    runner = Runner()
    app.exec()
    if runner.result is None:
        return {"status": "incomplete", "issues": ["Synthetic window exited before measurement"]}
    forbidden = [
        name for name in sys.modules
        if name == "sounddevice" or name.startswith("sounddevice.")
        or name in ("voice_changer.audio.virtual_mic", "voice_changer.audio.sounddevice_backend")
    ]
    if forbidden:
        return {"status": "incomplete", "issues": [f"Unexpected audio imports: {forbidden}"]}
    return runner.result


def _execute_trial(layout: str, enabled: bool, updates: int) -> dict:
    command = [
        sys.executable, "-B", str(Path(__file__).resolve()), "--_worker", layout,
        "--updates", str(updates),
    ]
    if enabled:
        command.append("--_observed")
    environment = dict(
        os.environ, QT_QPA_PLATFORM="offscreen", PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(ROOT / "src"),
    )
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=environment, capture_output=True, text=True,
            timeout=(updates + WARMUP_UPDATES) * INTERVAL_MS / 1000 * 3 + 10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise OverheadError(f"{layout} worker could not finish: {error}") from error
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise OverheadError(
            f"{layout} worker exited {completed.returncode} or returned invalid JSON: "
            f"{completed.stderr.strip() or error}"
        ) from error
    if not isinstance(result, dict):
        raise OverheadError(f"{layout} worker returned an invalid report")
    if completed.returncode:
        raise OverheadError(
            f"{layout} worker exited {completed.returncode}: "
            f"{result.get('issues', [])}; {completed.stderr.strip()}"
        )
    return result


def run_comparison(pairs: int, updates: int) -> dict:
    trials = []
    report = {
        "kind": "synthetic_qt_observer_overhead",
        "status": "incomplete",
        "issues": [],
        "environment": {
            "os": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "logical_cpus": os.cpu_count(),
            "affinity_cpus": None,
        },
        "method": {
            "pairs_per_layout": pairs,
            "updates_per_trial": updates,
            "warmup_updates": WARMUP_UPDATES,
            "interval_ms": INTERVAL_MS,
            "order": "off/on then on/off, alternating",
            "cpu_metric": "process CPU time / monotonic wall time; percent of one CPU",
            "observer": "PaintSession with PaintCounter and shortcuts, started directly",
        },
        "trials": trials,
        "comparison": None,
        "limitations": [
            "Synthetic offscreen Qt widgets only; this is not Pi M2 or real audio.",
            "Matching update calls do not prove equal paint work in the unobserved trial.",
            "Short alternating pairs do not establish a statistical confidence interval.",
            "Start and stop costs are included once per short observed interval.",
            "Session setup, shortcut key presses, and JSON output are outside the timed interval.",
        ],
    }
    try:
        report["environment"]["affinity_cpus"] = _affinity_count()
        for layout in LAYOUTS:
            for pair in range(1, pairs + 1):
                order = (False, True) if pair % 2 else (True, False)
                for enabled in order:
                    print(
                        f"Synthetic {layout}: pair {pair}/{pairs}, observer "
                        f"{'on' if enabled else 'off'}",
                        file=sys.stderr, flush=True,
                    )
                    trial = _execute_trial(layout, enabled, updates)
                    trial["pair"] = pair
                    trials.append(trial)
                    _validate_trial(trial, layout, enabled, updates)
        report["comparison"] = summarize(trials, pairs, updates)
        report["status"] = "complete"
    except OverheadError as error:
        report["issues"].append(str(error))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=4, help="Paired trials per layout (3–6).")
    parser.add_argument("--updates", type=int, default=80, help="50 ms updates per trial (60–200).")
    parser.add_argument("--_worker", choices=LAYOUTS, help=argparse.SUPPRESS)
    parser.add_argument("--_observed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 3 <= args.pairs <= 6:
        parser.error("--pairs must be between 3 and 6")
    if not 60 <= args.updates <= 200:
        parser.error("--updates must be between 60 and 200")
    if args._observed and args._worker is None:
        parser.error("--_observed is only available to a synthetic worker")
    report = (
        _worker(args._worker, args._observed, args.updates)
        if args._worker is not None else run_comparison(args.pairs, args.updates)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
