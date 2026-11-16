"""Synthetic-only checks for the W08 observer-overhead comparison."""

import copy
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "w08_observer_overhead.py"
spec = importlib.util.spec_from_file_location("w08_observer_overhead", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


def trial(layout, pair, enabled, cpu_percent):
    targets = {
        name: {
            "x": index * 10, "y": index * 20, "width": 100, "height": 20,
            "class": "SyntheticWidget", "fully_visible": True,
        }
        for index, name in enumerate(sorted(benchmark.TARGET_NAMES[layout]))
    }
    return {
        "status": "complete",
        "issues": [],
        "layout": layout,
        "pair": pair,
        "observer_enabled": enabled,
        "python": "3.10",
        "pyside": "6.8",
        "qt": "6.8",
        "qt_platform": "offscreen",
        "measurement_updates": 80,
        "wall_seconds": 4.0,
        "cpu_seconds": cpu_percent * 4.0 / 100,
        "cpu_percent": cpu_percent,
        "targets_start": targets,
        "targets_end": copy.deepcopy(targets),
        "observer_shortcuts": 2 if enabled else 0,
        "observer_status": "complete" if enabled else None,
        "observer_issues": [] if enabled else None,
        "paint_events": {name: 80 for name in targets} if enabled else None,
    }


class W08ObserverOverheadTests(unittest.TestCase):
    def samples(self):
        samples = []
        for layout in benchmark.LAYOUTS:
            for pair, difference in enumerate((0.2, -0.1, 0.4), 1):
                samples.append(trial(layout, pair, False, 2.0))
                samples.append(trial(layout, pair, True, 2.0 + difference))
        return samples

    def test_paired_summary_keeps_negative_differences(self):
        report = benchmark.summarize(self.samples(), pairs=3, updates=80)
        for layout in benchmark.LAYOUTS:
            summary = report[layout]
            self.assertAlmostEqual(summary["mean_difference_percentage_points"], 1 / 6)
            self.assertAlmostEqual(summary["median_difference_percentage_points"], 0.2)
            self.assertAlmostEqual(summary["min_difference_percentage_points"], -0.1)
            self.assertAlmostEqual(summary["max_difference_percentage_points"], 0.4)
            self.assertEqual(len(summary["pairs"]), 3)

    def test_incomplete_or_incomparable_trials_fail_explicitly(self):
        mutations = (
            ("missing updates", lambda item: item.update(measurement_updates=79)),
            ("invalid time", lambda item: item.update(wall_seconds=2.9)),
            ("invalid time", lambda item: item.update(cpu_seconds=float("nan"))),
            ("offscreen", lambda item: item.update(qt_platform="xcb")),
            ("hidden", lambda item: item["targets_start"][next(iter(item["targets_start"]))].update(
                fully_visible=False,
            )),
            ("missing observer shortcuts", lambda item: item.update(observer_shortcuts=1)),
            ("observer was incomplete", lambda item: item.update(observer_status="incomplete")),
            ("missing target paint", lambda item: item.update(paint_events={})),
        )
        for expected, mutate in mutations:
            with self.subTest(expected=expected):
                samples = self.samples()
                mutate(samples[1])
                with self.assertRaisesRegex(benchmark.OverheadError, expected):
                    benchmark.summarize(samples, pairs=3, updates=80)

    def test_mismatched_geometry_versions_and_missing_pairs_are_rejected(self):
        samples = self.samples()
        first = samples[1]
        name = next(iter(first["targets_start"]))
        first["targets_start"][name]["width"] += 1
        first["targets_end"][name]["width"] += 1
        with self.assertRaisesRegex(benchmark.OverheadError, "different target geometry"):
            benchmark.summarize(samples, pairs=3, updates=80)

        samples = self.samples()
        samples[-1]["qt"] = "different"
        with self.assertRaisesRegex(benchmark.OverheadError, "versions changed"):
            benchmark.summarize(samples, pairs=3, updates=80)

        with self.assertRaisesRegex(benchmark.OverheadError, "missing trials"):
            benchmark.summarize(samples[:-1], pairs=3, updates=80)

    def test_worker_failure_is_reported_instead_of_zero_data(self):
        failure = subprocess.CompletedProcess(
            [], 1, '{"status":"incomplete","issues":["synthetic failure"]}', "",
        )
        with patch.object(benchmark.subprocess, "run", return_value=failure):
            with self.assertRaisesRegex(benchmark.OverheadError, "synthetic failure"):
                benchmark._execute_trial("legacy", True, 80)

        with patch.object(benchmark, "_execute_trial", side_effect=benchmark.OverheadError("failed")), \
             contextlib.redirect_stderr(io.StringIO()):
            report = benchmark.run_comparison(pairs=3, updates=80)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["issues"], ["failed"])
        self.assertIsNone(report["comparison"])

    def test_worker_does_not_report_invalid_measurement_as_complete(self):
        incomplete = trial("legacy", 1, True, 2.0)
        incomplete["observer_status"] = "incomplete"
        with patch.object(benchmark, "_run_worker", return_value=incomplete), \
             patch.dict(os.environ):
            report = benchmark._worker("legacy", True, updates=80)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("observer was incomplete", report["issues"][-1])

    def test_affinity_failure_is_reported_as_incomplete(self):
        with patch.object(benchmark.os, "sched_getaffinity", side_effect=PermissionError("denied")):
            report = benchmark.run_comparison(pairs=3, updates=80)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("CPU affinity", report["issues"][0])
        self.assertEqual(report["trials"], [])

    def test_comparison_alternates_order_and_reports_session_scope(self):
        calls = []

        def fake_trial(layout, enabled, updates):
            calls.append((layout, enabled))
            return trial(layout, 0, enabled, 2.2 if enabled else 2.0)

        with patch.object(benchmark, "_execute_trial", side_effect=fake_trial), \
             contextlib.redirect_stderr(io.StringIO()):
            report = benchmark.run_comparison(pairs=3, updates=80)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(
            calls,
            [(layout, enabled) for layout in benchmark.LAYOUTS
             for enabled in (False, True, True, False, False, True)],
        )
        self.assertIn("PaintSession", report["method"]["observer"])
        self.assertEqual(len(report["trials"]), 12)

    def test_cli_rejects_short_or_incomplete_comparisons(self):
        for arguments in (("--pairs", "2"), ("--updates", "59"), ("--_observed",)):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [sys.executable, "-B", str(SCRIPT), *arguments],
                    cwd=ROOT, capture_output=True, text=True, timeout=5,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("error:", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_synthetic_workers_use_visible_widgets_without_audio(self):
        for layout, enabled in (("legacy", False), ("current", True)):
            with self.subTest(layout=layout, enabled=enabled):
                with tempfile.TemporaryDirectory(prefix="w08-overhead-test-runtime-") as runtime:
                    environment = dict(
                        os.environ, XDG_RUNTIME_DIR=runtime, QT_QPA_PLATFORM="offscreen",
                        PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT / "src"),
                    )
                    command = [
                        sys.executable, "-B", str(SCRIPT), "--_worker", layout, "--updates", "60",
                    ]
                    if enabled:
                        command.append("--_observed")
                    result = subprocess.run(
                        command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=15,
                    )
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                benchmark._validate_trial(report, layout, enabled, updates=60)
                self.assertEqual(report["measurement_updates"], 60)
                self.assertGreaterEqual(report["wall_seconds"], 3.0)
                self.assertEqual(report["qt_platform"], "offscreen")


if __name__ == "__main__":
    unittest.main()
