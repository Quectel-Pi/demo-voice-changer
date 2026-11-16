"""Controlled and local-only checks for the W08 process CPU observer."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "w08_cpu_monitor", ROOT / "scripts" / "w08_cpu_monitor.py",
)
monitor = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = monitor
spec.loader.exec_module(monitor)


def proc_stat(pid, *, user=100, system=30, start=900, state="S"):
    fields = [state] + ["0"] * 19
    fields[11], fields[12], fields[19] = str(user), str(system), str(start)
    return f"{pid} (name with ) and spaces) {' '.join(fields)}"


def snapshot(seconds, cpu_ticks, *, start_ticks=900, cpus=(0, 1, 2, 3), pid=42,
             observer_ns=0):
    return monitor.Snapshot(
        round(seconds * 1_000_000_000), cpus,
        (monitor.ProcessCounters(pid, start_ticks, cpu_ticks),),
        observer_ns, (0.25, 0.50, 0.75), 1_000_000,
    )


class FakeClock:
    def __init__(self, sleep_factor=1):
        self.now_ns = 0
        self.sleep_factor = sleep_factor

    def monotonic_ns(self):
        return self.now_ns

    def sleep(self, seconds):
        self.now_ns += round(seconds * self.sleep_factor * 1_000_000_000)


class W08CpuMonitorTests(unittest.TestCase):
    def test_online_cpu_parser_accepts_ranges_and_rejects_bad_lists(self):
        self.assertEqual(monitor.parse_online_cpus("0-3,8,10-11\n"), (0, 1, 2, 3, 8, 10, 11))
        for invalid in ("", "4-2", "0-2,2", "-1", "1--2", "0-1048576"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(monitor.MeasurementError):
                    monitor.parse_online_cpus(invalid)

    def test_process_stat_handles_spaces_and_parentheses_and_rejects_invalid_data(self):
        self.assertEqual(
            monitor.parse_process_stat(proc_stat(42), 42),
            monitor.ProcessCounters(pid=42, start_ticks=900, cpu_ticks=130),
        )
        for invalid in (
            proc_stat(41), proc_stat(42, state="Z"),
            proc_stat(42, user=-1), "42 (broken) S 1",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(monitor.MeasurementError):
                    monitor.parse_process_stat(invalid, 42)

    def test_reads_only_requested_fake_proc_and_online_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "42").mkdir()
            (root / "42" / "stat").write_text(proc_stat(42), encoding="ascii")
            online = root / "online"
            online.write_text("0-3", encoding="ascii")
            times = iter((1_000, 3_000))
            result = monitor.capture_snapshot(
                (42,), proc_root=root, cpu_online=online,
                monotonic_ns=lambda: next(times),
                process_time_ns=lambda: 500,
                getloadavg=lambda: (1.0, 2.0, 3.0),
            )
            self.assertEqual(result.monotonic_ns, 2_000)
            self.assertEqual(result.read_span_ns, 2_000)
            self.assertEqual(result.online_cpus, (0, 1, 2, 3))
            self.assertEqual(result.processes[0].cpu_ticks, 130)
            with self.assertRaisesRegex(monitor.MeasurementError, "Cannot read"):
                monitor.capture_snapshot((43,), proc_root=root, cpu_online=online)

    def test_online_set_change_during_a_snapshot_is_not_ignored(self):
        class ChangingOnlineFile:
            def __init__(self):
                self.values = iter(("0-1", "2-3"))

            def read_text(self, *, encoding):
                return next(self.values)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "42").mkdir()
            (root / "42" / "stat").write_text(proc_stat(42), encoding="ascii")
            with self.assertRaisesRegex(monitor.MeasurementError, "changed during"):
                monitor.capture_snapshot((42,), proc_root=root, cpu_online=ChangingOnlineFile())

    def test_permission_denied_is_an_explicit_measurement_error(self):
        online = Path("/synthetic/online")

        def fake_read(path, *, encoding):
            if path == online:
                return "0-3"
            raise PermissionError("synthetic process access denied")

        with patch.object(Path, "read_text", fake_read):
            with self.assertRaisesRegex(monitor.MeasurementError, "process access denied"):
                monitor.capture_snapshot((42,), proc_root=Path("/synthetic/proc"), cpu_online=online)

    def test_interval_and_weighted_summary_include_threads_once_and_explicit_children(self):
        first = monitor.Snapshot(
            0, (0, 1, 2, 3),
            (monitor.ProcessCounters(42, 900, 100), monitor.ProcessCounters(43, 901, 200)),
            0, (0.0, 0.0, 0.0), 0,
        )
        second = monitor.Snapshot(
            1_000_000_000, first.online_cpus,
            (monitor.ProcessCounters(42, 900, 125), monitor.ProcessCounters(43, 901, 250)),
            10_000_000, (1.0, 2.0, 3.0), 1_000_000,
        )
        third = monitor.Snapshot(
            3_000_000_000, first.online_cpus,
            (monitor.ProcessCounters(42, 900, 175), monitor.ProcessCounters(43, 901, 300)),
            30_000_000, (1.0, 2.0, 3.0), 2_000_000,
        )
        a = monitor.interval_report(first, second, 100, 2)
        b = monitor.interval_report(second, third, 100, 2)
        self.assertEqual(a["raw_percent"], 75)
        self.assertEqual(a["normalized_percent"], 18.75)
        self.assertEqual([p["raw_percent"] for p in a["processes"]], [25, 50])
        self.assertEqual(a["observer_raw_percent"], 1)
        summary = monitor.summarize([a, b])
        self.assertAlmostEqual(summary["raw_mean_percent"], 175 / 3)
        self.assertAlmostEqual(summary["normalized_mean_percent"], 175 / 12)
        self.assertEqual(summary["raw_peak_percent"], 75)
        self.assertEqual(summary["normalized_peak_percent"], 18.75)
        self.assertEqual(summary["covered_seconds"], 3)

    def test_identity_online_set_time_and_counter_changes_are_errors(self):
        baseline = snapshot(1, 100)
        changed = (
            snapshot(2, 120, start_ticks=901),
            snapshot(2, 120, cpus=(4, 5, 6, 7)),
            snapshot(1, 120),
            snapshot(2, 99),
            snapshot(2, 120, pid=43),
        )
        for latest in changed:
            with self.subTest(latest=latest):
                with self.assertRaises(monitor.MeasurementError):
                    monitor.interval_report(baseline, latest, 100, 1)

    def test_controlled_sampling_and_missing_sample_are_explicit(self):
        for sleep_factor, expected in ((1, "complete"), (2, "incomplete")):
            with self.subTest(sleep_factor=sleep_factor):
                clock = FakeClock(sleep_factor)

                def capture():
                    seconds = clock.now_ns / 1_000_000_000
                    return snapshot(seconds, 100 + round(seconds * 50))

                report = monitor.collect(
                    (42,), 3, 1, 100, capture,
                    clock_ns=clock.monotonic_ns, sleep=clock.sleep,
                )
                self.assertEqual(report["status"], expected)
                self.assertAlmostEqual(report["summary"]["raw_mean_percent"], 50)
                self.assertAlmostEqual(report["summary"]["normalized_mean_percent"], 12.5)
                if sleep_factor == 1:
                    self.assertEqual(len(report["samples"]), 3)
                    self.assertIsNone(report["error"])
                else:
                    self.assertTrue(report["samples"][0]["late"])
                    self.assertIn("sampling interval", report["error"])

    def test_missing_process_is_incomplete_not_a_zero_measurement(self):
        def capture():
            raise monitor.MeasurementError("PID 42 unavailable")

        report = monitor.collect((42,), 1, 1, 100, capture)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["error"], "PID 42 unavailable")
        self.assertIsNone(report["summary"]["raw_mean_percent"])
        self.assertEqual(report["samples"], [])

    def test_rejects_invalid_or_unbounded_requests(self):
        for pids, duration, interval in (
            ((), 1, 1), ((42, 42), 1, 1), ((0,), 1, 1), ((42,), -1, 1),
            ((42,), float("nan"), 1), ((42,), 1, 0.01), ((42,), 3601, 1),
            ((42,), 3600, 0.05),
        ):
            with self.subTest(pids=pids, duration=duration, interval=interval):
                with self.assertRaises(ValueError):
                    monitor.collect(pids, duration, interval, 100, lambda: snapshot(0, 100))

    def test_cli_reads_our_own_process_without_audio_or_network(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts" / "w08_cpu_monitor.py"),
             "--pid", str(os.getpid()), "--seconds", "0.5", "--interval", "0.25"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        report = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["target_pids"], [os.getpid()])
        self.assertGreaterEqual(report["summary"]["covered_seconds"], 0.5)
        self.assertTrue(report["online_cpus"])
        self.assertGreaterEqual(len(report["samples"]), 2)

    def test_cli_reports_a_missing_process_without_fabricating_cpu_values(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts" / "w08_cpu_monitor.py"),
             "--pid", "2147483647", "--seconds", "0.1", "--interval", "0.1"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        report = json.loads(result.stdout)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("Cannot read", report["error"])
        self.assertIsNone(report["summary"]["raw_mean_percent"])
        self.assertIn("incomplete", result.stderr)

    def test_cli_detects_cpu_use_in_an_owned_synthetic_child(self):
        code = (
            "import threading, time\n"
            "def work():\n"
            "    deadline = time.monotonic() + 1.5\n"
            "    while time.monotonic() < deadline:\n"
            "        pass\n"
            "worker = threading.Thread(target=work)\n"
            "worker.start()\n"
            "worker.join()\n"
        )
        with subprocess.Popen(
            [sys.executable, "-B", "-c", code], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        ) as child:
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "scripts" / "w08_cpu_monitor.py"),
                 "--pid", str(child.pid), "--seconds", "0.5", "--interval", "0.25"],
                cwd=ROOT, capture_output=True, text=True, timeout=5,
            )
            self.assertEqual(child.wait(timeout=5), 0)
        report = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["status"], "complete")
        self.assertGreater(report["summary"]["raw_mean_percent"], 0)


if __name__ == "__main__":
    unittest.main()
