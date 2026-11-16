#!/usr/bin/env python3
"""Read CPU usage of explicitly selected Linux processes; never opens audio."""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from collections.abc import Callable, Sequence


class MeasurementError(Exception):
    """The observation is incomplete or cannot be compared safely."""


@dataclass(frozen=True)
class ProcessCounters:
    pid: int
    start_ticks: int
    cpu_ticks: int


@dataclass(frozen=True)
class Snapshot:
    monotonic_ns: int
    online_cpus: tuple[int, ...]
    processes: tuple[ProcessCounters, ...]
    observer_cpu_ns: int
    load_average: tuple[float, float, float]
    read_span_ns: int


def parse_online_cpus(text: str) -> tuple[int, ...]:
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", part):
            raise MeasurementError(f"Invalid online CPU list: {text!r}")
        bounds = [int(value) for value in part.split("-")]
        first, last = bounds[0], bounds[-1]
        if last < first or last > 1_048_575:
            raise MeasurementError(f"Invalid online CPU range: {part!r}")
        for cpu in range(first, last + 1):
            if cpu in cpus:
                raise MeasurementError(f"Duplicate online CPU: {cpu}")
            cpus.add(cpu)
    if not cpus:
        raise MeasurementError("The online CPU list is empty")
    return tuple(sorted(cpus))


def parse_process_stat(text: str, expected_pid: int) -> ProcessCounters:
    opening, closing = text.find("("), text.rfind(")")
    if opening < 1 or closing <= opening:
        raise MeasurementError(f"Malformed /proc stat for PID {expected_pid}")
    try:
        pid = int(text[:opening].strip())
        fields = text[closing + 1:].split()
        if len(fields) < 20 or fields[0] in ("Z", "X", "x"):
            raise ValueError("process exited or stat fields are missing")
        user_ticks, system_ticks, start_ticks = (
            int(fields[11]), int(fields[12]), int(fields[19])
        )
    except (ValueError, IndexError) as error:
        raise MeasurementError(f"Invalid /proc stat for PID {expected_pid}: {error}") from error
    if pid != expected_pid or min(user_ticks, system_ticks, start_ticks) < 0:
        raise MeasurementError(f"Invalid process identity or CPU counters for PID {expected_pid}")
    return ProcessCounters(pid, start_ticks, user_ticks + system_ticks)


def capture_snapshot(
    pids: tuple[int, ...],
    *,
    proc_root: Path = Path("/proc"),
    cpu_online: Path = Path("/sys/devices/system/cpu/online"),
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    process_time_ns: Callable[[], int] = time.process_time_ns,
    getloadavg: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> Snapshot:
    started = monotonic_ns()
    try:
        online_before = parse_online_cpus(cpu_online.read_text(encoding="ascii"))
        processes = tuple(
            parse_process_stat(
                (proc_root / str(pid) / "stat").read_text(encoding="ascii"), pid,
            )
            for pid in pids
        )
        load_average = getloadavg()
        observer_cpu_ns = process_time_ns()
        online_after = parse_online_cpus(cpu_online.read_text(encoding="ascii"))
    except (OSError, UnicodeError) as error:
        raise MeasurementError(f"Cannot read process or CPU statistics: {error}") from error
    ended = monotonic_ns()
    if online_before != online_after:
        raise MeasurementError("The online CPU set changed during a sample")
    if ended < started:
        raise MeasurementError("The monotonic clock moved backwards during a sample")
    if len(load_average) != 3 or not all(
        math.isfinite(value) and value >= 0 for value in load_average
    ):
        raise MeasurementError("System load averages are invalid")
    return Snapshot(
        (started + ended) // 2, online_before, processes, observer_cpu_ns,
        tuple(load_average), ended - started,
    )


def interval_report(
    before: Snapshot, after: Snapshot, ticks_per_second: int, interval_seconds: float,
) -> dict:
    if ticks_per_second <= 0 or interval_seconds <= 0:
        raise MeasurementError("CPU tick rate and sampling interval must be positive")
    elapsed_ns = after.monotonic_ns - before.monotonic_ns
    if elapsed_ns <= 0:
        raise MeasurementError("The sampling clock did not advance")
    if before.online_cpus != after.online_cpus or not before.online_cpus:
        raise MeasurementError("The online CPU set changed between samples")
    prior = {process.pid: process for process in before.processes}
    current = {process.pid: process for process in after.processes}
    if len(prior) != len(before.processes) or len(current) != len(after.processes):
        raise MeasurementError("The same PID was included more than once")
    if not prior or prior.keys() != current.keys():
        raise MeasurementError("The selected process set changed between samples")
    elapsed = elapsed_ns / 1_000_000_000
    processes = []
    for pid, previous in prior.items():
        latest = current[pid]
        if previous.start_ticks != latest.start_ticks:
            raise MeasurementError(f"PID {pid} was replaced during measurement")
        delta_ticks = latest.cpu_ticks - previous.cpu_ticks
        if delta_ticks < 0:
            raise MeasurementError(f"CPU counters decreased for PID {pid}")
        cpu_seconds = delta_ticks / ticks_per_second
        processes.append({
            "pid": pid,
            "start_ticks": previous.start_ticks,
            "cpu_seconds": cpu_seconds,
            "raw_percent": 100 * cpu_seconds / elapsed,
        })
    observer_delta_ns = after.observer_cpu_ns - before.observer_cpu_ns
    if observer_delta_ns < 0:
        raise MeasurementError("Observer CPU counters decreased")
    cpu_seconds = sum(process["cpu_seconds"] for process in processes)
    raw_percent = 100 * cpu_seconds / elapsed
    return {
        "start_monotonic_ns": before.monotonic_ns,
        "end_monotonic_ns": after.monotonic_ns,
        "wall_seconds": elapsed,
        "online_cpu_count": len(before.online_cpus),
        "processes": processes,
        "cpu_seconds": cpu_seconds,
        "raw_percent": raw_percent,
        "normalized_percent": raw_percent / len(before.online_cpus),
        "observer_raw_percent": 100 * observer_delta_ns / elapsed_ns,
        "load_average": list(after.load_average),
        "snapshot_read_seconds": after.read_span_ns / 1_000_000_000,
        "late": elapsed > interval_seconds * 1.5,
    }


def summarize(samples: list[dict]) -> dict:
    if not samples:
        return {
            "covered_seconds": 0.0,
            "cpu_seconds": None,
            "raw_mean_percent": None,
            "normalized_mean_percent": None,
            "raw_peak_percent": None,
            "normalized_peak_percent": None,
            "observer_raw_mean_percent": None,
        }
    wall = sum(sample["wall_seconds"] for sample in samples)
    return {
        "covered_seconds": wall,
        "cpu_seconds": sum(sample["cpu_seconds"] for sample in samples),
        "raw_mean_percent": sum(
            sample["raw_percent"] * sample["wall_seconds"] for sample in samples
        ) / wall,
        "normalized_mean_percent": sum(
            sample["normalized_percent"] * sample["wall_seconds"] for sample in samples
        ) / wall,
        "raw_peak_percent": max(sample["raw_percent"] for sample in samples),
        "normalized_peak_percent": max(sample["normalized_percent"] for sample in samples),
        "observer_raw_mean_percent": sum(
            sample["observer_raw_percent"] * sample["wall_seconds"] for sample in samples
        ) / wall,
    }


def collect(
    pids: tuple[int, ...],
    duration_seconds: float,
    interval_seconds: float,
    ticks_per_second: int,
    capture: Callable[[], Snapshot],
    *,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    if (
        not pids or any(pid <= 0 for pid in pids) or len(set(pids)) != len(pids)
        or not math.isfinite(duration_seconds) or not math.isfinite(interval_seconds)
        or duration_seconds <= 0 or interval_seconds < 0.05
        or duration_seconds > 3600 or math.ceil(duration_seconds / interval_seconds) > 7200
        or ticks_per_second <= 0
    ):
        raise ValueError("Invalid PIDs, sampling duration, interval, or CPU tick rate")

    report = {
        "kind": "explicit_process_cpu",
        "status": "incomplete",
        "error": None,
        "target_pids": list(pids),
        "coverage": "Only explicitly selected PIDs; new or omitted child processes are not discovered.",
        "observer_pid": os.getpid(),
        "ticks_per_second": ticks_per_second,
        "requested_duration_seconds": duration_seconds,
        "sampling_interval_seconds": interval_seconds,
        "online_cpus": None,
        "initial_load_average": None,
        "initial_snapshot_read_seconds": None,
        "start_monotonic_ns": None,
        "end_monotonic_ns": None,
        "samples": [],
    }
    try:
        previous = capture()
        if tuple(process.pid for process in previous.processes) != pids:
            raise MeasurementError("The initial process set does not match the selected PIDs")
        report["online_cpus"] = list(previous.online_cpus)
        report["initial_load_average"] = list(previous.load_average)
        report["initial_snapshot_read_seconds"] = previous.read_span_ns / 1_000_000_000
        report["start_monotonic_ns"] = previous.monotonic_ns
        report["end_monotonic_ns"] = previous.monotonic_ns
        interval_ns = round(interval_seconds * 1_000_000_000)
        end_ns = previous.monotonic_ns + round(duration_seconds * 1_000_000_000)
        next_ns = previous.monotonic_ns + interval_ns
        while previous.monotonic_ns < end_ns:
            deadline_ns = min(next_ns, end_ns)
            remaining_ns = deadline_ns - clock_ns()
            while remaining_ns > 0:
                sleep(remaining_ns / 1_000_000_000)
                remaining_ns = deadline_ns - clock_ns()
            current = capture()
            sample = interval_report(previous, current, ticks_per_second, interval_seconds)
            report["samples"].append(sample)
            report["end_monotonic_ns"] = current.monotonic_ns
            if sample["late"]:
                raise MeasurementError("A sampling interval exceeded 1.5 times the requested interval")
            previous = current
            next_ns += interval_ns
        report["status"] = "complete"
    except MeasurementError as error:
        report["error"] = str(error)
    except KeyboardInterrupt:
        report["error"] = "Measurement interrupted"
    report["summary"] = summarize(report["samples"])
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, action="append", required=True,
                        help="Explicitly selected process PID; repeat for confirmed child processes.")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        ticks_per_second = os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError) as error:
        print(f"Cannot determine CPU tick rate: {error}", file=sys.stderr)
        return 1
    pids = tuple(args.pid)
    try:
        report = collect(
            pids, args.seconds, args.interval, ticks_per_second,
            lambda: capture_snapshot(pids),
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "complete":
        print(f"CPU measurement incomplete: {report['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
