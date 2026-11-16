"""Record measured latency facts; no thresholds and no pass/fail verdicts.

The demo records what it can measure instead of claiming a latency target. A
configuration value such as ``blocksize / samplerate`` is a *scheduling* fact,
not an end-to-end measurement, so both are kept separate.
"""

from __future__ import annotations

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class LatencyFacts:
    """One measured configuration; ``None`` means "not measured", never zero."""

    samplerate: int
    blocksize: int
    channels: int
    buffer_ms: float
    dsp_startup_frames: int
    dsp_startup_ms: float
    stream_input_ms: float | None = None
    stream_output_ms: float | None = None
    callback_peak_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("samplerate", "blocksize", "channels", "dsp_startup_frames"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("buffer_ms", "dsp_startup_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative number")
        for name in ("stream_input_ms", "stream_output_ms", "callback_peak_seconds"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be None or a nonnegative number")

    @property
    def buffer_only_ms(self) -> float:
        return self.blocksize / self.samplerate * 1000

    def documented(self) -> dict[str, object]:
        return {
            "samplerate": self.samplerate,
            "blocksize": self.blocksize,
            "channels": self.channels,
            "scheduling_buffer_ms": self.buffer_ms,
            "dsp_startup_frames": self.dsp_startup_frames,
            "dsp_startup_ms": self.dsp_startup_ms,
            "stream_reported_input_ms": self.stream_input_ms,
            "stream_reported_output_ms": self.stream_output_ms,
            "callback_peak_seconds": self.callback_peak_seconds,
            "end_to_end_acoustic_ms": None,
            "note": (
                "Scheduling and driver-reported values only. End-to-end acoustic "
                "latency was not measured; no threshold is applied."
            ),
        }


def startup_frames(blocksize: int, start_pad: int) -> int:
    """The reserve the streaming processor inserts before its first output."""
    if type(blocksize) is not int or blocksize <= 0:
        raise ValueError("blocksize must be a positive integer")
    if type(start_pad) is not int or start_pad < 0:
        raise ValueError("start_pad must be a nonnegative integer")
    return max(4 * blocksize, 2 * start_pad)


def measure_facts(
    *, samplerate: int, blocksize: int, channels: int, start_pad: int,
    stream_input_ms: float | None = None, stream_output_ms: float | None = None,
    callback_peak_seconds: float | None = None,
) -> LatencyFacts:
    frames = startup_frames(blocksize, start_pad)
    return LatencyFacts(
        samplerate=samplerate, blocksize=blocksize, channels=channels,
        buffer_ms=blocksize / samplerate * 1000,
        dsp_startup_frames=frames,
        dsp_startup_ms=frames / samplerate * 1000,
        stream_input_ms=stream_input_ms, stream_output_ms=stream_output_ms,
        callback_peak_seconds=callback_peak_seconds,
    )


def write_report(facts: LatencyFacts, path) -> None:
    from pathlib import Path

    Path(path).write_text(
        json.dumps(facts.documented(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
