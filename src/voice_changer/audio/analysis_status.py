"""Immutable, device-independent results for the live audio comparison."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisReading:
    bands: tuple[float | None, float | None, float | None] | None
    pitch_hz: float | None
    silent: bool = False
    error: str | None = None


@dataclass(frozen=True)
class AnalysisSnapshot:
    sequence: int
    sample_rate: int
    input: AnalysisReading | None
    output: AnalysisReading | None
    dropped_blocks: int = 0
