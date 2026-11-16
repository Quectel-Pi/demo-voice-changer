"""Local synthetic event detector for M0 latency-method validation only."""

from array import array
from dataclasses import dataclass
import math
from typing import Sequence


EVENT_CODE = (
    1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0,
    1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0,
)
CHIP_MILLISECONDS = 8
HOP_MILLISECONDS = 1
MIN_CORRELATION = 0.75
AMBIGUITY_MARGIN = 0.05
MAX_SIGNAL_SECONDS = 60
MAX_SAMPLE_RATE = 384_000
MAX_ABSOLUTE_SAMPLE = 1_000_000.0


class DetectionError(ValueError):
    """Raised when an event cannot be located unambiguously."""


@dataclass(frozen=True)
class Detection:
    start_frame: int
    score: float
    uncertainty_frames: int


@dataclass(frozen=True)
class LatencyEstimate:
    delay_frames: int
    uncertainty_frames: int
    input_event: Detection
    output_event: Detection


def _validated_signal(samples: Sequence[float], sample_rate: int) -> array:
    if (
        isinstance(samples, (str, bytes, bytearray))
        or not hasattr(samples, "__len__")
        or not hasattr(samples, "__iter__")
    ):
        raise ValueError("samples must be a finite numeric sequence")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or not 0 < sample_rate <= MAX_SAMPLE_RATE
    ):
        raise ValueError("sample_rate must be an integer in the supported range")
    if len(samples) < 1 or len(samples) > sample_rate * MAX_SIGNAL_SECONDS:
        raise ValueError("samples length is outside the supported range")

    prefix_energy = array("d", [0.0])
    total = 0.0
    for sample in samples:
        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            raise ValueError("samples must contain finite numeric values")
        try:
            value = float(sample)
        except OverflowError as exc:
            raise ValueError("samples must contain finite numeric values") from exc
        if not math.isfinite(value) or abs(value) > MAX_ABSOLUTE_SAMPLE:
            raise ValueError("sample is outside the supported numeric range")
        square = value * value
        if not math.isfinite(square) or not math.isfinite(total + square):
            raise ValueError("sample energy exceeds the supported numeric range")
        total += square
        prefix_energy.append(total)
    return prefix_energy


def _correlation(observed: list[float], code: Sequence[int]) -> float:
    count = len(code)
    observed_mean = math.fsum(observed) / count
    code_mean = math.fsum(code) / count
    covariance = math.fsum(
        (value - observed_mean) * (bit - code_mean)
        for value, bit in zip(observed, code)
    )
    observed_variance = math.fsum((value - observed_mean) ** 2 for value in observed)
    code_variance = math.fsum((bit - code_mean) ** 2 for bit in code)
    denominator = math.sqrt(observed_variance * code_variance)
    return covariance / denominator if denominator > 0 else 0.0


def detect_event(samples: Sequence[float], sample_rate: int) -> Detection:
    """Locate one coded envelope in a trace without correlating carrier waveforms."""
    prefix_energy = _validated_signal(samples, sample_rate)
    code = EVENT_CODE
    chip_frames = round(sample_rate * CHIP_MILLISECONDS / 1000)
    hop_frames = round(sample_rate * HOP_MILLISECONDS / 1000)
    event_frames = chip_frames * len(code)
    if chip_frames < 1 or hop_frames < 1 or len(prefix_energy) - 1 < event_frames:
        raise DetectionError("signal is too short to contain the coded event")

    candidates = []
    last_start = len(prefix_energy) - 1 - event_frames
    for start in range(0, last_start + 1, hop_frames):
        envelope = []
        for chip in range(len(code)):
            first = start + chip * chip_frames
            after = first + chip_frames
            energy = prefix_energy[after] - prefix_energy[first]
            envelope.append(math.sqrt(max(0.0, energy / chip_frames)))
        candidates.append((start, _correlation(envelope, code)))

    best_start, best_score = max(candidates, key=lambda item: item[1])
    if best_score < MIN_CORRELATION:
        raise DetectionError(
            f"coded event not detected: best correlation {best_score:.3f}"
        )

    local_competitors = [
        score for start, score in candidates
        if hop_frames < abs(start - best_start) < event_frames
    ]
    local_second_score = max(local_competitors, default=None)
    if (
        local_second_score is not None
        and best_score - local_second_score < AMBIGUITY_MARGIN
    ):
        raise DetectionError("coded event localization is ambiguous")

    competitors = [
        score for start, score in candidates
        if abs(start - best_start) >= event_frames
    ]
    second_score = max(competitors, default=None)
    if second_score is not None and best_score - second_score < AMBIGUITY_MARGIN:
        raise DetectionError("coded event is ambiguous")

    return Detection(
        start_frame=best_start,
        score=best_score,
        uncertainty_frames=hop_frames,
    )


def estimate_latency(input_reference: Sequence[float], output_reference: Sequence[float],
                     sample_rate: int) -> LatencyEstimate:
    """Estimate output minus input event position and report summed grid steps."""
    input_event = detect_event(input_reference, sample_rate)
    output_event = detect_event(output_reference, sample_rate)
    return LatencyEstimate(
        delay_frames=output_event.start_frame - input_event.start_frame,
        uncertainty_frames=(
            input_event.uncertainty_frames + output_event.uncertainty_frames
        ),
        input_event=input_event,
        output_event=output_event,
    )
