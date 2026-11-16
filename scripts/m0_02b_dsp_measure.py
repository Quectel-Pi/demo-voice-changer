"""Opt-in in-memory Rubber Band parameter probe; never opens audio devices or files."""

import argparse
from array import array
import bisect
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import platform
import sys
import time


RATE = 48_000
BLOCK_SIZE = 1_024
BLOCK_SIZE_CHOICES = (256, 512, 1_024)
INPUT_FRAMES = 2 * RATE
ANALYSIS_FRAMES = RATE
FFT_SIZE = 262_144
TRANSIENT_FRAMES = RATE // 4
FORMANT_PRESERVED = 0x01000000
OPTIONS = 0x00000001 | 0x00010000 | 0x20000000 | FORMANT_PRESERVED
DEFAULT_LIBRARY = "/usr/lib/aarch64-linux-gnu/librubberband.so.2"
BASE_FORMANT_BANDS = ((500.0, 1_000.0), (1_100.0, 1_900.0), (2_200.0, 2_900.0))
SCALED_FORMANT_BANDS = ((700.0, 1_050.0), (1_500.0, 2_250.0), (2_700.0, 3_750.0))
FORMANT_CENTERS = (700.0, 1_500.0, 2_500.0)
FORMANT_WIDTHS = (90.0, 140.0, 180.0)


def make_pitch_probe():
    return array(
        "f",
        (0.1 * math.cos(2.0 * math.pi * 220.0 * frame / RATE)
         for frame in range(INPUT_FRAMES)),
    )


def make_formant_probe():
    samples = array("d", [0.0]) * INPUT_FRAMES
    for harmonic in range(1, 67):
        frequency = 120.0 * harmonic
        envelope = 0.02 + sum(
            math.exp(-0.5 * ((frequency - center) / width) ** 2)
            for center, width in zip(FORMANT_CENTERS, FORMANT_WIDTHS)
        )
        amplitude = envelope / harmonic
        phase = 2.0 * math.pi * ((harmonic * math.sqrt(2.0)) % 1.0)
        step = 2.0 * math.pi * frequency / RATE
        cos_step, sin_step = math.cos(step), math.sin(step)
        cos_phase, sin_phase = math.cos(phase), math.sin(phase)
        for frame in range(INPUT_FRAMES):
            samples[frame] += amplitude * cos_phase
            cos_phase, sin_phase = (
                cos_phase * cos_step - sin_phase * sin_step,
                sin_phase * cos_step + cos_phase * sin_step,
            )
    peak = max(abs(sample) for sample in samples)
    if not math.isfinite(peak) or peak <= 0.0:
        raise RuntimeError("Generated formant probe has an invalid peak")
    scale = 0.1 / peak
    return array("f", (sample * scale for sample in samples))


def parabolic_peak_frequency(frequencies, values, index):
    if index <= 0 or index >= len(values) - 1:
        return None
    left, center, right = (
        math.log(max(values[index - 1], 1e-300)),
        math.log(max(values[index], 1e-300)),
        math.log(max(values[index + 1], 1e-300)),
    )
    denominator = left - 2.0 * center + right
    if denominator == 0.0:
        return frequencies[index]
    offset = 0.5 * (left - right) / denominator
    if not math.isfinite(offset) or abs(offset) > 1.0:
        return None
    spacing = frequencies[index + 1] - frequencies[index]
    return frequencies[index] + offset * spacing


def find_spectral_peak(frequencies, power, low_hz, high_hz):
    first = bisect.bisect_left(frequencies, low_hz)
    after_last = bisect.bisect_right(frequencies, high_hz)
    if first < 1 or after_last >= len(power) or first >= after_last:
        return None
    index = max(range(first, after_last), key=power.__getitem__)
    return parabolic_peak_frequency(frequencies, power, index)


def smooth_power_by_octave(frequencies, power, low_hz=400.0, high_hz=4_000.0):
    prefix = [0.0]
    for value in power:
        prefix.append(prefix[-1] + value)
    half_width = 2.0 ** (1.0 / 12.0)
    start = bisect.bisect_left(frequencies, low_hz)
    end = bisect.bisect_right(frequencies, high_hz)
    selected_frequencies = frequencies[start:end]
    smoothed = []
    for frequency in selected_frequencies:
        left = bisect.bisect_left(frequencies, frequency / half_width)
        right = bisect.bisect_right(frequencies, frequency * half_width)
        smoothed.append((prefix[right] - prefix[left]) / max(right - left, 1))
    return selected_frequencies, smoothed


def find_local_envelope_peak(frequencies, values, low_hz, high_hz):
    first = bisect.bisect_left(frequencies, low_hz)
    after_last = bisect.bisect_right(frequencies, high_hz)
    local_maxima = [
        index for index in range(max(first, 1), min(after_last, len(values) - 1))
        if (
            values[index] >= values[index - 1]
            and values[index] >= values[index + 1]
            and (values[index] > values[index - 1] or values[index] > values[index + 1])
        )
    ]
    if not local_maxima:
        return None
    index = max(local_maxima, key=values.__getitem__)
    return parabolic_peak_frequency(frequencies, values, index)


def analyze_window(audio, start, expected_f0, formant_bands):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for FFT analysis; do not install it as part of this task") from exc

    window = np.frombuffer(audio, dtype=np.float32, count=ANALYSIS_FRAMES, offset=start * 4)
    if window.size != ANALYSIS_FRAMES:
        return None
    window = window.astype(np.float64, copy=True) * np.hanning(ANALYSIS_FRAMES)
    spectrum = np.fft.rfft(window, n=FFT_SIZE)
    max_bin = int(4_000.0 * FFT_SIZE / RATE)
    power = (spectrum.real[:max_bin + 1] ** 2 + spectrum.imag[:max_bin + 1] ** 2).tolist()
    frequencies = [index * RATE / FFT_SIZE for index in range(len(power))]
    f0_hz = find_spectral_peak(
        frequencies,
        power,
        expected_f0 * 0.85,
        expected_f0 * 1.15,
    )
    formant_peaks = None
    if formant_bands is not None:
        envelope_frequencies, envelope_power = smooth_power_by_octave(frequencies, power)
        formant_peaks = [
            find_local_envelope_peak(envelope_frequencies, envelope_power, low, high)
            for low, high in formant_bands
        ]
    return {"f0_hz": f0_hz, "formant_peaks_hz": formant_peaks}


def _finite_samples(samples, label):
    if isinstance(samples, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence of finite numeric samples")
    try:
        values = [float(sample) for sample in samples]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a sequence of finite numeric samples") from exc
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"{label}[{index}] is not finite")
    return values


def _distribution_summary(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50_nearest_rank": None,
            "p95_nearest_rank": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": math.fsum(ordered) / len(ordered),
        "p50_nearest_rank": ordered[max(0, math.ceil(0.50 * len(ordered)) - 1)],
        "p95_nearest_rank": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "max": ordered[-1],
    }


def _elapsed_ns(started_ns, finished_ns, label):
    elapsed_ns = finished_ns - started_ns
    if elapsed_ns < 0:
        raise RuntimeError(f"{label} monotonic timer moved backwards")
    return elapsed_ns


def _host_load_snapshot():
    try:
        online_logical_cpu_count = os.sysconf("SC_NPROCESSORS_ONLN")
    except (OSError, ValueError) as exc:
        raise RuntimeError("Unable to read online logical CPU count") from exc
    if (
        isinstance(online_logical_cpu_count, bool)
        or not isinstance(online_logical_cpu_count, int)
        or online_logical_cpu_count <= 0
    ):
        raise RuntimeError("Online logical CPU count must be a positive integer")
    try:
        load_averages = os.getloadavg()
    except OSError as exc:
        raise RuntimeError("Unable to read system load averages") from exc
    if len(load_averages) != 3 or any(
        not math.isfinite(value) or value < 0.0 for value in load_averages
    ):
        raise RuntimeError("System load averages must be three finite non-negative values")
    return {
        "online_logical_cpu_count": online_logical_cpu_count,
        "load_average_1m": load_averages[0],
        "load_average_5m": load_averages[1],
        "load_average_15m": load_averages[2],
    }


def _retrieve_distribution_summary(values):
    summary = _distribution_summary(values)
    summary["values"] = values
    return summary


def _validate_retrieve_segments(segments, total_output_frames):
    if isinstance(total_output_frames, bool) or not isinstance(total_output_frames, int):
        raise ValueError("total_output_frames must be a non-negative integer")
    if total_output_frames < 0:
        raise ValueError("total_output_frames must be a non-negative integer")

    next_frame = 0
    total_received = 0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"retrieve segment {index} must be an object")
        start = segment.get("output_start_frame")
        end = segment.get("output_end_frame_exclusive")
        received = segment.get("received_frames")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end, received)):
            raise ValueError(f"retrieve segment {index} has invalid frame indexes")
        if start != next_frame or end <= start or received <= 0 or end - start != received:
            raise ValueError(f"retrieve segment {index} has a gap, overlap, or invalid frame count")
        available = segment.get("available_frames")
        requested = segment.get("requested_frames")
        if (
            isinstance(available, bool)
            or not isinstance(available, int)
            or isinstance(requested, bool)
            or not isinstance(requested, int)
            or available < received
            or requested != available
        ):
            raise ValueError(f"retrieve segment {index} has invalid available/requested frame counts")
        next_frame = end
        total_received += received

    if next_frame != total_output_frames or total_received != total_output_frames:
        raise ValueError(
            "retrieve segment frame totals do not match the concatenated output"
        )


def retrieve_segment_boundary_diagnostics(samples, segments):
    values = _finite_samples(samples, "samples")
    _validate_retrieve_segments(segments, len(values))

    boundary_indices = [
        segment["output_start_frame"]
        for segment in segments[1:]
    ]
    boundary_deltas = [
        abs(values[index] - values[index - 1])
        for index in boundary_indices
        if index > 0
    ]
    interior_indices = []
    for segment in segments:
        start = segment["output_start_frame"]
        end = segment["output_end_frame_exclusive"]
        midpoint = start + (end - start) // 2
        if midpoint < start + 2 or midpoint > end - 2:
            continue
        if any(abs(midpoint - boundary) < 2 for boundary in boundary_indices):
            continue
        interior_indices.append(midpoint)

    boundary_slope_jumps = [
        abs(
            (values[index] - values[index - 1])
            - (values[index - 1] - values[index - 2])
        )
        for index in boundary_indices
        if index >= 2
    ]
    interior_deltas = [
        abs(values[index] - values[index - 1])
        for index in interior_indices
    ]
    interior_slope_jumps = [
        abs(
            (values[index] - values[index - 1])
            - (values[index - 1] - values[index - 2])
        )
        for index in interior_indices
        if index >= 2
    ]
    return {
        "status": "measured" if boundary_indices else "insufficient_boundaries",
        "diagnostic_only": True,
        "boundary_definition": (
            "start frame of each non-first positive rubberband_retrieve return; "
            "not rubberband_process or audio callback boundaries"
        ),
        "retrieve_segment_count": len(segments),
        "output_frame_count": len(values),
        "boundary_frame_indices": boundary_indices,
        "interior_frame_indices": interior_indices,
        "boundary_adjacent_delta": _retrieve_distribution_summary(boundary_deltas),
        "interior_adjacent_delta": _retrieve_distribution_summary(interior_deltas),
        "boundary_slope_jump": _retrieve_distribution_summary(boundary_slope_jumps),
        "interior_slope_jump": _retrieve_distribution_summary(interior_slope_jumps),
    }


def output_block_boundary_diagnostics(samples, block_size):
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    values = _finite_samples(samples, "samples")
    if len(values) < 2:
        return {
            "status": "insufficient_samples",
            "diagnostic_only": True,
            "block_size_frames": block_size,
            "boundary_definition": (
                "frame indices divisible by block_size in concatenated output; "
                "not API call boundaries"
            ),
            "boundary_frame_indices": [],
            "boundary_adjacent_delta": _distribution_summary([]),
            "interior_adjacent_delta": _distribution_summary([]),
            "boundary_slope_jump": _distribution_summary([]),
            "interior_slope_jump": _distribution_summary([]),
        }

    boundary_indices = list(range(block_size, len(values), block_size))
    boundary_set = set(boundary_indices)
    adjacent_deltas = [
        abs(values[index] - values[index - 1])
        for index in range(1, len(values))
    ]
    boundary_deltas = [
        adjacent_deltas[index - 1]
        for index in boundary_indices
    ]
    interior_deltas = [
        delta
        for index, delta in enumerate(adjacent_deltas, start=1)
        if index not in boundary_set
    ]
    boundary_slope_jumps = [
        abs(
            (values[index] - values[index - 1])
            - (values[index - 1] - values[index - 2])
        )
        for index in boundary_indices
        if index >= 2
    ]
    interior_slope_jumps = [
        abs(
            (values[index] - values[index - 1])
            - (values[index - 1] - values[index - 2])
        )
        for index in range(2, len(values))
        if index not in boundary_set
    ]
    return {
        "status": "measured" if boundary_indices else "insufficient_boundaries",
        "diagnostic_only": True,
        "block_size_frames": block_size,
        "boundary_definition": (
            "frame indices divisible by block_size in concatenated output; "
            "not API call boundaries"
        ),
        "boundary_frame_indices": boundary_indices,
        "boundary_adjacent_delta": _distribution_summary(boundary_deltas),
        "interior_adjacent_delta": _distribution_summary(interior_deltas),
        "boundary_slope_jump": _distribution_summary(boundary_slope_jumps),
        "interior_slope_jump": _distribution_summary(interior_slope_jumps),
    }


def reset_fresh_diagnostics(reset_samples, fresh_samples):
    reset_values = _finite_samples(reset_samples, "reset_samples")
    fresh_values = _finite_samples(fresh_samples, "fresh_samples")
    compared_frames = min(len(reset_values), len(fresh_values))
    reset_compared = reset_values[:compared_frames]
    fresh_compared = fresh_values[:compared_frames]
    differences = [
        abs(reset_value - fresh_value)
        for reset_value, fresh_value in zip(reset_compared, fresh_compared)
    ]
    if compared_frames:
        mean_abs_difference = math.fsum(differences) / compared_frames
        rms_difference = math.sqrt(
            math.fsum(difference * difference for difference in differences)
            / compared_frames
        )
        max_abs_difference = max(differences)
        fresh_rms = math.sqrt(
            math.fsum(sample * sample for sample in fresh_compared) / compared_frames
        )
    else:
        mean_abs_difference = None
        rms_difference = None
        max_abs_difference = None
        fresh_rms = None
    return {
        "status": "measured" if compared_frames else "insufficient_samples",
        "diagnostic_only": True,
        "comparison": "index-aligned overlapping samples; caller must align and trim transients",
        "reset_frame_count": len(reset_values),
        "fresh_instance_frame_count": len(fresh_values),
        "compared_frame_count": compared_frames,
        "uncompared_reset_frame_count": len(reset_values) - compared_frames,
        "uncompared_fresh_instance_frame_count": len(fresh_values) - compared_frames,
        "mean_abs_difference": mean_abs_difference,
        "rms_difference": rms_difference,
        "max_abs_difference": max_abs_difference,
        "fresh_instance_rms": fresh_rms,
        "normalized_rms_difference": (
            None if not compared_frames or fresh_rms == 0.0
            else rms_difference / fresh_rms
        ),
        "normalization_status": (
            "insufficient_samples" if not compared_frames
            else "zero_reference" if fresh_rms == 0.0
            else "measured"
        ),
    }


def _validate_scale_parameters(pitch_scale, formant_scale):
    def finite_number(name, value):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite number")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number")
        return number

    pitch_scale = finite_number("pitch_scale", pitch_scale)
    if not 0.5 <= pitch_scale <= 2.0:
        raise ValueError(
            "pitch_scale must be between 0.5 and 2.0 (the PRD ±12-semitone target)"
        )

    formant_scale = finite_number("formant_scale", formant_scale)
    if formant_scale < 0.0:
        raise ValueError(
            "formant_scale must be non-negative; 0 selects Rubber Band automatic scaling"
        )
    return pitch_scale, formant_scale


def _validate_block_size(block_size):
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ValueError(f"block_size must be one of {BLOCK_SIZE_CHOICES}")
    if block_size not in BLOCK_SIZE_CHOICES:
        raise ValueError(f"block_size must be one of {BLOCK_SIZE_CHOICES}")
    return block_size


def _configure_state(api, state, pitch_scale, formant_scale, block_size=BLOCK_SIZE):
    api.rubberband_set_max_process_size(state, block_size)
    api.rubberband_set_pitch_scale(state, pitch_scale)
    api.rubberband_set_formant_scale(state, formant_scale)
    api.rubberband_set_formant_option(state, FORMANT_PRESERVED)


def _drain_state(
    ffi,
    api,
    state,
    audio,
    label,
    block_size_frames,
    retrieve_segments,
    retrieve_call_offset,
    process_call_index,
    process_final,
    drain_invocation_index,
    drain_phase,
    timing=None,
):
    drain_started_ns = time.perf_counter_ns()
    retrieve_calls = 0
    available_api_ns = []
    available_frames = []
    while True:
        available_started_ns = time.perf_counter_ns()
        available = int(api.rubberband_available(state))
        available_api_ns.append(
            _elapsed_ns(
                available_started_ns,
                time.perf_counter_ns(),
                "rubberband_available",
            )
        )
        available_frames.append(available)
        if available <= 0:
            if timing is not None:
                timing.update({
                    "drain_wall_ns": _elapsed_ns(
                        drain_started_ns,
                        time.perf_counter_ns(),
                        "drain",
                    ),
                    "available_api_ns": available_api_ns,
                    "available_frames": available_frames,
                })
            return retrieve_calls
        output = ffi.new("float[]", available)
        output_vector = ffi.new("float *[1]")
        output_vector[0] = output
        retrieve_started_ns = time.perf_counter_ns()
        received = int(api.rubberband_retrieve(state, output_vector, available))
        retrieve_api_ns = _elapsed_ns(
            retrieve_started_ns,
            time.perf_counter_ns(),
            "rubberband_retrieve",
        )
        retrieve_calls += 1
        if received < 0 or received > available:
            raise RuntimeError(
                f"{label}: invalid retrieve count {received} for {available} available frames"
            )
        if received == 0:
            raise RuntimeError(
                f"{label}: Rubber Band reported {available} available frames "
                "but retrieve returned 0"
            )
        output_start_frame = len(audio)
        audio.frombytes(bytes(ffi.buffer(output, received * ffi.sizeof("float"))))
        retrieve_segments.append({
            "case": label,
            "block_size_frames": block_size_frames,
            "retrieve_call_index": retrieve_call_offset + retrieve_calls,
            "drain_invocation_index": drain_invocation_index,
            "drain_phase": drain_phase,
            "process_call_index": process_call_index,
            "process_final": bool(process_final),
            "available_frames": available,
            "requested_frames": available,
            "received_frames": received,
            "retrieve_api_ns": retrieve_api_ns,
            "output_start_frame": output_start_frame,
            "output_end_frame_exclusive": output_start_frame + received,
        })


def _process_signal(
    ffi,
    api,
    state,
    signal,
    start_pad_frames,
    final=True,
    block_size=BLOCK_SIZE,
    drain_output=True,
    label="Rubber Band stream",
):
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if isinstance(start_pad_frames, bool) or not isinstance(start_pad_frames, int):
        raise ValueError("start_pad_frames must be a non-negative integer")
    if start_pad_frames < 0:
        raise ValueError("start_pad_frames must be a non-negative integer")
    if not signal:
        raise RuntimeError(f"{label}: input signal is empty")

    audio = array("f")
    process_calls = 0
    retrieve_calls = 0
    drain_invocations = 0
    retrieve_segments = []
    process_blocks = []
    drain_timings = []

    def drain(drain_phase, process_call_index, process_final):
        nonlocal retrieve_calls, drain_invocations
        drain_invocations += 1
        timing = {}
        retrieve_calls += _drain_state(
            ffi,
            api,
            state,
            audio,
            label,
            block_size,
            retrieve_segments,
            retrieve_calls,
            process_call_index,
            process_final,
            drain_invocations,
            drain_phase,
            timing,
        )
        drain_record = {
            "case": label,
            "block_size_frames": block_size,
            "drain_invocation_index": drain_invocations,
            "drain_phase": drain_phase,
            "process_call_index": process_call_index,
            "process_final": bool(process_final),
            **timing,
        }
        drain_timings.append(drain_record)
        return drain_record

    def process(samples, is_final=False):
        nonlocal process_calls
        if not samples or len(samples) > block_size:
            raise RuntimeError(f"{label}: invalid process block size {len(samples)}")
        process_block_started_ns = time.perf_counter_ns()
        input_vector = ffi.new("const float *[1]")
        input_vector[0] = ffi.from_buffer("const float[]", samples)
        process_api_started_ns = time.perf_counter_ns()
        api.rubberband_process(state, input_vector, len(samples), int(is_final))
        process_api_ns = _elapsed_ns(
            process_api_started_ns,
            time.perf_counter_ns(),
            "rubberband_process",
        )
        process_calls += 1
        drain_record = None
        if drain_output:
            drain_record = drain("after_process", process_calls, is_final)
        process_block_wall_ns = _elapsed_ns(
            process_block_started_ns,
            time.perf_counter_ns(),
            "process block",
        )
        budget_ns = len(samples) * 1_000_000_000 / RATE
        process_blocks.append({
            "case": label,
            "block_size_frames": block_size,
            "process_call_index": process_calls,
            "process_final": bool(is_final),
            "input_frames": len(samples),
            "budget_ns": budget_ns,
            "process_api_ns": process_api_ns,
            "after_process_drain_invocation_index": (
                None if drain_record is None
                else drain_record["drain_invocation_index"]
            ),
            "after_process_drain_wall_ns": (
                None if drain_record is None
                else drain_record["drain_wall_ns"]
            ),
            "process_block_wall_ns": process_block_wall_ns,
            "over_budget": process_block_wall_ns > budget_ns,
        })

    remaining_pad = start_pad_frames
    zeros = array("f", [0.0]) * block_size
    while remaining_pad:
        count = min(block_size, remaining_pad)
        process(memoryview(zeros)[:count])
        remaining_pad -= count

    for offset in range(0, len(signal), block_size):
        count = min(block_size, len(signal) - offset)
        process(
            memoryview(signal)[offset:offset + count],
            final and offset + count == len(signal),
        )
    if drain_output:
        drain(
            "post_final_drain" if final else "post_input_drain",
            process_calls,
            final,
        )
    return {
        "audio": audio,
        "process_calls": process_calls,
        "retrieve_calls": retrieve_calls,
        "retrieve_segments": retrieve_segments,
        "process_blocks": process_blocks,
        "drain_timings": drain_timings,
    }


def _trim_comparison_window(audio, start_delay_frames):
    start = start_delay_frames + TRANSIENT_FRAMES
    return audio[start:start + ANALYSIS_FRAMES], start


def load_api(library_path):
    try:
        from cffi import FFI
    except ImportError as exc:
        raise RuntimeError("CFFI is required to call the Rubber Band C API") from exc

    ffi = FFI()
    ffi.cdef(
        """
        typedef struct RubberBandState_ *RubberBandState;
        RubberBandState rubberband_new(unsigned int, unsigned int, int, double, double);
        void rubberband_delete(RubberBandState);
        void rubberband_reset(RubberBandState);
        int rubberband_get_engine_version(RubberBandState);
        void rubberband_set_pitch_scale(RubberBandState, double);
        void rubberband_set_formant_scale(RubberBandState, double);
        void rubberband_set_formant_option(RubberBandState, int);
        unsigned int rubberband_get_preferred_start_pad(const RubberBandState);
        unsigned int rubberband_get_start_delay(const RubberBandState);
        unsigned int rubberband_get_latency(const RubberBandState);
        void rubberband_set_max_process_size(RubberBandState, unsigned int);
        void rubberband_process(RubberBandState, const float *const *, unsigned int, int);
        int rubberband_available(const RubberBandState);
        unsigned int rubberband_retrieve(const RubberBandState, float *const *, unsigned int);
        """
    )
    return ffi, ffi.dlopen(library_path)


def run_case(
    ffi,
    api,
    name,
    signal,
    pitch_scale,
    formant_scale,
    expected_f0,
    formant_bands,
    block_size=BLOCK_SIZE,
):
    pitch_scale, formant_scale = _validate_scale_parameters(pitch_scale, formant_scale)
    block_size = _validate_block_size(block_size)
    state = api.rubberband_new(RATE, 1, OPTIONS, 1.0, 1.0)
    if state == ffi.NULL:
        raise RuntimeError(f"{name}: Rubber Band returned a null state")

    started_wall_ns = time.perf_counter_ns()
    started_cpu_ns = time.process_time_ns()
    try:
        _configure_state(api, state, pitch_scale, formant_scale, block_size)
        engine_version = int(api.rubberband_get_engine_version(state))
        start_pad = int(api.rubberband_get_preferred_start_pad(state))
        start_delay = int(api.rubberband_get_start_delay(state))
        latency = int(api.rubberband_get_latency(state))
        stream = _process_signal(
            ffi, api, state, signal, start_pad, block_size=block_size, label=name
        )
        audio = stream["audio"]
        elapsed_wall_ns = _elapsed_ns(
            started_wall_ns,
            time.perf_counter_ns(),
            f"{name} processing wall",
        )
        elapsed_cpu_ns = _elapsed_ns(
            started_cpu_ns,
            time.process_time_ns(),
            f"{name} processing CPU",
        )
        if elapsed_wall_ns <= 0:
            raise RuntimeError(f"{name}: processing wall time must be positive")

        if engine_version != 3:
            raise RuntimeError(f"{name}: expected Rubber Band engine version 3, got {engine_version}")
        if not audio or any(not math.isfinite(sample) for sample in audio):
            raise RuntimeError(f"{name}: empty or non-finite DSP output")
        peak = max(abs(sample) for sample in audio)
        if peak <= 1e-8:
            raise RuntimeError(f"{name}: DSP output is silent")

        analysis_start = start_delay + TRANSIENT_FRAMES
        analysis_end = analysis_start + ANALYSIS_FRAMES
        has_window = analysis_end + TRANSIENT_FRAMES <= len(audio)
        features = (
            analyze_window(audio, analysis_start, expected_f0, formant_bands)
            if has_window else None
        )
        rms = math.sqrt(math.fsum(sample * sample for sample in audio) / len(audio))
        max_delta = max(
            (abs(audio[index] - audio[index - 1]) for index in range(1, len(audio))),
            default=0.0,
        )
        try:
            retrieve_diagnostics = retrieve_segment_boundary_diagnostics(
                audio, stream["retrieve_segments"]
            )
        except ValueError as exc:
            raise RuntimeError(
                f"{name}: retrieve segment accounting failed: {exc}"
            ) from exc
        process_blocks = stream["process_blocks"]
        drain_timings = stream["drain_timings"]
        available_api_times = [
            elapsed_ns
            for drain_timing in drain_timings
            for elapsed_ns in drain_timing["available_api_ns"]
        ]
        timing_summary = {
            "process_api_ns": _distribution_summary(
                [block["process_api_ns"] for block in process_blocks]
            ),
            "available_api_ns": _distribution_summary(available_api_times),
            "retrieve_api_ns": _distribution_summary(
                [segment["retrieve_api_ns"] for segment in stream["retrieve_segments"]]
            ),
            "drain_wall_ns": _distribution_summary(
                [drain_timing["drain_wall_ns"] for drain_timing in drain_timings]
            ),
            "process_block_wall_ns": _distribution_summary(
                [block["process_block_wall_ns"] for block in process_blocks]
            ),
        }
        return {
            "case": name,
            "pitch_scale": pitch_scale,
            "formant_scale": formant_scale,
            "formant_option": "FormantPreserved",
            "engine_version": engine_version,
            "input_frames": len(signal),
            "block_size_frames": block_size,
            "output_frames": len(audio),
            "process_calls": stream["process_calls"],
            "retrieve_calls": stream["retrieve_calls"],
            "process_blocks": process_blocks,
            "drain_timings": drain_timings,
            "timing_summary": timing_summary,
            "process_block_over_budget_count": sum(
                block["over_budget"] for block in process_blocks
            ),
            "max_available_frames": max(
                available
                for drain_timing in drain_timings
                for available in drain_timing["available_frames"]
            ),
            "total_retrieved_frames": sum(
                segment["received_frames"] for segment in stream["retrieve_segments"]
            ),
            "process_cpu_ns": elapsed_cpu_ns,
            "process_wall_ns": elapsed_wall_ns,
            "processing_seconds": elapsed_wall_ns / 1_000_000_000,
            "start_pad_frames": start_pad,
            "start_delay_frames": start_delay,
            "reported_latency_frames": latency,
            "analysis_status": "measured" if features is not None else "indeterminate_insufficient_output",
            "f0_hz": None if features is None else features["f0_hz"],
            "formant_peaks_hz": None if features is None else features["formant_peaks_hz"],
            "rms": rms,
            "peak": peak,
            "max_adjacent_delta": max_delta,
            "retrieve_segments": stream["retrieve_segments"],
            "retrieve_segment_boundary_diagnostics": retrieve_diagnostics,
            "output_block_boundary_diagnostics": output_block_boundary_diagnostics(
                audio, block_size
            ),
        }
    finally:
        api.rubberband_delete(state)


def run_reset_comparison(
    ffi,
    api,
    history_signal,
    comparison_signal,
    pitch_scale=1.0,
    formant_scale=1.0,
    drain_history_output=True,
    block_size=BLOCK_SIZE,
):
    pitch_scale, formant_scale = _validate_scale_parameters(pitch_scale, formant_scale)
    block_size = _validate_block_size(block_size)
    try:
        reset_api = api.rubberband_reset
    except AttributeError as exc:
        raise RuntimeError("Rubber Band library does not expose rubberband_reset") from exc

    reset_state = api.rubberband_new(RATE, 1, OPTIONS, 1.0, 1.0)
    if reset_state == ffi.NULL:
        raise RuntimeError("reset comparison: Rubber Band returned a null reset state")
    fresh_state = ffi.NULL
    try:
        _configure_state(api, reset_state, pitch_scale, formant_scale, block_size)
        reset_engine_version = int(api.rubberband_get_engine_version(reset_state))
        history_start_pad = int(api.rubberband_get_preferred_start_pad(reset_state))
        history_stream = _process_signal(
            ffi,
            api,
            reset_state,
            history_signal,
            history_start_pad,
            final=False,
            block_size=block_size,
            drain_output=drain_history_output,
            label="reset comparison history",
        )
        history_available_before_reset = int(api.rubberband_available(reset_state))
        reset_api(reset_state)
        history_available_after_reset = int(api.rubberband_available(reset_state))
        _configure_state(api, reset_state, pitch_scale, formant_scale, block_size)
        reset_start_pad = int(api.rubberband_get_preferred_start_pad(reset_state))
        reset_start_delay = int(api.rubberband_get_start_delay(reset_state))

        fresh_state = api.rubberband_new(RATE, 1, OPTIONS, 1.0, 1.0)
        if fresh_state == ffi.NULL:
            raise RuntimeError("reset comparison: Rubber Band returned a null fresh state")
        _configure_state(api, fresh_state, pitch_scale, formant_scale, block_size)
        fresh_engine_version = int(api.rubberband_get_engine_version(fresh_state))
        fresh_start_pad = int(api.rubberband_get_preferred_start_pad(fresh_state))
        fresh_start_delay = int(api.rubberband_get_start_delay(fresh_state))

        reset_stream = _process_signal(
            ffi,
            api,
            reset_state,
            comparison_signal,
            reset_start_pad,
            block_size=block_size,
            label="reset comparison reset state",
        )
        fresh_stream = _process_signal(
            ffi,
            api,
            fresh_state,
            comparison_signal,
            fresh_start_pad,
            block_size=block_size,
            label="reset comparison fresh state",
        )
        if reset_engine_version != 3 or fresh_engine_version != 3:
            raise RuntimeError(
                "reset comparison: expected Rubber Band engine version 3 for both states"
            )

        reset_window, reset_window_start = _trim_comparison_window(
            reset_stream["audio"], reset_start_delay
        )
        fresh_window, fresh_window_start = _trim_comparison_window(
            fresh_stream["audio"], fresh_start_delay
        )
        try:
            diagnostics = reset_fresh_diagnostics(reset_window, fresh_window)
        except ValueError as exc:
            raise RuntimeError(f"reset comparison produced invalid output: {exc}") from exc
        return {
            "status": diagnostics["status"],
            "diagnostic_only": True,
            "history_input_frames": len(history_signal),
            "comparison_input_frames": len(comparison_signal),
            "history_start_pad_frames": history_start_pad,
            "history_output_policy": (
                "drained_before_reset" if drain_history_output
                else "left_pending_for_reset"
            ),
            "history_retrieved_frames": len(history_stream["audio"]),
            "history_available_before_reset": history_available_before_reset,
            "history_available_after_reset": history_available_after_reset,
            "reset_start_pad_frames": reset_start_pad,
            "fresh_start_pad_frames": fresh_start_pad,
            "reset_start_delay_frames": reset_start_delay,
            "fresh_start_delay_frames": fresh_start_delay,
            "reset_output_frames": len(reset_stream["audio"]),
            "fresh_output_frames": len(fresh_stream["audio"]),
            "reset_process_calls": reset_stream["process_calls"],
            "reset_retrieve_calls": reset_stream["retrieve_calls"],
            "fresh_process_calls": fresh_stream["process_calls"],
            "fresh_retrieve_calls": fresh_stream["retrieve_calls"],
            "comparison_window_start_frames": {
                "reset": reset_window_start,
                "fresh": fresh_window_start,
            },
            "comparison": diagnostics,
            "history_process_calls": history_stream["process_calls"],
            "history_retrieve_calls": history_stream["retrieve_calls"],
        }
    finally:
        try:
            if fresh_state != ffi.NULL:
                api.rubberband_delete(fresh_state)
        finally:
            api.rubberband_delete(reset_state)


def within_ratio(value, expected, tolerance):
    return value is not None and abs(value / expected - 1.0) <= tolerance


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def evaluate(cases):
    by_name = {case["case"]: case for case in cases}
    pitch_ratio = ratio(by_name["pitch_only"]["f0_hz"], by_name["pitch_reference"]["f0_hz"])
    formant_f0_ratio = ratio(
        by_name["formant_only"]["f0_hz"], by_name["formant_reference"]["f0_hz"]
    )
    combined_f0_ratio = ratio(
        by_name["combined"]["f0_hz"], by_name["formant_reference"]["f0_hz"]
    )

    def peak_ratios(changed_case):
        changed_peaks = changed_case["formant_peaks_hz"]
        baseline_peaks = by_name["formant_reference"]["formant_peaks_hz"]
        if changed_peaks is None or baseline_peaks is None:
            return [None, None, None]
        return [
            ratio(changed, baseline)
            for changed, baseline in zip(changed_peaks, baseline_peaks)
        ]

    formant_ratios = peak_ratios(by_name["formant_only"])
    combined_ratios = peak_ratios(by_name["combined"])
    identified_formants = [value for value in formant_ratios if value is not None]

    def status(conditions, enough=True):
        if not enough:
            return "indeterminate"
        return "pass" if all(conditions) else "fail"

    pitch_status = (
        "indeterminate" if pitch_ratio is None
        else "pass" if within_ratio(pitch_ratio, 1.25, 0.01)
        else "fail"
    )
    formant_f0_status = (
        "indeterminate" if formant_f0_ratio is None
        else "pass" if abs(formant_f0_ratio - 1.0) <= 0.01
        else "fail"
    )
    combined_pitch_status = (
        "indeterminate" if combined_f0_ratio is None
        else "pass" if within_ratio(combined_f0_ratio, 1.25, 0.01)
        else "fail"
    )
    formant_status = status(
        [within_ratio(value, 1.25, 0.10) for value in identified_formants],
        len(identified_formants) >= 2,
    )
    combined_formant_status = "observational"
    component_statuses = (
        pitch_status,
        formant_f0_status,
        combined_pitch_status,
        formant_status,
    )
    overall = (
        "fail" if "fail" in component_statuses
        else "indeterminate" if "indeterminate" in component_statuses
        else "pass"
    )
    return {
        "pitch_ratio": pitch_ratio,
        "pitch_status": pitch_status,
        "formant_only_f0_ratio": formant_f0_ratio,
        "formant_only_f0_status": formant_f0_status,
        "combined_f0_ratio": combined_f0_ratio,
        "combined_pitch_status": combined_pitch_status,
        "formant_peak_ratios": formant_ratios,
        "formant_status": formant_status,
        "combined_formant_peak_ratios": combined_ratios,
        "combined_formant_status": combined_formant_status,
        "overall_candidate_status": overall,
    }


def source_sha256():
    source = Path(__file__)
    if not source.is_file():
        return None
    return hashlib.sha256(source.read_bytes()).hexdigest()


def run_measurement(
    library_path,
    include_reset=False,
    reset_before_history_drain=False,
    block_size=BLOCK_SIZE,
):
    block_size = _validate_block_size(block_size)
    if reset_before_history_drain and not include_reset:
        raise ValueError("--reset-before-history-drain requires --reset-comparison")
    host_load_before = _host_load_snapshot()
    ffi, api = load_api(library_path)
    pitch_probe = make_pitch_probe()
    formant_probe = make_formant_probe()
    cases = [
        run_case(
            ffi, api, "pitch_reference", pitch_probe, 1.0, 1.0, 220.0, None,
            block_size=block_size,
        ),
        run_case(
            ffi, api, "pitch_only", pitch_probe, 1.25, 1.0, 275.0, None,
            block_size=block_size,
        ),
        run_case(
            ffi, api, "formant_reference", formant_probe, 1.0, 1.0, 120.0,
            BASE_FORMANT_BANDS, block_size=block_size,
        ),
        run_case(
            ffi, api, "formant_only", formant_probe, 1.0, 1.25, 120.0,
            SCALED_FORMANT_BANDS, block_size=block_size,
        ),
        run_case(
            ffi, api, "combined", formant_probe, 1.25, 1.25, 150.0,
            SCALED_FORMANT_BANDS, block_size=block_size,
        ),
    ]
    reset_comparison = {
        "status": "not_measured",
        "reason": "The optional reset comparison was not enabled; reset was not called.",
    }
    if include_reset:
        reset_comparison = run_reset_comparison(
            ffi,
            api,
            formant_probe,
            pitch_probe,
            pitch_scale=1.0,
            formant_scale=1.0,
            drain_history_output=not reset_before_history_drain,
            block_size=block_size,
        )
    host_load_after = _host_load_snapshot()
    online_cpu_count = host_load_before["online_logical_cpu_count"]
    for case in cases:
        case["process_cpu_percent_normalized"] = (
            100.0 * case["process_cpu_ns"] / case["process_wall_ns"] / online_cpu_count
        )
    return {
        "protocol": "M0-02b-DSP-MEASURE-v2",
        "python_version": platform.python_version(),
        "library_path": library_path,
        "source_sha256": source_sha256(),
        "host_load": {
            "before": host_load_before,
            "after": host_load_after,
        },
        "runtime_audio_metrics": {
            "application_input_queue": "not_measured",
            "callback_underrun": "not_measured",
            "hardware_xrun": "not_measured",
        },
        "settings": {
            "sample_rate_hz": RATE,
            "channels": 1,
            "block_size_frames": block_size,
            "input_frames_per_case": INPUT_FRAMES,
            "rubberband_options": "real-time, threading never, R3/Finer, FormantPreserved",
            "formant_preserved_option": hex(FORMANT_PRESERVED),
            "analysis_window_frames": ANALYSIS_FRAMES,
            "fft_size": FFT_SIZE,
            "transient_frames_each_side": TRANSIENT_FRAMES,
            "reset_comparison_enabled": include_reset,
            "reset_before_history_drain": reset_before_history_drain,
        },
        "cases": cases,
        "reset_comparison": reset_comparison,
        "summary": evaluate(cases),
    }


def parser():
    result = argparse.ArgumentParser(
        description="Measure Rubber Band pitch/formant behavior on deterministic in-memory signals."
    )
    result.add_argument(
        "--run",
        action="store_true",
        required=True,
        help="required opt-in; loads and calls the DSP library but never opens audio devices",
    )
    result.add_argument("--library", default=DEFAULT_LIBRARY, help="Rubber Band shared-library path")
    result.add_argument(
        "--block-size",
        type=int,
        choices=BLOCK_SIZE_CHOICES,
        default=BLOCK_SIZE,
        help="input processing block size in frames (default: 1024)",
    )
    result.add_argument(
        "--reset-comparison",
        action="store_true",
        help="opt in to a history/reset/fresh-instance diagnostic comparison",
    )
    result.add_argument(
        "--reset-before-history-drain",
        action="store_true",
        help="leave history output queued until reset; requires --reset-comparison",
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = run_measurement(
            args.library,
            include_reset=args.reset_comparison,
            reset_before_history_drain=args.reset_before_history_drain,
            block_size=args.block_size,
        )
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
