import contextlib
import importlib.util
import io
import math
from array import array
from pathlib import Path
from unittest.mock import patch
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/m0_02b_dsp_measure.py"
SPEC = importlib.util.spec_from_file_location("m0_02b_dsp_measure", SCRIPT)
measure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measure)


class FakeFFI:
    NULL = None

    @staticmethod
    def new(ctype, count=None):
        if ctype == "float[]":
            return array("f", [0.0]) * count
        if ctype in ("float *[1]", "const float *[1]"):
            return [None]
        raise AssertionError(f"Unexpected CFFI allocation: {ctype}")

    @staticmethod
    def from_buffer(_ctype, samples):
        return memoryview(samples)

    @staticmethod
    def buffer(samples, size):
        return memoryview(samples).cast("B")[:size]

    @staticmethod
    def sizeof(ctype):
        if ctype != "float":
            raise AssertionError(f"Unexpected CFFI type size: {ctype}")
        return 4


class FakeRubberBandState:
    def __init__(self, state_id):
        self.state_id = state_id
        self.pending = []
        self.bias = 0.0
        self.pitch_scale = None
        self.formant_scale = None
        self.max_process_size = None
        self.reset_calls = 0
        self.deleted = False


class FakeRubberBandApi:
    def __init__(
        self,
        fail_new_call=None,
        fail_process_call=None,
        fail_retrieve_call=None,
        fail_reset=False,
        retrieve_chunk_size=None,
        zero_retrieve_call=None,
    ):
        self.states = []
        self.events = []
        self.fail_new_call = fail_new_call
        self.fail_process_call = fail_process_call
        self.fail_retrieve_call = fail_retrieve_call
        self.fail_reset = fail_reset
        self.retrieve_chunk_size = retrieve_chunk_size
        self.zero_retrieve_call = zero_retrieve_call
        self.process_call_count = 0
        self.retrieve_call_count = 0

    def rubberband_new(self, *_args):
        state_id = len(self.states)
        self.events.append(("new", state_id))
        if state_id + 1 == self.fail_new_call:
            return None
        state = FakeRubberBandState(state_id)
        self.states.append(state)
        return state

    def rubberband_delete(self, state):
        self.events.append(("delete", state.state_id))
        state.deleted = True

    def rubberband_reset(self, state):
        self.events.append(("reset", state.state_id))
        if self.fail_reset:
            raise RuntimeError("fake reset failure")
        state.pending.clear()
        state.bias = 0.0
        state.reset_calls += 1

    @staticmethod
    def rubberband_set_max_process_size(state, size):
        state.max_process_size = size

    @staticmethod
    def rubberband_set_pitch_scale(state, scale):
        state.pitch_scale = scale

    @staticmethod
    def rubberband_set_formant_scale(state, scale):
        state.formant_scale = scale

    @staticmethod
    def rubberband_set_formant_option(_state, _option):
        return None

    @staticmethod
    def rubberband_get_engine_version(_state):
        return 3

    @staticmethod
    def rubberband_get_preferred_start_pad(_state):
        return 2

    @staticmethod
    def rubberband_get_start_delay(_state):
        return 2

    @staticmethod
    def rubberband_get_latency(_state):
        return 2

    def rubberband_process(self, state, input_vector, frame_count, final):
        self.process_call_count += 1
        samples = list(input_vector[0][:frame_count])
        self.events.append(("process", state.state_id, frame_count, final))
        if self.process_call_count == self.fail_process_call:
            raise RuntimeError("fake process failure")
        state.pending.extend(sample + state.bias for sample in samples)
        if any(samples):
            state.bias = math.fsum(samples) / len(samples)

    @staticmethod
    def rubberband_available(state):
        return len(state.pending)

    def rubberband_retrieve(self, state, output_vector, frame_count):
        self.retrieve_call_count += 1
        if self.retrieve_call_count == self.fail_retrieve_call:
            raise RuntimeError("fake retrieve failure")
        if self.retrieve_call_count == self.zero_retrieve_call:
            self.events.append(("retrieve", state.state_id, 0))
            return 0
        received = min(frame_count, len(state.pending))
        if self.retrieve_chunk_size is not None:
            received = min(received, self.retrieve_chunk_size)
        for index in range(received):
            output_vector[0][index] = state.pending.pop(0)
        self.events.append(("retrieve", state.state_id, received))
        return received


class IncrementingClock:
    def __init__(self, step_ns):
        self.current_ns = 0
        self.step_ns = step_ns

    def __call__(self):
        current_ns = self.current_ns
        self.current_ns += self.step_ns
        return current_ns


class ProbeSignalTests(unittest.TestCase):
    def test_pitch_probe_is_fixed_length_and_starts_at_peak(self):
        samples = measure.make_pitch_probe()
        self.assertEqual(len(samples), measure.INPUT_FRAMES)
        self.assertAlmostEqual(samples[0], 0.1, places=7)
        self.assertAlmostEqual(max(abs(value) for value in samples), 0.1, places=6)

    def test_formant_probe_is_fixed_length_finite_and_normalized(self):
        samples = measure.make_formant_probe()
        self.assertEqual(len(samples), measure.INPUT_FRAMES)
        self.assertTrue(all(math.isfinite(value) for value in samples))
        self.assertAlmostEqual(max(abs(value) for value in samples), 0.1, places=6)


class SpectrumAnalysisTests(unittest.TestCase):
    def test_parabolic_peak_interpolation_recovers_fractional_bin(self):
        frequencies = [99.0, 100.0, 101.0]
        center = 100.3
        powers = [math.exp(-((frequency - center) ** 2)) for frequency in frequencies]
        self.assertAlmostEqual(
            measure.parabolic_peak_frequency(frequencies, powers, 1),
            center,
            places=10,
        )

    def test_local_envelope_peak_is_limited_to_search_band(self):
        frequencies = list(range(500, 1001))
        values = [1.0] * len(frequencies)
        values[700 - 500] = 4.0
        values[875 - 500] = 3.0
        peak = measure.find_local_envelope_peak(frequencies, values, 800, 950)
        self.assertAlmostEqual(peak, 875.0, places=6)

    def test_local_envelope_peak_reports_unidentifiable_band(self):
        frequencies = list(range(500, 601))
        values = [1.0] * len(frequencies)
        self.assertIsNone(measure.find_local_envelope_peak(frequencies, values, 520, 580))

    def test_octave_smoothing_preserves_frequency_grid_and_finite_values(self):
        frequencies = list(range(400, 4_001))
        power = [0.0] * len(frequencies)
        power[700 - 400] = 1.0
        smoothed_frequencies, smoothed = measure.smooth_power_by_octave(frequencies, power)
        self.assertEqual(smoothed_frequencies[0], 400)
        self.assertEqual(smoothed_frequencies[-1], 4_000)
        self.assertEqual(len(smoothed_frequencies), len(smoothed))
        self.assertTrue(all(math.isfinite(value) for value in smoothed))
        self.assertGreater(max(smoothed), 0.0)


class StreamDiagnosticTests(unittest.TestCase):
    def test_output_block_boundary_diagnostics_compares_boundary_and_interior(self):
        result = measure.output_block_boundary_diagnostics(
            [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0],
            block_size=3,
        )
        self.assertEqual(result["status"], "measured")
        self.assertTrue(result["diagnostic_only"])
        self.assertIn("not API call boundaries", result["boundary_definition"])
        self.assertEqual(result["boundary_frame_indices"], [3, 6])
        self.assertEqual(result["boundary_adjacent_delta"]["count"], 2)
        self.assertEqual(result["boundary_adjacent_delta"]["mean"], 4.5)
        self.assertEqual(result["boundary_adjacent_delta"]["p95_nearest_rank"], 6.0)
        self.assertEqual(result["interior_adjacent_delta"]["count"], 4)
        self.assertEqual(result["interior_adjacent_delta"]["max"], 5.0)
        self.assertEqual(result["boundary_slope_jump"]["mean"], 1.0)
        self.assertEqual(result["interior_slope_jump"]["max"], 1.0)
        self.assertNotIn(result["status"], ("pass", "fail"))

    def test_output_block_boundary_diagnostics_marks_missing_boundaries(self):
        result = measure.output_block_boundary_diagnostics([0.0, 1.0, 2.0], 4)
        self.assertEqual(result["status"], "insufficient_boundaries")
        self.assertEqual(result["boundary_adjacent_delta"]["count"], 0)
        self.assertIsNone(result["boundary_adjacent_delta"]["p95_nearest_rank"])

        result = measure.output_block_boundary_diagnostics([], 4)
        self.assertEqual(result["status"], "insufficient_samples")
        self.assertEqual(result["boundary_frame_indices"], [])

    def test_output_block_boundary_diagnostics_rejects_invalid_input(self):
        for block_size in (0, -1, 1.5, True):
            with self.subTest(block_size=block_size):
                with self.assertRaises(ValueError):
                    measure.output_block_boundary_diagnostics([0.0, 1.0], block_size)
        with self.assertRaisesRegex(ValueError, r"samples\[1\] is not finite"):
            measure.output_block_boundary_diagnostics([0.0, math.nan], 1)

    def test_retrieve_segment_diagnostics_uses_recorded_segment_seams(self):
        samples = [0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5]
        segments = [
            {
                "available_frames": 2,
                "requested_frames": 2,
                "received_frames": 2,
                "output_start_frame": 0,
                "output_end_frame_exclusive": 2,
            },
            {
                "available_frames": 4,
                "requested_frames": 4,
                "received_frames": 4,
                "output_start_frame": 2,
                "output_end_frame_exclusive": 6,
            },
            {
                "available_frames": 4,
                "requested_frames": 4,
                "received_frames": 4,
                "output_start_frame": 6,
                "output_end_frame_exclusive": 10,
            },
        ]
        result = measure.retrieve_segment_boundary_diagnostics(samples, segments)
        self.assertEqual(result["status"], "measured")
        self.assertTrue(result["diagnostic_only"])
        self.assertIn("not rubberband_process", result["boundary_definition"])
        self.assertEqual(result["boundary_frame_indices"], [2, 6])
        self.assertEqual(result["interior_frame_indices"], [4, 8])
        self.assertEqual(result["boundary_adjacent_delta"]["count"], 2)
        self.assertEqual(result["boundary_adjacent_delta"]["values"], [0.25, 0.25])
        self.assertEqual(result["boundary_adjacent_delta"]["max"], 0.25)
        self.assertEqual(result["interior_adjacent_delta"]["max"], 0.0)

    def test_retrieve_segment_diagnostics_includes_first_segment_interior(self):
        segments = [
            {
                "available_frames": 4,
                "requested_frames": 4,
                "received_frames": 4,
                "output_start_frame": 0,
                "output_end_frame_exclusive": 4,
            },
            {
                "available_frames": 4,
                "requested_frames": 4,
                "received_frames": 4,
                "output_start_frame": 4,
                "output_end_frame_exclusive": 8,
            },
            {
                "available_frames": 4,
                "requested_frames": 4,
                "received_frames": 4,
                "output_start_frame": 8,
                "output_end_frame_exclusive": 12,
            },
        ]

        result = measure.retrieve_segment_boundary_diagnostics([0.0] * 12, segments)

        self.assertEqual(result["interior_frame_indices"], [2, 6, 10])

    def test_retrieve_segment_diagnostics_rejects_gaps_and_mismatched_totals(self):
        first = {
            "available_frames": 2,
            "requested_frames": 2,
            "received_frames": 2,
            "output_start_frame": 0,
            "output_end_frame_exclusive": 2,
        }
        gap = {
            "available_frames": 3,
            "requested_frames": 3,
            "received_frames": 3,
            "output_start_frame": 3,
            "output_end_frame_exclusive": 6,
        }
        with self.assertRaisesRegex(ValueError, "gap, overlap"):
            measure.retrieve_segment_boundary_diagnostics([0.0] * 6, [first, gap])
        with self.assertRaisesRegex(ValueError, "totals do not match"):
            measure.retrieve_segment_boundary_diagnostics([0.0] * 3, [first])

    def test_reset_fresh_diagnostics_reports_aligned_overlap_without_gate(self):
        result = measure.reset_fresh_diagnostics(
            reset_samples=[1.0, 2.0],
            fresh_samples=[1.0, 1.0, 1.0],
        )
        self.assertEqual(result["status"], "measured")
        self.assertTrue(result["diagnostic_only"])
        self.assertEqual(result["compared_frame_count"], 2)
        self.assertEqual(result["uncompared_reset_frame_count"], 0)
        self.assertEqual(result["uncompared_fresh_instance_frame_count"], 1)
        self.assertEqual(result["mean_abs_difference"], 0.5)
        self.assertAlmostEqual(result["rms_difference"], math.sqrt(0.5))
        self.assertEqual(result["max_abs_difference"], 1.0)
        self.assertAlmostEqual(result["normalized_rms_difference"], math.sqrt(0.5))
        self.assertNotIn(result["status"], ("pass", "fail"))

    def test_reset_fresh_diagnostics_reports_missing_or_zero_reference(self):
        result = measure.reset_fresh_diagnostics([], [])
        self.assertEqual(result["status"], "insufficient_samples")
        self.assertIsNone(result["rms_difference"])
        self.assertIsNone(result["normalized_rms_difference"])

        result = measure.reset_fresh_diagnostics([0.0], [0.0])
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["normalization_status"], "zero_reference")
        self.assertIsNone(result["normalized_rms_difference"])

    def test_reset_fresh_diagnostics_rejects_non_finite_samples(self):
        with self.assertRaisesRegex(ValueError, r"reset_samples\[0\] is not finite"):
            measure.reset_fresh_diagnostics([math.inf], [0.0])


class ResetComparisonPathTests(unittest.TestCase):
    def test_normal_case_uses_shared_stream_helper_without_reset(self):
        ffi = FakeFFI()
        api = FakeRubberBandApi()
        signal = array("f", [0.25] * 8)
        result = measure.run_case(
            ffi, api, "fake_case", signal, 1.0, 1.0, 220.0, None, block_size=256
        )
        self.assertEqual(result["input_frames"], 8)
        self.assertEqual(result["block_size_frames"], 256)
        self.assertEqual(result["output_frames"], 10)
        self.assertEqual(result["process_calls"], 2)
        self.assertEqual(result["start_pad_frames"], 2)
        self.assertEqual(api.states[0].max_process_size, 256)
        self.assertEqual(api.states[0].reset_calls, 0)
        self.assertTrue(api.states[0].deleted)

    def test_run_case_records_chunk_ranges_and_process_drain_provenance(self):
        ffi = FakeFFI()
        api = FakeRubberBandApi(retrieve_chunk_size=4)
        result = measure.run_case(
            ffi,
            api,
            "fake_case",
            array("f", [0.25] * 8),
            1.0,
            1.0,
            220.0,
            None,
            block_size=256,
        )

        self.assertEqual(result["retrieve_calls"], 3)
        self.assertEqual(result["retrieve_segment_boundary_diagnostics"]["boundary_frame_indices"], [2, 6])
        self.assertEqual(
            [
                (
                    segment["case"],
                    segment["block_size_frames"],
                    segment["retrieve_call_index"],
                    segment["drain_invocation_index"],
                    segment["drain_phase"],
                    segment["process_call_index"],
                    segment["process_final"],
                    segment["available_frames"],
                    segment["requested_frames"],
                    segment["received_frames"],
                    segment["output_start_frame"],
                    segment["output_end_frame_exclusive"],
                )
                for segment in result["retrieve_segments"]
            ],
            [
                ("fake_case", 256, 1, 1, "after_process", 1, False, 2, 2, 2, 0, 2),
                ("fake_case", 256, 2, 2, "after_process", 2, True, 8, 8, 4, 2, 6),
                ("fake_case", 256, 3, 2, "after_process", 2, True, 4, 4, 4, 6, 10),
            ],
        )

    def test_drain_state_records_post_final_drain_segments(self):
        ffi = FakeFFI()
        api = FakeRubberBandApi()
        state = FakeRubberBandState(0)
        state.pending.extend([0.1, 0.2, 0.3])
        audio = array("f")
        segments = []

        calls = measure._drain_state(
            ffi,
            api,
            state,
            audio,
            "final drain test",
            256,
            segments,
            retrieve_call_offset=4,
            process_call_index=3,
            process_final=True,
            drain_invocation_index=4,
            drain_phase="post_final_drain",
        )

        self.assertEqual(calls, 1)
        self.assertEqual(len(audio), 3)
        for actual, expected in zip(audio, (0.1, 0.2, 0.3)):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(
            {
                key: value
                for key, value in segments[0].items()
                if key != "retrieve_api_ns"
            },
            {
                "case": "final drain test",
                "block_size_frames": 256,
                "retrieve_call_index": 5,
                "drain_invocation_index": 4,
                "drain_phase": "post_final_drain",
                "process_call_index": 3,
                "process_final": True,
                "available_frames": 3,
                "requested_frames": 3,
                "received_frames": 3,
                "output_start_frame": 0,
                "output_end_frame_exclusive": 3,
            },
        )
        self.assertGreaterEqual(segments[0]["retrieve_api_ns"], 0)

    def test_run_case_fails_if_available_output_cannot_be_retrieved(self):
        api = FakeRubberBandApi(zero_retrieve_call=1)
        with self.assertRaisesRegex(RuntimeError, "available frames but retrieve returned 0"):
            measure.run_case(
                FakeFFI(),
                api,
                "fake_case",
                array("f", [0.25] * 8),
                1.0,
                1.0,
                220.0,
                None,
                block_size=256,
            )
        self.assertTrue(api.states[0].deleted)

    def test_reset_path_resets_history_then_compares_against_fresh_state(self):
        ffi = FakeFFI()
        api = FakeRubberBandApi()
        history = array("f", [0.5] * 8)
        comparison = array("f", [0.25] * 8)

        with (
            patch.object(measure, "TRANSIENT_FRAMES", 1),
            patch.object(measure, "ANALYSIS_FRAMES", 4),
        ):
            result = measure.run_reset_comparison(
                ffi, api, history, comparison, block_size=256
            )

        self.assertEqual(result["status"], "measured")
        self.assertTrue(result["diagnostic_only"])
        self.assertEqual(result["comparison"]["compared_frame_count"], 4)
        self.assertEqual(result["comparison"]["rms_difference"], 0.0)
        self.assertEqual(result["history_output_policy"], "drained_before_reset")
        self.assertEqual(result["history_available_before_reset"], 0)
        self.assertEqual(result["history_available_after_reset"], 0)
        self.assertNotIn(result["status"], ("pass", "fail"))
        reset_state, fresh_state = api.states
        self.assertEqual(reset_state.reset_calls, 1)
        self.assertEqual(fresh_state.reset_calls, 0)
        self.assertEqual(reset_state.max_process_size, 256)
        self.assertEqual(fresh_state.max_process_size, 256)
        self.assertTrue(reset_state.deleted)
        self.assertTrue(fresh_state.deleted)

        reset_event = api.events.index(("reset", reset_state.state_id))
        history_process_events = [
            index for index, event in enumerate(api.events)
            if event[0] == "process" and event[1] == reset_state.state_id
        ]
        self.assertLess(history_process_events[0], reset_event)
        self.assertGreater(history_process_events[-1], reset_event)
        fresh_new_event = api.events.index(("new", fresh_state.state_id))
        fresh_process_event = next(
            index for index, event in enumerate(api.events)
            if event[0] == "process" and event[1] == fresh_state.state_id
        )
        self.assertLess(reset_event, fresh_new_event)
        self.assertLess(fresh_new_event, fresh_process_event)
        self.assertTrue(
            all(
                api.events[index][0] == "process"
                for index in history_process_events
            )
        )

    def test_reset_path_reports_output_pending_before_reset_and_cleared_after(self):
        ffi = FakeFFI()
        api = FakeRubberBandApi()
        with (
            patch.object(measure, "TRANSIENT_FRAMES", 1),
            patch.object(measure, "ANALYSIS_FRAMES", 4),
        ):
            result = measure.run_reset_comparison(
                ffi,
                api,
                array("f", [0.5] * 8),
                array("f", [0.25] * 8),
                drain_history_output=False,
                block_size=256,
            )

        self.assertEqual(result["history_output_policy"], "left_pending_for_reset")
        self.assertEqual(result["history_retrieved_frames"], 0)
        self.assertEqual(result["history_available_before_reset"], 10)
        self.assertEqual(result["history_available_after_reset"], 0)
        reset_state = api.states[0]
        reset_event = api.events.index(("reset", reset_state.state_id))
        history_retrieve_events = [
            index for index, event in enumerate(api.events)
            if (
                index < reset_event
                and event[0] == "retrieve"
                and event[1] == reset_state.state_id
            )
        ]
        self.assertEqual(history_retrieve_events, [])
        self.assertTrue(
            any(
                index > reset_event
                and event[0] == "process"
                and event[1] == reset_state.state_id
                for index, event in enumerate(api.events)
            )
        )

    def test_reset_path_cleans_up_reset_state_when_fresh_creation_fails(self):
        api = FakeRubberBandApi(fail_new_call=2)
        with self.assertRaisesRegex(RuntimeError, "null fresh state"):
            measure.run_reset_comparison(
                FakeFFI(), api, array("f", [0.5]), array("f", [0.25])
            )
        self.assertEqual(len(api.states), 1)
        self.assertTrue(api.states[0].deleted)

    def test_reset_failure_cleans_up_created_state(self):
        api = FakeRubberBandApi(fail_reset=True)
        with self.assertRaisesRegex(RuntimeError, "fake reset failure"):
            measure.run_reset_comparison(
                FakeFFI(), api, array("f", [0.5]), array("f", [0.25])
            )
        self.assertEqual(len(api.states), 1)
        self.assertTrue(api.states[0].deleted)


class TimingInstrumentationTests(unittest.TestCase):
    def test_run_case_reports_api_drain_and_block_timings(self):
        wall_clock = IncrementingClock(1_000_000)
        cpu_clock = IncrementingClock(100)
        api = FakeRubberBandApi(retrieve_chunk_size=128)
        with (
            patch.object(measure.time, "perf_counter_ns", side_effect=wall_clock),
            patch.object(measure.time, "process_time_ns", side_effect=cpu_clock),
        ):
            result = measure.run_case(
                FakeFFI(),
                api,
                "timed_case",
                array("f", [0.25] * 257),
                1.0,
                1.0,
                220.0,
                None,
                block_size=256,
            )

        blocks = result["process_blocks"]
        drains = result["drain_timings"]
        self.assertTrue(all(block["case"] == "timed_case" for block in blocks))
        self.assertTrue(all(block["block_size_frames"] == 256 for block in blocks))
        self.assertTrue(all(drain["case"] == "timed_case" for drain in drains))
        self.assertTrue(all(drain["block_size_frames"] == 256 for drain in drains))
        self.assertEqual([block["input_frames"] for block in blocks], [2, 256, 1])
        self.assertEqual([block["process_api_ns"] for block in blocks], [1_000_000] * 3)
        self.assertEqual(len(drains), 4)
        self.assertEqual(drains[-1]["drain_phase"], "post_final_drain")
        self.assertTrue(all(block["over_budget"] for block in blocks))
        self.assertEqual(result["process_block_over_budget_count"], 3)
        self.assertEqual(
            [block["budget_ns"] for block in blocks],
            [frames * 1_000_000_000 / measure.RATE for frames in (2, 256, 1)],
        )
        self.assertEqual(result["process_cpu_ns"], 100)
        self.assertEqual(
            result["processing_seconds"],
            result["process_wall_ns"] / 1_000_000_000,
        )
        self.assertEqual(
            result["timing_summary"]["process_api_ns"]["count"],
            len(blocks),
        )
        self.assertEqual(
            result["timing_summary"]["retrieve_api_ns"]["count"],
            api.retrieve_call_count,
        )
        self.assertEqual(
            result["timing_summary"]["available_api_ns"]["count"],
            sum(len(drain["available_api_ns"]) for drain in drains),
        )
        self.assertEqual(result["total_retrieved_frames"], result["output_frames"])
        self.assertEqual(result["max_available_frames"], 256)
        self.assertTrue(
            all(
                len(drain["available_api_ns"]) == len(drain["available_frames"])
                for drain in drains
            )
        )

    def test_measurement_reports_cpu_normalization_and_host_load_snapshots(self):
        snapshots = [
            {
                "online_logical_cpu_count": 4,
                "load_average_1m": 0.1,
                "load_average_5m": 0.2,
                "load_average_15m": 0.3,
            },
            {
                "online_logical_cpu_count": 4,
                "load_average_1m": 0.4,
                "load_average_5m": 0.5,
                "load_average_15m": 0.6,
            },
        ]
        with (
            patch.object(measure, "load_api", return_value=(FakeFFI(), FakeRubberBandApi())),
            patch.object(
                measure,
                "make_pitch_probe",
                return_value=array("f", [0.25] * 8),
            ),
            patch.object(
                measure,
                "make_formant_probe",
                return_value=array("f", [0.125] * 8),
            ),
            patch.object(measure, "_host_load_snapshot", side_effect=snapshots),
        ):
            result = measure.run_measurement("/unused/library", block_size=256)

        self.assertEqual(result["host_load"], {"before": snapshots[0], "after": snapshots[1]})
        self.assertEqual(
            result["runtime_audio_metrics"],
            {
                "application_input_queue": "not_measured",
                "callback_underrun": "not_measured",
                "hardware_xrun": "not_measured",
            },
        )
        for case in result["cases"]:
            self.assertGreater(case["process_wall_ns"], 0)
            self.assertAlmostEqual(
                case["process_cpu_percent_normalized"],
                100.0 * case["process_cpu_ns"] / case["process_wall_ns"] / 4,
            )

    def test_host_load_snapshot_reports_raw_load_averages(self):
        with (
            patch.object(measure.os, "sysconf", return_value=8),
            patch.object(measure.os, "getloadavg", return_value=(1.1, 2.2, 3.3)),
        ):
            snapshot = measure._host_load_snapshot()

        self.assertEqual(
            snapshot,
            {
                "online_logical_cpu_count": 8,
                "load_average_1m": 1.1,
                "load_average_5m": 2.2,
                "load_average_15m": 3.3,
            },
        )


class ParameterValidationTests(unittest.TestCase):
    @staticmethod
    def _run_case(api, pitch_scale, formant_scale):
        return measure.run_case(
            FakeFFI(),
            api,
            "parameter_validation",
            array("f", [0.25] * 8),
            pitch_scale,
            formant_scale,
            220.0,
            None,
        )

    def test_invalid_parameters_are_rejected_before_any_c_api_call(self):
        invalid = (
            (0.499, 1.0),
            (2.001, 1.0),
            (math.nan, 1.0),
            (math.inf, 1.0),
            (-math.inf, 1.0),
            (True, 1.0),
            (None, 1.0),
            ("1.0", 1.0),
            (1.0, -0.5),
            (1.0, math.nan),
            (1.0, math.inf),
            (1.0, -math.inf),
            (1.0, True),
            (1.0, None),
        )
        for pitch_scale, formant_scale in invalid:
            with self.subTest(pitch_scale=pitch_scale, formant_scale=formant_scale):
                api = FakeRubberBandApi()
                with self.assertRaises(ValueError):
                    self._run_case(api, pitch_scale, formant_scale)
                self.assertEqual(api.events, [])

    def test_pitch_boundaries_and_supported_formant_scales_are_accepted(self):
        for pitch_scale, formant_scale in ((0.5, 0.0), (2.0, 5.0)):
            with self.subTest(pitch_scale=pitch_scale, formant_scale=formant_scale):
                api = FakeRubberBandApi()
                self._run_case(api, pitch_scale, formant_scale)
                self.assertEqual(api.states[0].pitch_scale, pitch_scale)
                self.assertEqual(api.states[0].formant_scale, formant_scale)
                self.assertTrue(api.states[0].deleted)

    def test_run_reset_comparison_validates_before_creating_state(self):
        for pitch_scale, formant_scale in ((0.49, 1.0), (1.0, -0.01), (1.0, math.nan)):
            with self.subTest(pitch_scale=pitch_scale, formant_scale=formant_scale):
                api = FakeRubberBandApi()
                with self.assertRaises(ValueError):
                    measure.run_reset_comparison(
                        FakeFFI(),
                        api,
                        array("f", [0.5]),
                        array("f", [0.25]),
                        pitch_scale,
                        formant_scale,
                    )
                self.assertEqual(api.events, [])

    def test_run_case_cleans_up_state_after_process_or_retrieve_error(self):
        for api in (
            FakeRubberBandApi(fail_process_call=1),
            FakeRubberBandApi(fail_retrieve_call=1),
        ):
            with self.subTest(api=api):
                with self.assertRaisesRegex(RuntimeError, "fake (process|retrieve) failure"):
                    self._run_case(api, 1.0, 1.0)
                self.assertEqual(len(api.states), 1)
                self.assertTrue(api.states[0].deleted)

    def test_instance_creation_failure_does_not_create_or_delete_a_state(self):
        api = FakeRubberBandApi(fail_new_call=1)
        with self.assertRaisesRegex(RuntimeError, "null state"):
            self._run_case(api, 1.0, 1.0)
        self.assertEqual(api.states, [])
        self.assertEqual(api.events, [("new", 0)])


class CandidateEvaluationTests(unittest.TestCase):
    @staticmethod
    def _cases(formant_peaks, combined_peaks):
        return [
            {"case": "pitch_reference", "f0_hz": 220.0},
            {"case": "pitch_only", "f0_hz": 275.0},
            {
                "case": "formant_reference",
                "f0_hz": 120.0,
                "formant_peaks_hz": [700.0, 1_500.0, 2_500.0],
            },
            {
                "case": "formant_only",
                "f0_hz": 120.0,
                "formant_peaks_hz": formant_peaks,
            },
            {
                "case": "combined",
                "f0_hz": 150.0,
                "formant_peaks_hz": combined_peaks,
            },
        ]

    def test_expected_parameter_effects_pass(self):
        result = measure.evaluate(
            self._cases([875.0, 1_875.0, 3_125.0], [875.0, 1_875.0, 3_125.0])
        )
        self.assertEqual(result["pitch_status"], "pass")
        self.assertEqual(result["formant_only_f0_status"], "pass")
        self.assertEqual(result["formant_status"], "pass")
        self.assertEqual(result["combined_formant_status"], "observational")
        self.assertEqual(result["overall_candidate_status"], "pass")

    def test_formant_gate_requires_two_identifiable_matching_peaks(self):
        result = measure.evaluate(
            self._cases([875.0, 1_875.0, None], [875.0, 1_875.0, 3_125.0])
        )
        self.assertEqual(result["formant_status"], "pass")

        result = measure.evaluate(
            self._cases([875.0, None, None], [875.0, 1_875.0, 3_125.0])
        )
        self.assertEqual(result["formant_status"], "indeterminate")

    def test_combined_formant_peaks_are_observational_not_a_gate(self):
        result = measure.evaluate(
            self._cases([875.0, 1_875.0, 3_125.0], [875.0, 2_250.0, 3_125.0])
        )
        self.assertEqual(result["formant_status"], "pass")
        self.assertEqual(result["combined_formant_status"], "observational")
        self.assertEqual(result["combined_formant_peak_ratios"], [1.25, 1.5, 1.25])
        self.assertEqual(result["overall_candidate_status"], "pass")

    def test_recorded_board_measurement_passes_under_current_criteria(self):
        cases = self._cases(
            [795.868369283822, 1_813.7945973747057, 3_174.2229580173653],
            [994.3552012815467, 2_126.863385787912, 3_685.82165476117],
        )
        cases[0]["f0_hz"] = 220.00000515027727
        cases[1]["f0_hz"] = 274.99995051209385
        cases[2]["f0_hz"] = 120.00006813269032
        cases[3]["f0_hz"] = 120.00006820794074
        cases[4]["f0_hz"] = 150.00006785507497
        result = measure.evaluate(cases)
        self.assertAlmostEqual(result["pitch_ratio"], 1.2499997457920389)
        self.assertAlmostEqual(result["formant_only_f0_ratio"], 1.0, places=8)
        self.assertAlmostEqual(result["combined_f0_ratio"], 1.249999855743516)
        self.assertEqual(result["formant_status"], "pass")
        self.assertEqual(result["combined_formant_status"], "observational")
        self.assertEqual(result["overall_candidate_status"], "pass")

    def test_missing_analysis_is_indeterminate_not_success(self):
        result = measure.evaluate(self._cases(None, None))
        self.assertEqual(result["formant_status"], "indeterminate")
        self.assertEqual(result["combined_formant_status"], "observational")
        self.assertEqual(result["overall_candidate_status"], "indeterminate")


class CliSafetyTests(unittest.TestCase):
    def test_running_requires_explicit_opt_in(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            measure.parser().parse_args([])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--run", stderr.getvalue())

    def test_reset_comparison_requires_separate_opt_in(self):
        self.assertFalse(measure.parser().parse_args(["--run"]).reset_comparison)
        self.assertTrue(
            measure.parser().parse_args(["--run", "--reset-comparison"]).reset_comparison
        )
        self.assertTrue(
            measure.parser().parse_args(
                ["--run", "--reset-comparison", "--reset-before-history-drain"]
            ).reset_before_history_drain
        )

    def test_pending_history_output_flag_requires_reset_comparison(self):
        with patch.object(measure, "load_api") as load_api:
            with self.assertRaisesRegex(ValueError, "requires --reset-comparison"):
                measure.run_measurement(
                    "/unused/library",
                    reset_before_history_drain=True,
                )
        load_api.assert_not_called()

    def test_cli_reports_measurement_errors_without_success_json(self):
        stderr = io.StringIO()
        stdout = io.StringIO()
        with (
            patch.object(
                measure,
                "run_measurement",
                side_effect=RuntimeError("retrieve accounting failed"),
            ),
            contextlib.redirect_stderr(stderr),
            contextlib.redirect_stdout(stdout),
        ):
            result = measure.main(["--run"])

        self.assertEqual(result, 2)
        self.assertIn("ERROR: retrieve accounting failed", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")


class BlockSizeConfigurationTests(unittest.TestCase):
    def test_run_case_uses_selected_size_for_api_and_each_input_block(self):
        for block_size in measure.BLOCK_SIZE_CHOICES:
            with self.subTest(block_size=block_size):
                ffi = FakeFFI()
                api = FakeRubberBandApi()
                signal = array("f", [0.25] * (block_size + 1))

                result = measure.run_case(
                    ffi,
                    api,
                    "block_size_case",
                    signal,
                    1.0,
                    1.0,
                    220.0,
                    None,
                    block_size=block_size,
                )

                process_events = [event for event in api.events if event[0] == "process"]
                self.assertEqual(api.states[0].max_process_size, block_size)
                self.assertEqual(
                    [event[2] for event in process_events],
                    [2, block_size, 1],
                )
                self.assertEqual(result["block_size_frames"], block_size)
                self.assertEqual(
                    result["output_block_boundary_diagnostics"]["block_size_frames"],
                    block_size,
                )

    def test_measurement_reports_and_applies_selected_sizes(self):
        pitch_probe = array("f", [0.25] * 8)
        formant_probe = array("f", [0.125] * 8)
        for requested_size in (None,) + measure.BLOCK_SIZE_CHOICES:
            block_size = measure.BLOCK_SIZE if requested_size is None else requested_size
            with self.subTest(block_size=block_size):
                api = FakeRubberBandApi()
                with (
                    patch.object(measure, "load_api", return_value=(FakeFFI(), api)),
                    patch.object(measure, "make_pitch_probe", return_value=pitch_probe),
                    patch.object(measure, "make_formant_probe", return_value=formant_probe),
                ):
                    run_options = (
                        {} if requested_size is None
                        else {"block_size": requested_size}
                    )
                    result = measure.run_measurement("/unused/library", **run_options)

                self.assertEqual(result["settings"]["block_size_frames"], block_size)
                self.assertEqual(
                    [case["block_size_frames"] for case in result["cases"]],
                    [block_size] * 5,
                )
                self.assertEqual(
                    [state.max_process_size for state in api.states],
                    [block_size] * 5,
                )

    def test_measurement_rejects_unsupported_size_before_loading_library(self):
        with patch.object(measure, "load_api") as load_api:
            with self.assertRaisesRegex(ValueError, "block_size must be one of"):
                measure.run_measurement("/unused/library", block_size=768)
        load_api.assert_not_called()

    def test_cli_defaults_to_1024_and_accepts_all_supported_sizes(self):
        self.assertEqual(measure.parser().parse_args(["--run"]).block_size, 1_024)
        for block_size in measure.BLOCK_SIZE_CHOICES:
            with self.subTest(block_size=block_size):
                args = measure.parser().parse_args(
                    ["--run", "--block-size", str(block_size)]
                )
                self.assertEqual(args.block_size, block_size)

    def test_cli_rejects_unsupported_size(self):
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            measure.parser().parse_args(["--run", "--block-size", "768"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("invalid choice", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
