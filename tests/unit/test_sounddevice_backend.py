"""Real NumPy buffers with an in-test device library; never load PortAudio."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from voice_changer.audio.engine import AudioEngine, EngineError, EngineState
from voice_changer.audio import sounddevice_backend
from voice_changer.audio.sounddevice_backend import (
    DeviceRef, SoundDeviceBackend, StreamConfig,
)
from voice_changer.dsp.parameters import LiveControl, PRESETS


ROOT = Path(__file__).resolve().parents[2]


class CallbackAbort(Exception):
    pass


class FakeRawStream:
    def __init__(self, library, options):
        self.library = library
        self.options = options
        self.device = options["device"]
        self.samplerate = options["samplerate"]
        self.blocksize = options["blocksize"]
        self.channels = options["channels"]
        self.dtype = ("float32", "float32")
        self.active = False
        self.closed = False
        self.errors = {}

    def start(self):
        self.library.events.append("start")
        if "start" in self.errors:
            raise self.errors["start"]
        self.active = True
        if self.library.on_start:
            self.library.on_start(self)

    def stop(self):
        self.library.events.append("stop")
        if "stop" in self.errors:
            raise self.errors["stop"]
        self.active = False

    def close(self):
        self.library.events.append("close")
        if "close" in self.errors:
            raise self.errors["close"]
        self.active = False
        self.closed = True


class FakeSoundDevice(ModuleType):
    CallbackAbort = CallbackAbort

    def __init__(self):
        super().__init__("sounddevice")
        self.events = []
        self.streams = []
        self.devices = {
            3: {"name": "selected input", "hostapi": 1, "max_input_channels": 2},
            7: {"name": "selected output", "hostapi": 1, "max_output_channels": 2},
        }
        self.settings_error = None
        self.create_error = None
        self.on_create = None
        self.on_start = None

    def query_devices(self, index, kind):
        self.events.append(("query", index, kind))
        return self.devices[index]

    def check_input_settings(self, **options):
        self.events.append(("input", options))
        if self.settings_error:
            raise self.settings_error

    def check_output_settings(self, **options):
        self.events.append(("output", options))
        if self.settings_error:
            raise self.settings_error

    def Stream(self, **options):
        self.events.append("create")
        if self.create_error:
            raise self.create_error
        stream = FakeRawStream(self, options)
        self.streams.append(stream)
        if self.on_create:
            self.on_create(stream)
        return stream


class FakeCallbackProcessor:
    def __init__(self, *, gain=0.25):
        self.gain = gain
        self.error = None
        self.started = False
        self.closed = False
        self.health_checks = 0

    def start(self):
        self.started = True

    def exchange(self, samples, output):
        output[:] = samples * self.gain
        return True

    def check_health(self):
        self.health_checks += 1

    def close(self):
        self.closed = True


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.library = FakeSoundDevice()
        self.module_patch = patch.dict(sys.modules, {"sounddevice": self.library})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.config = StreamConfig(
            DeviceRef(3, "selected input", 1), DeviceRef(7, "selected output", 1),
            samplerate=48000, blocksize=4, channels=2, output_ceiling=0.1,
        )

    def start_engine(self):
        engine = AudioEngine(SoundDeviceBackend(self.config, output_confirmed=True))
        engine.start()
        self.addCleanup(engine.stop)
        return engine, self.library.streams[-1]

    def callback(self, raw, data, output=None, *, frames=4, status=False):
        if output is None:
            output = np.full((4, 2), 9, dtype=np.float32)
        raw.options["callback"](data, output, frames, None, status)
        return output

    def test_device_ref_rejects_defaults_and_invalid_identity(self):
        for values in (
            (-1, "input", 1), (True, "input", 1), (None, "input", 1),
            (3, "", 1), (3, " ", 1), (3, "input", -1), (3, "input", True),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                DeviceRef(*values)

    def test_config_rejects_invalid_formats_and_ceiling(self):
        invalid = {
            "input_device": (None, "default"),
            "output_device": (None,),
            "samplerate": (0, -1, True, 48000.0),
            "blocksize": (0, -1, False, 4.0),
            "channels": (0, -1, True, 2.0),
            "output_ceiling": (0, -0.1, 1.01, float("nan"), float("inf"), True, "0.1"),
        }
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    replace(self.config, **{key: value})
        self.assertEqual(self.library.events, [])

    def test_confirmation_is_required_without_any_device_calls(self):
        for confirmation in (False, None, 1, "yes"):
            backend = SoundDeviceBackend(self.config, output_confirmed=confirmation)
            with self.assertRaisesRegex(RuntimeError, "confirmation"):
                backend.open_stream()
        self.assertEqual(self.library.events, [])

    def test_open_only_prepares_buffers_without_query_or_native_stream(self):
        stream = SoundDeviceBackend(self.config, output_confirmed=True).open_stream()
        self.assertEqual(self.library.events, [])
        stream.close()
        with self.assertRaisesRegex(RuntimeError, "cannot be started again"):
            stream.start()

    def test_explicit_devices_format_and_lifecycle(self):
        engine, raw = self.start_engine()
        self.assertEqual(raw.options["device"], (3, 7))
        self.assertEqual(raw.options["samplerate"], 48000)
        self.assertEqual(raw.options["blocksize"], 4)
        self.assertEqual(raw.options["channels"], (2, 2))
        self.assertEqual(raw.options["dtype"], "float32")
        self.assertEqual(raw.options["latency"], "low")
        self.assertIn(("input", {
            "device": 3, "channels": 2, "samplerate": 48000, "dtype": "float32",
        }), self.library.events)
        self.assertIn(("output", {
            "device": 7, "channels": 2, "samplerate": 48000, "dtype": "float32",
        }), self.library.events)
        engine.check_health()
        engine.stop()
        self.assertTrue(raw.closed)
        self.assertFalse(engine.has_stream)

    def test_injected_processor_factory_owns_audio_processing_lifecycle(self):
        control = LiveControl(PRESETS["original"])
        created = []

        def factory(config, received_control):
            self.assertEqual(config, self.config)
            self.assertIs(received_control, control)
            processor = FakeCallbackProcessor()
            created.append(processor)
            return processor

        backend = SoundDeviceBackend(
            self.config, output_confirmed=True, control=control,
            processor_factory=factory,
        )
        engine = AudioEngine(backend)
        engine.start()
        self.addCleanup(engine.stop)
        raw = self.library.streams[-1]
        self.assertEqual(raw.options["latency"], "high")
        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].started)
        output = self.callback(raw, np.ones((4, 2), dtype=np.float32) * 0.4)
        np.testing.assert_allclose(output, np.ones((4, 2), dtype=np.float32) * 0.1)
        engine.check_health()
        self.assertGreater(created[0].health_checks, 0)
        engine.stop()
        self.assertTrue(created[0].closed)
        self.assertTrue(raw.closed)

    def test_injected_processor_factory_runs_without_dsp_control(self):
        created = []

        def factory(config, control):
            self.assertEqual(config, self.config)
            self.assertIsNone(control)
            processor = FakeCallbackProcessor()
            created.append(processor)
            return processor

        backend = SoundDeviceBackend(
            self.config, output_confirmed=True, processor_factory=factory,
        )
        engine = AudioEngine(backend)
        engine.start()
        self.addCleanup(engine.stop)
        raw = self.library.streams[-1]
        source = np.full((4, 2), 0.4, dtype=np.float32)
        output = self.callback(raw, source)
        np.testing.assert_allclose(output, source * 0.25)
        engine.stop()
        self.assertTrue(created[0].closed)

    def test_active_stream_without_callbacks_faults_at_two_seconds_and_releases(self):
        with patch.object(sounddevice_backend, "monotonic", return_value=0.0) as clock:
            engine, raw = self.start_engine()
            clock.return_value = 1.999
            engine.check_health()
            clock.return_value = 2.0
            with self.assertRaisesRegex(EngineError, "callback stopped advancing for 2 seconds"):
                engine.check_health()
        self.assertIs(engine.state, EngineState.FAULT)
        self.assertTrue(raw.closed)
        self.assertFalse(engine.has_stream)

    def test_callback_progress_resets_watchdog_without_reading_clock_in_callback(self):
        with patch.object(sounddevice_backend, "monotonic", return_value=0.0) as clock:
            engine, raw = self.start_engine()
            clock.return_value = 1.0
            clock.reset_mock()
            self.callback(raw, np.zeros((4, 2), dtype=np.float32))
            clock.assert_not_called()
            engine.check_health()
            clock.return_value = 2.999
            engine.check_health()
            clock.return_value = 3.0
            with self.assertRaisesRegex(EngineError, "callback stopped advancing"):
                engine.check_health()
        self.assertTrue(raw.closed)
        self.assertFalse(engine.has_stream)

    def test_changed_device_or_missing_channels_refuses_native_creation(self):
        for field, value in (("name", "replacement"), ("hostapi", 2),
                             ("max_input_channels", 1)):
            with self.subTest(field=field):
                old = self.library.devices[3][field]
                self.library.devices[3][field] = value
                engine = AudioEngine(SoundDeviceBackend(self.config, output_confirmed=True))
                with self.assertRaises(EngineError):
                    engine.start()
                self.assertEqual(engine.state, EngineState.FAULT)
                self.assertFalse(engine.has_stream)
                self.assertEqual(self.library.streams, [])
                self.library.devices[3][field] = old

    def test_unsupported_settings_and_constructor_failure_surface(self):
        self.library.settings_error = ValueError("unsupported rate")
        engine = AudioEngine(SoundDeviceBackend(self.config, output_confirmed=True))
        with self.assertRaisesRegex(EngineError, "unsupported rate"):
            engine.start()
        self.assertEqual(self.library.streams, [])
        self.library.settings_error = None
        self.library.create_error = OSError("permission denied")
        engine.stop()
        with self.assertRaisesRegex(EngineError, "permission denied"):
            engine.start()
        self.assertFalse(engine.has_stream)

    def test_native_format_change_closes_without_starting(self):
        for field, value in (
            ("device", (3, 9)), ("samplerate", 44100), ("blocksize", 8),
            ("channels", (1, 2)), ("dtype", ("float64", "float32")),
        ):
            with self.subTest(field=field):
                self.library.on_create = lambda stream: setattr(stream, field, value)
                engine = AudioEngine(SoundDeviceBackend(self.config, output_confirmed=True))
                with self.assertRaisesRegex(EngineError, "changed"):
                    engine.start()
                self.assertTrue(self.library.streams[-1].closed)
                self.assertFalse(engine.has_stream)
        self.assertNotIn("start", self.library.events)

    def test_device_change_during_creation_is_detected_before_start(self):
        def change_device(stream):
            self.library.devices[7]["name"] = "replacement"
        self.library.on_create = change_device
        engine = AudioEngine(SoundDeviceBackend(self.config, output_confirmed=True))
        with self.assertRaisesRegex(EngineError, "device changed"):
            engine.start()
        self.assertTrue(self.library.streams[0].closed)
        self.assertNotIn("start", self.library.events)

    def test_original_samples_and_stereo_are_preserved_below_ceiling(self):
        engine, raw = self.start_engine()
        data = np.array([[0.02, -0.04], [0.03, -0.01]] * 2, dtype=np.float32)
        before = data.copy()
        output = self.callback(raw, data)
        np.testing.assert_array_equal(output, before)
        np.testing.assert_array_equal(data, before)
        self.assertEqual(engine.state, EngineState.RUNNING)

    def test_equal_callback_levels_publish_distinct_capture_snapshots(self):
        control = LiveControl(PRESETS["original"])
        stream = SoundDeviceBackend(
            self.config, output_confirmed=True, control=control,
        ).open_stream()
        self.addCleanup(stream.close)
        samples = np.zeros((4, 2), dtype=np.float32)
        output = np.empty_like(samples)
        stream._callback(samples, output, 4, None, False)
        first = control.levels
        stream._callback(samples, output, 4, None, False)
        self.assertEqual(first, (0.0, 0.0))
        self.assertEqual(control.levels, first)
        self.assertIsNot(control.levels, first)

    def test_peak_never_exceeds_exact_requested_ceiling(self):
        _, raw = self.start_engine()
        data = np.array([[10, -10], [0.1, -0.1]] * 2, dtype=np.float32)
        output = self.callback(raw, data)
        self.assertLessEqual(float(np.abs(output).max()), self.config.output_ceiling)
        self.assertGreater(float(output.max()), 0)
        self.assertLess(float(output.min()), 0)
        np.testing.assert_array_equal(data[:, 0], np.array([10, 0.1] * 2, dtype=np.float32))

    def test_unrepresentably_small_ceiling_is_rejected_without_device_calls(self):
        backend = SoundDeviceBackend(
            replace(self.config, output_ceiling=1e-100), output_confirmed=True,
        )
        with self.assertRaisesRegex(ValueError, "too small"):
            backend.open_stream()
        self.assertEqual(self.library.events, [])

    def test_nonfinite_input_silences_aborts_and_health_poll_stops_engine(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                engine, raw = self.start_engine()
                output = np.ones((4, 2), dtype=np.float32)
                with self.assertRaises(CallbackAbort):
                    self.callback(raw, np.full((4, 2), value, dtype=np.float32), output)
                np.testing.assert_array_equal(output, 0)
                with self.assertRaisesRegex(EngineError, "Non-finite"):
                    engine.check_health()
                self.assertEqual(engine.state, EngineState.FAULT)
                self.assertTrue(raw.closed)
                self.assertFalse(engine.has_stream)

    def test_callback_status_is_sticky_and_never_resumes_sound(self):
        engine, raw = self.start_engine()
        data = np.full((4, 2), 0.01, dtype=np.float32)
        for status in (True, False):
            output = np.ones((4, 2), dtype=np.float32)
            with self.assertRaises(CallbackAbort):
                self.callback(raw, data, output, status=status)
            np.testing.assert_array_equal(output, 0)
        with self.assertRaisesRegex(EngineError, "callback status"):
            engine.check_health()
        self.assertTrue(raw.closed)

    def test_callback_status_details_survive_fault_cleanup(self):
        class Status:
            def __str__(self):
                return "input overflow, output underflow"

        engine, raw = self.start_engine()
        output = np.ones((4, 2), dtype=np.float32)
        with self.assertRaises(CallbackAbort):
            self.callback(raw, np.zeros_like(output), output, status=Status())
        np.testing.assert_array_equal(output, 0)
        with self.assertRaises(EngineError) as caught:
            engine.check_health()
        message = str(caught.exception)
        self.assertIn("input overflow, output underflow", message)
        self.assertIn("rate=48000, block=4", message)
        self.assertIn("selected input -> selected output", message)
        self.assertTrue(raw.closed)
        self.assertFalse(engine.has_stream)

    def test_callback_frames_shapes_and_dtype_fail_closed(self):
        cases = (
            (np.ones((4, 2), dtype=np.float32), np.ones((4, 2), dtype=np.float32), 3),
            (np.ones((3, 2), dtype=np.float32), np.ones((4, 2), dtype=np.float32), 4),
            (np.ones((4, 2), dtype=np.float32), np.ones((3, 2), dtype=np.float32), 4),
            (np.ones((4, 2), dtype=np.float64), np.ones((4, 2), dtype=np.float32), 4),
            (np.ones((4, 2), dtype=np.float32), np.ones((4, 2), dtype=np.float64), 4),
        )
        for data, output, frames in cases:
            with self.subTest(shape=data.shape, dtype=data.dtype, frames=frames):
                engine, raw = self.start_engine()
                with self.assertRaises(CallbackAbort):
                    self.callback(raw, data, output, frames=frames)
                np.testing.assert_array_equal(output, 0)
                with self.assertRaisesRegex(EngineError, "frame count, shape or dtype"):
                    engine.check_health()

    def test_processing_exception_silences_and_preserves_original_cause(self):
        engine, raw = self.start_engine()
        data = np.zeros((4, 2), dtype=np.float32)
        output = np.ones((4, 2), dtype=np.float32)
        cause = RuntimeError("numeric processing failed")
        def fail_after_partial_write(*args, **kwargs):
            kwargs["out"].fill(0.9)
            raise cause
        with patch.object(np, "clip", side_effect=fail_after_partial_write):
            with self.assertRaises(CallbackAbort):
                self.callback(raw, data, output)
        np.testing.assert_array_equal(output, 0)
        with self.assertRaises(EngineError) as caught:
            engine.check_health()
        self.assertIs(caught.exception.__cause__.__cause__, cause)
        self.assertTrue(raw.closed)

    def test_inactive_stream_is_faulted_without_automatic_restart(self):
        engine, raw = self.start_engine()
        raw.active = False
        with self.assertRaisesRegex(EngineError, "not active"):
            engine.check_health()
        self.assertTrue(raw.closed)
        self.assertEqual(engine.state, EngineState.FAULT)
        self.assertEqual(len(self.library.streams), 1)

    def test_callback_fault_during_start_is_not_reported_as_running(self):
        def fail_callback(raw):
            with self.assertRaises(CallbackAbort):
                self.callback(
                    raw, np.zeros((4, 2), dtype=np.float32), status=True,
                )
            raw.active = False
        self.library.on_start = fail_callback
        engine = AudioEngine(SoundDeviceBackend(self.config, output_confirmed=True))
        with self.assertRaisesRegex(EngineError, "callback status"):
            engine.start()
        self.assertEqual(engine.state, EngineState.FAULT)
        self.assertTrue(self.library.streams[0].closed)
        self.assertFalse(engine.has_stream)

    def test_success_shaped_native_stop_and_close_are_not_accepted(self):
        engine, raw = self.start_engine()
        with patch.object(raw, "stop", return_value=None):
            with self.assertRaisesRegex(EngineError, "did not stop"):
                engine.stop()
        self.assertTrue(raw.closed)
        self.assertFalse(engine.has_stream)
        engine.stop()
        engine.start()
        raw = self.library.streams[-1]
        with patch.object(raw, "close", return_value=None):
            with self.assertRaisesRegex(EngineError, "did not close"):
                engine.stop()
        self.assertTrue(engine.has_stream)
        engine.stop()
        self.assertTrue(raw.closed)
        self.assertFalse(engine.has_stream)

    def test_finite_mask_is_preallocated_and_reused(self):
        _, raw = self.start_engine()
        scratch = raw.options["callback"].__self__._finite
        original_isfinite = np.isfinite
        def check_mask(data, *, out):
            self.assertIs(out, scratch)
            return original_isfinite(data, out=out)
        data = np.zeros((4, 2), dtype=np.float32)
        output = np.empty_like(data)
        with patch.object(np, "isfinite", side_effect=check_mask) as operation:
            for _ in range(10):
                self.callback(raw, data, output)
            self.assertEqual(operation.call_count, 10)

    def test_native_start_and_close_failure_retains_owned_handle_for_retry(self):
        def fail_start(stream):
            stream.errors["close"] = OSError("close failed")
            raise OSError("start failed")
        self.library.on_start = fail_start
        engine = AudioEngine(SoundDeviceBackend(self.config, output_confirmed=True))
        with self.assertRaisesRegex(EngineError, "start failed") as caught:
            engine.start()
        self.assertEqual([f.operation for f in caught.exception.failures], ["start", "close"])
        self.assertTrue(engine.has_stream)
        raw = self.library.streams[0]
        self.assertFalse(raw.closed)
        with self.assertRaisesRegex(EngineError, "fault"):
            engine.start()
        raw.errors.clear()
        engine.stop()
        self.assertTrue(raw.closed)
        self.assertFalse(engine.has_stream)
        self.assertEqual(len(self.library.streams), 1)

    def test_import_does_not_load_numpy_sounddevice_or_qt(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             "import sys; import voice_changer.audio.sounddevice_backend; "
             "assert not any(n.startswith(('numpy', 'sounddevice', 'PySide6')) "
             "for n in sys.modules)"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
