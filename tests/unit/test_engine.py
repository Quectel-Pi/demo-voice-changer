"""Lifecycle tests use only in-test substitutes, never real devices."""

import unittest

from voice_changer.audio.engine import AudioEngine, EngineError, EngineState


class FakeStream:
    def __init__(self, events):
        self.events = events
        self.errors = {}
        self.hooks = {}

    def call(self, operation):
        self.events.append(operation)
        if operation in self.hooks:
            self.hooks[operation]()
        if operation in self.errors:
            raise self.errors[operation]

    def start(self):
        self.call("start")

    def stop(self):
        self.call("stop")

    def close(self):
        self.call("close")

    def check_health(self):
        self.call("health")


class FakeBackend:
    def __init__(self):
        self.events = []
        self.streams = []
        self.open_error = None
        self.on_open = None
        self.stream_errors = {}
        self.stream_hooks = {}

    def open_stream(self):
        self.events.append("open")
        if self.on_open:
            self.on_open()
        if self.open_error:
            raise self.open_error
        stream = FakeStream(self.events)
        stream.errors = self.stream_errors.copy()
        stream.hooks = self.stream_hooks.copy()
        self.streams.append(stream)
        return stream


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.engine = AudioEngine(self.backend)

    def assert_fault(self, operations, *, has_stream=False):
        self.assertEqual(self.engine.state, EngineState.FAULT)
        self.assertEqual(self.engine.has_stream, has_stream)
        self.assertIsNotNone(self.engine.last_error)
        self.assertEqual(
            [failure.operation for failure in self.engine.last_error.failures],
            operations,
        )

    def test_safe_defaults_and_idle_stop_do_not_open_stream(self):
        self.assertEqual(self.engine.state, EngineState.STOPPED)
        self.assertFalse(self.engine.has_stream)
        self.assertIsNone(self.engine.last_error)
        self.engine.stop()
        self.assertEqual(self.backend.events, [])

    def test_missing_backend_never_reports_running(self):
        engine = AudioEngine()
        with self.assertRaisesRegex(EngineError, "not configured") as caught:
            engine.start()
        self.assertEqual(engine.state, EngineState.FAULT)
        self.assertIs(engine.last_error, caught.exception)
        self.assertFalse(engine.has_stream)
        engine.stop()
        self.assertEqual(engine.state, EngineState.STOPPED)

    def test_successful_start_stop_and_restart(self):
        self.engine.start()
        self.assertEqual(self.engine.state, EngineState.RUNNING)
        self.assertTrue(self.engine.has_stream)
        self.engine.stop()
        self.assertEqual(self.engine.state, EngineState.STOPPED)
        self.assertFalse(self.engine.has_stream)
        self.assertEqual(self.backend.events, ["open", "start", "stop", "close"])
        self.engine.stop()
        self.assertEqual(len(self.backend.events), 4)
        self.engine.start()
        self.assertEqual(len(self.backend.streams), 2)
        self.assertEqual(self.engine.state, EngineState.RUNNING)
        self.engine.stop()

    def test_running_start_is_rejected_without_second_stream(self):
        self.engine.start()
        with self.assertRaisesRegex(EngineError, "running"):
            self.engine.start()
        self.assertEqual(self.engine.state, EngineState.RUNNING)
        self.assertIsNone(self.engine.last_error)
        self.assertEqual(self.backend.events, ["open", "start"])
        self.engine.stop()

    def test_transitions_and_reentry_are_guarded(self):
        def verify_starting():
            self.assertEqual(self.engine.state, EngineState.STARTING)
            for action in (self.engine.start, self.engine.stop):
                with self.assertRaisesRegex(EngineError, "starting"):
                    action()

        self.backend.on_open = verify_starting
        self.backend.stream_hooks["start"] = verify_starting
        self.engine.start()
        stream = self.backend.streams[-1]

        def verify_stopping():
            self.assertEqual(self.engine.state, EngineState.STOPPING)
            for action in (self.engine.start, self.engine.stop):
                with self.assertRaisesRegex(EngineError, "stopping"):
                    action()

        stream.hooks["stop"] = verify_stopping
        stream.hooks["close"] = verify_stopping
        self.engine.stop()
        self.assertEqual(self.backend.events, ["open", "start", "stop", "close"])

    def test_open_failure_preserves_cause_and_requires_explicit_stop(self):
        cause = OSError("permission denied")
        self.backend.open_error = cause
        with self.assertRaises(EngineError) as caught:
            self.engine.start()
        self.assert_fault(["open"])
        self.assertIs(caught.exception.__cause__, cause)
        self.assertIs(caught.exception.failures[0].error, cause)
        self.assertEqual(self.backend.events, ["open"])
        self.backend.open_error = None
        with self.assertRaisesRegex(EngineError, "fault"):
            self.engine.start()
        self.engine.stop()
        self.assertIsNone(self.engine.last_error)
        self.assertEqual(self.backend.events, ["open"])
        self.engine.start()
        self.engine.stop()

    def test_start_failure_attempts_stop_and_close(self):
        self.backend.stream_errors["start"] = OSError("device vanished")
        with self.assertRaisesRegex(EngineError, "device vanished"):
            self.engine.start()
        self.assert_fault(["start"])
        self.assertEqual(self.backend.events, ["open", "start", "stop", "close"])

    def test_start_and_cleanup_failures_are_all_preserved(self):
        self.backend.stream_errors = {
            name: OSError(f"{name} failed") for name in ("start", "stop", "close")
        }
        with self.assertRaises(EngineError) as caught:
            self.engine.start()
        self.assert_fault(["start", "stop", "close"], has_stream=True)
        for operation in ("start", "stop", "close"):
            self.assertIn(f"{operation} failed", str(caught.exception))
        with self.assertRaisesRegex(EngineError, "fault"):
            self.engine.start()
        self.assertEqual(len(self.backend.streams), 1)
        self.backend.streams[0].errors.clear()
        self.engine.stop()
        self.assertEqual(self.engine.state, EngineState.STOPPED)
        self.assertFalse(self.engine.has_stream)

    def test_stop_failure_still_closes_and_remains_faulted(self):
        self.engine.start()
        self.backend.streams[0].errors["stop"] = OSError("stop failed")
        with self.assertRaisesRegex(EngineError, "stop failed"):
            self.engine.stop()
        self.assert_fault(["stop"])
        self.assertEqual(self.backend.events, ["open", "start", "stop", "close"])
        self.engine.stop()
        self.assertEqual(self.engine.state, EngineState.STOPPED)
        self.assertEqual(len(self.backend.events), 4)

    def test_close_failure_retains_handle_and_retries_before_restart(self):
        self.engine.start()
        self.backend.streams[0].errors["close"] = OSError("close failed")
        with self.assertRaisesRegex(EngineError, "close failed"):
            self.engine.stop()
        self.assert_fault(["close"], has_stream=True)
        with self.assertRaisesRegex(EngineError, "fault"):
            self.engine.start()
        with self.assertRaisesRegex(EngineError, "close failed"):
            self.engine.stop()
        self.assert_fault(["close"], has_stream=True)
        self.assertEqual(len(self.backend.streams), 1)
        self.backend.streams[0].errors.clear()
        self.engine.stop()
        self.assertEqual(self.backend.events[-2:], ["stop", "close"])
        self.assertFalse(self.engine.has_stream)
        self.assertIsNone(self.engine.last_error)
        self.assertEqual(self.engine.state, EngineState.STOPPED)

    def test_runtime_fault_stops_releases_and_does_not_restart(self):
        self.engine.start()
        cause = OSError("input disconnected")
        with self.assertRaises(EngineError) as caught:
            self.engine.handle_fault(cause)
        self.assert_fault(["runtime"])
        self.assertIs(caught.exception.__cause__, cause)
        self.assertEqual(self.backend.events, ["open", "start", "stop", "close"])
        self.engine.stop()
        self.assertEqual(self.engine.state, EngineState.STOPPED)
        self.assertEqual(len(self.backend.events), 4)

    def test_runtime_fault_preserves_cleanup_errors(self):
        self.engine.start()
        stream = self.backend.streams[0]
        stream.errors = {
            "stop": OSError("stop failed"), "close": OSError("close failed"),
        }
        with self.assertRaises(EngineError):
            self.engine.handle_fault(OSError("stream failed"))
        self.assert_fault(["runtime", "stop", "close"], has_stream=True)
        stream.errors.clear()
        self.engine.stop()
        self.assertEqual(self.engine.state, EngineState.STOPPED)

    def test_fault_report_outside_running_does_not_change_state(self):
        for state in (EngineState.STOPPED, EngineState.FAULT):
            with self.subTest(state=state):
                if state is EngineState.FAULT:
                    self.backend.open_error = OSError("open failed")
                    with self.assertRaises(EngineError):
                        self.engine.start()
                previous = self.engine.last_error
                with self.assertRaisesRegex(EngineError, state.value):
                    self.engine.handle_fault(OSError("unexpected fault"))
                self.assertEqual(self.engine.state, state)
                self.assertIs(self.engine.last_error, previous)

    def test_health_poll_stops_on_fault_and_requires_running(self):
        with self.assertRaisesRegex(EngineError, "stopped"):
            self.engine.check_health()
        self.engine.start()
        self.engine.check_health()
        self.assertEqual(self.engine.state, EngineState.RUNNING)
        self.backend.streams[0].errors["health"] = OSError("callback failed")
        with self.assertRaisesRegex(EngineError, "callback failed"):
            self.engine.check_health()
        self.assert_fault(["runtime"])
        self.assertEqual(self.backend.events[-3:], ["health", "stop", "close"])


if __name__ == "__main__":
    unittest.main()
