"""Virtual microphone lifecycle through the controller, with a test-only session."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from voice_changer.app.engine_controller import EngineController
from voice_changer.audio import virtual_mic
from voice_changer.audio.device_types import CaptureConfig, OutputConfig, DeviceRef, StreamConfig

from device_fakes import FakeDeviceLibrary
from qt_wait import wait_until


APP = QApplication.instance() or QApplication([])


class FakeSession:
    instances: list["FakeSession"] = []

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.fail = None
        FakeSession.instances.append(self)

    @property
    def running(self) -> bool:
        return self.started > self.stopped

    def start(self) -> None:
        if self.fail is not None:
            raise RuntimeError(self.fail)
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


class FakeStream:
    def __init__(self, backend) -> None:
        self.backend = backend
        self.monitoring = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass

    def check_health(self) -> None:
        pass

    def enable_monitor(self, config, *, output_confirmed=False):
        self.monitoring = True

    def disable_monitor(self):
        self.monitoring = False


class FakeBackend:
    def __init__(self) -> None:
        self.streams = []

    def open_stream(self):
        stream = FakeStream(self)
        self.streams.append(stream)
        return stream


def _capabilities(available=True, reason=None):
    from pathlib import Path

    return virtual_mic.Availability(
        virtual_mic.Support(available, reason),
        Path("/run/user/1000/voice-changer/asound.conf") if available else None,
    )


class VirtualMicControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeSession.instances = []
        self.controller = EngineController(FakeBackend(), preferences_file=None)
        self.assertTrue(wait_until(lambda: self.controller.is_ready, timeout=5.0))

    def tearDown(self) -> None:
        if not self.controller.is_closed:
            self.controller.request_shutdown()
            wait_until(lambda: self.controller.is_closed, timeout=5.0)

    def test_listen_with_virtual_output_starts_and_stops_the_session(self):
        devices = FakeDeviceLibrary()
        self.controller.device_selection  # touch to keep the selection object
        with patch.object(virtual_mic, "VirtualMicSession", FakeSession), \
             patch.object(virtual_mic, "capabilities", lambda: _capabilities()):
            worker = self.controller._worker
            worker._start_virtual_mic(DeviceRef(1, virtual_mic.PCM_NAME, 0))
            self.assertEqual(len(FakeSession.instances), 1)
            self.assertEqual(FakeSession.instances[0].started, 1)
            worker._stop_virtual_mic()
            self.assertEqual(FakeSession.instances[0].stopped, 1)

    def test_unavailable_support_fails_before_opening_audio(self):
        with patch.object(virtual_mic, "capabilities",
                          lambda: _capabilities(False, "no PipeWire session")):
            worker = self.controller._worker
            with self.assertRaises(RuntimeError) as raised:
                worker._start_virtual_mic(DeviceRef(1, virtual_mic.PCM_NAME, 0))
        self.assertIn("no PipeWire session", str(raised.exception))
        self.assertIsNone(worker.virtual_mic)

    def test_real_output_device_never_creates_a_session(self):
        session = FakeSession()
        worker = self.controller._worker
        worker.virtual_mic = session
        worker._start_virtual_mic(DeviceRef(1, "test output", 0))
        self.assertEqual(session.stopped, 1)
        self.assertIsNone(worker.virtual_mic)

    def test_stopping_the_virtual_output_keeps_one_session_running(self):
        with patch.object(virtual_mic, "VirtualMicSession", FakeSession), \
             patch.object(virtual_mic, "capabilities", lambda: _capabilities()):
            worker = self.controller._worker
            reference = DeviceRef(1, virtual_mic.PCM_NAME, 0)
            worker._start_virtual_mic(reference)
            worker._start_virtual_mic(reference)
            self.assertEqual(len(FakeSession.instances), 1)
            self.assertEqual(FakeSession.instances[0].started, 1)

    def test_session_start_failure_is_reported_and_leaves_no_session(self):
        session = FakeSession()
        session.fail = "pw-loopback exited with 1"
        with patch.object(virtual_mic, "VirtualMicSession", lambda: session), \
             patch.object(virtual_mic, "capabilities", lambda: _capabilities()):
            worker = self.controller._worker
            with self.assertRaises(RuntimeError):
                worker._start_virtual_mic(DeviceRef(1, virtual_mic.PCM_NAME, 0))
        self.assertIsNone(worker.virtual_mic)


if __name__ == "__main__":
    unittest.main()
