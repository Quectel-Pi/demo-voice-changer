"""Qt worker tests use only in-test backends, no audio devices or network."""

import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QCoreApplication, QEvent, QTimer, Qt, SIGNAL
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from voice_changer.app.engine_controller import (
    EngineController, _EngineWorker, _LatestPolledStatus, _PolledStatus,
)
from voice_changer.app.device_selection import DeviceSelectionError
from voice_changer.audio.device_types import DeviceRef, StreamConfig
from voice_changer.audio.engine import EngineError, EngineState
from voice_changer.audio.media_status import MediaStatus
from voice_changer.audio.analysis_status import AnalysisReading, AnalysisSnapshot
from voice_changer.dsp.parameters import LiveControl, PRESETS

from controller_fakes import FakeBackend
from device_fakes import FakeDeviceLibrary, FakePortAudioError
from qt_wait import wait_until


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.backend = FakeBackend()
        self.controllers = []
        self.config_fields = dict(samplerate=48000, blocksize=512, channels=1, output_ceiling=0.1)
        self.addCleanup(self.cleanup_controllers)

    def wait_for(self, condition, timeout=2):
        self.assertTrue(wait_until(condition, timeout), "Timed out waiting for Qt worker")

    def test_native_wait_helper_dispatches_timeout_and_propagates_predicate_errors(self):
        events = []
        QTimer.singleShot(0, lambda: events.append(True))
        self.assertTrue(wait_until(lambda: bool(events)))
        self.assertFalse(wait_until(lambda: False, timeout=0.01))
        cause = ValueError("synthetic wait predicate failed")
        def fail():
            raise cause
        with self.assertRaises(ValueError) as caught:
            wait_until(fail)
        self.assertIs(caught.exception, cause)

    def controller(self, backend=None, *, wait=True):
        controller = EngineController(backend)
        self.controllers.append(controller)
        if wait:
            self.wait_for(lambda: controller.is_ready)
        return controller

    def cleanup_controllers(self):
        self.backend.start_release.set()
        self.backend.close_error = None
        self.backend.stop_error = None
        for controller in self.controllers:
            if not controller.is_closed:
                if controller.closing:
                    self.wait_for(lambda: controller.is_closed or not controller.closing)
                if not controller.is_closed:
                    controller.request_shutdown()
                    self.wait_for(lambda: controller.is_closed)
            self.assertIsNone(controller._thread)
            controller.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.controllers.clear()

    def start_running(self, controller):
        self.assertTrue(controller.request_start())
        self.wait_for(lambda: controller.state is EngineState.RUNNING and not controller.busy)

    def device_library(self):
        library = FakeDeviceLibrary()
        patcher = patch.dict(sys.modules, {"sounddevice": library})
        patcher.start()
        def cleanup():
            library.release.set()
            try:
                self.cleanup_controllers()
            finally:
                patcher.stop()
        self.addCleanup(cleanup)
        return library

    def refresh_devices(self, controller):
        self.assertTrue(controller.request_device_refresh())
        self.wait_for(lambda: not controller.busy)

    def select_pair(self, controller):
        devices = controller.device_selection.devices
        self.assertTrue(controller.select_device("input", devices[0].ref))
        self.assertTrue(controller.select_device("output", devices[1].ref))

    def selected_config(self, controller):
        self.refresh_devices(controller)
        self.select_pair(controller)
        return controller.device_selection.build_stream_config(**self.config_fields)

    def test_ai_processor_factory_reaches_backend_on_control_thread_and_fault_is_released(self):
        from voice_changer.audio import sounddevice_backend
        self.device_library()
        controller = self.controller(self.backend)
        config = replace(self.selected_config(controller), blocksize=2496)
        factory = lambda config, control: None
        constructions = []
        errors = []
        controller.errorOccurred.connect(errors.append)

        def construct(value, *, output_confirmed, control, processor_factory):
            constructions.append((value, output_confirmed, processor_factory, threading.get_ident()))
            return self.backend

        with patch.object(sounddevice_backend, "SoundDeviceBackend", side_effect=construct):
            self.assertFalse(controller.request_listen(
                config, PRESETS["original"], processor_factory=factory,
            ))
            self.assertEqual(constructions, [])
            self.assertTrue(controller.request_listen(
                config, PRESETS["original"], output_confirmed=True, processor_factory=factory,
            ))
            self.wait_for(lambda: not controller.busy)
            self.assertEqual(controller.state, EngineState.RUNNING)
            self.assertEqual(constructions[0][:3], (config, True, factory))
            self.assertNotEqual(constructions[0][3], threading.get_ident())
            self.backend.health_fault.set()
            self.wait_for(lambda: controller.state is EngineState.FAULT)
            self.assertFalse(controller.owns_devices)
            self.assertIn("disconnected", str(errors[-1]))
            self.assertTrue(controller.request_stop())
            self.wait_for(lambda: not controller.busy)
            self.assertEqual(controller.state, EngineState.STOPPED)

    def test_stale_meter_sources_do_not_replace_the_current_session(self):
        controller = self.controller()
        controller._state = EngineState.RUNNING
        controller._live_control = LiveControl(PRESETS["original"])
        received = []
        controller.liveStatus.connect(received.append)
        try:
            controller._on_media_status(MediaStatus(monitoring=True, meter_source_id=2))
            controller._on_live_status(((0.8, 0.9), 0, 0, 1))
            controller._on_media_status(MediaStatus(monitoring=True, meter_source_id=1))
            self.assertEqual(controller.media_status.meter_source_id, 2)
            self.assertEqual(received, [])
            controller._on_live_status(((0.1, 0.2), 0, 0, 2))
            self.assertEqual(received, [((0.1, 0.2), 0, 0)])
            controller._on_media_status(MediaStatus(playback="effect", meter_source_id=3))
            controller._on_live_status(((0.4, 0.5), 0, 0, 2))
            self.assertEqual(len(received), 1)
        finally:
            controller._state = EngineState.STOPPED
            controller._live_control = None

    def test_latest_periodic_status_has_only_one_replaceable_slot(self):
        mailbox = _LatestPolledStatus()
        for index in range(1000):
            latest = _PolledStatus(MediaStatus(meter_source_id=index), None)
            mailbox.publish(latest)
        self.assertEqual(mailbox.take(), latest)
        self.assertIsNone(mailbox.take())

    def test_playback_snapshot_reads_peak_and_sequence_as_one_sample(self):
        class Playback:
            sample = None
            reads = 0

            @property
            def output_peak_sample(self):
                self.reads += 1
                return self.sample

        playback = Playback()
        worker = _EngineWorker(None, _LatestPolledStatus())
        self.addCleanup(worker.deleteLater)
        worker.engine._state = EngineState.RUNNING
        worker.playing = "original"
        worker.playback_backend = SimpleNamespace(stream=playback)
        for sample, expected in (
            (None, (None, 0)),
            ((1, 0.0), (0.0, 1)),
            ((2, 0.0), (0.0, 2)),
            ((3, 0.25), (0.25, 3)),
        ):
            with self.subTest(sample=sample):
                playback.sample = sample
                before = playback.reads
                status = worker._media_snapshot()
                self.assertEqual((status.output_peak, status.output_peak_sequence), expected)
                self.assertEqual(playback.reads, before + 1)

    def test_analysis_snapshot_comes_from_the_current_capture_or_playback_only(self):
        worker = _EngineWorker(None, _LatestPolledStatus())
        self.addCleanup(worker.deleteLater)
        worker.engine._state = EngineState.RUNNING
        capture = AnalysisSnapshot(1, 48000, AnalysisReading((0.1, 0.2, 0.3), 220), None)
        playback = AnalysisSnapshot(2, 48000, None, AnalysisReading((0.4, 0.3, 0.2), 440))
        worker.live_control = LiveControl(PRESETS["original"])
        worker.live_control.analysis = SimpleNamespace(snapshot=capture)
        self.assertIs(worker._media_snapshot().analysis, capture)
        worker.playback_backend = SimpleNamespace(
            stream=SimpleNamespace(output_peak_sample=None, analysis=SimpleNamespace(snapshot=playback)),
        )
        self.assertIs(worker._media_snapshot().analysis, playback)

    def test_periodic_updates_do_not_accumulate_while_gui_is_blocked(self):
        controller = self.controller(self.backend)
        self.start_running(controller)
        statuses = []
        controller.mediaStatus.connect(statuses.append)
        health_before = sum(event[0] == "health" for event in self.backend.events)
        time.sleep(0.35)
        health_after = sum(event[0] == "health" for event in self.backend.events)
        self.assertGreaterEqual(health_after - health_before, 4)
        self.wait_for(lambda: bool(statuses))
        self.assertEqual(len(statuses), 1)

    def test_coalesced_status_does_not_cross_activity_or_source_transitions(self):
        controller = self.controller()
        controller._state = EngineState.RUNNING
        controller._live_control = LiveControl(PRESETS["original"])
        current = MediaStatus(monitoring=True, meter_source_id=2)
        controller._media_status = current
        received = []
        controller.liveStatus.connect(received.append)
        try:
            for other in (
                replace(current, meter_source_id=1),
                replace(current, meter_source_id=3),
                replace(current, monitoring=False, recording=True),
                replace(current, playback="original"),
            ):
                controller._polled_status.publish(_PolledStatus(other, ((0.8, 0.9), 0, 0, 2)))
                controller._receive_polled_status()
                self.assertEqual(controller.media_status, current)
                self.assertEqual(received, [])
            latest = replace(current, duration=0.25)
            controller._polled_status.publish(_PolledStatus(latest, ((0.1, 0.2), 0, 0, 2)))
            controller._receive_polled_status()
            self.assertEqual(controller.media_status, latest)
            self.assertEqual(received, [((0.1, 0.2), 0, 0)])
        finally:
            controller._state = EngineState.STOPPED
            controller._live_control = None

    def test_prepare_constructs_unconfirmed_backend_on_worker_and_start_is_blocked(self):
        from voice_changer.audio import sounddevice_backend
        library = self.device_library()
        controller = self.controller(self.backend)
        config = self.selected_config(controller)
        constructor = sounddevice_backend.SoundDeviceBackend
        constructions, prepared, signal_threads, errors = [], [], [], []
        controller.backendPrepared.connect(
            lambda value: (prepared.append(value), signal_threads.append(threading.get_ident()))
        )
        controller.errorOccurred.connect(errors.append)
        calls = list(library.calls)
        def construct(value, *, output_confirmed):
            constructions.append((value, output_confirmed, threading.get_ident()))
            return constructor(value, output_confirmed=output_confirmed)
        with patch.object(sounddevice_backend, "SoundDeviceBackend", side_effect=construct):
            self.assertTrue(controller.request_backend_prepare(config))
            self.assertTrue(controller.busy)
            self.wait_for(lambda: not controller.busy)
        self.assertEqual(prepared, [config])
        self.assertEqual(len(constructions), 1)
        self.assertEqual(constructions[0][:2], (config, False))
        self.assertEqual(constructions[0][2], library.calls[0][1])
        self.assertNotEqual(constructions[0][2], threading.get_ident())
        self.assertEqual(signal_threads, [threading.get_ident()])
        self.assertEqual(errors, [])
        self.assertEqual(library.calls, calls)
        self.assertEqual(self.backend.events, [])
        self.assertEqual(controller.state, EngineState.STOPPED)
        self.assertFalse(controller.owns_devices)
        with patch.dict(sys.modules, {"numpy": None}):
            self.assertTrue(controller.request_start())
            self.wait_for(lambda: controller.state is EngineState.FAULT and not controller.busy)
        self.assertIn("Output requires explicit device and volume confirmation", str(errors[-1]))
        self.assertIsInstance(errors[-1].failures[0].error, RuntimeError)
        self.assertFalse(controller.owns_devices)
        self.assertEqual(library.calls, calls)
        self.assertEqual(self.backend.events, [])

    def test_prepare_rejects_bad_or_stale_configuration_without_worker_or_state_change(self):
        library = self.device_library()
        controller = self.controller(self.backend)
        config = self.selected_config(controller)
        errors, prepared = [], []
        controller.errorOccurred.connect(errors.append)
        controller.backendPrepared.connect(prepared.append)
        calls = list(library.calls)
        invalid = (
            None, "default", True,
            replace(config, input_device=DeviceRef(9, "test input", 0)),
            replace(config, input_device=DeviceRef(0, "renamed", 0)),
            replace(config, output_device=DeviceRef(1, "test output", 1)),
            replace(config, channels=2),
        )
        for value in invalid:
            with self.subTest(value=value):
                self.assertFalse(controller.request_backend_prepare(value))
                self.assertEqual(errors[-1].failures[0].operation, "prepare")
                self.assertIs(errors[-1].failures[0].error, errors[-1].__cause__)
                self.assertFalse(controller.busy)
                self.assertEqual(controller.state, EngineState.STOPPED)
                self.assertEqual(controller.device_selection.input_device, config.input_device)
                self.assertEqual(controller.device_selection.output_device, config.output_device)
        library.devices[0]["name"] = "new input"
        self.refresh_devices(controller)
        self.select_pair(controller)
        self.assertFalse(controller.request_backend_prepare(config))
        self.assertEqual(prepared, [])
        self.assertEqual(self.backend.events, [])
        self.assertEqual(len(library.calls), len(calls) + 2)

    def test_accepted_prepare_failure_disarms_old_backend_and_preserves_cause_then_retries(self):
        from voice_changer.audio import sounddevice_backend
        self.device_library()
        controller = self.controller(self.backend)
        config = self.selected_config(controller)
        errors, prepared = [], []
        controller.errorOccurred.connect(errors.append)
        controller.backendPrepared.connect(prepared.append)
        for cause in (OSError("synthetic backend construction failed"),
                      ImportError("synthetic backend import failed")):
            with self.subTest(cause=cause):
                with patch.object(sounddevice_backend, "SoundDeviceBackend", side_effect=cause):
                    self.assertTrue(controller.request_backend_prepare(config))
                    self.wait_for(lambda: not controller.busy)
                self.assertIs(errors[-1].__cause__, cause)
                self.assertIs(errors[-1].failures[0].error, cause)
                self.assertEqual(errors[-1].failures[0].operation, "prepare")
                self.assertEqual(prepared, [])
                self.assertEqual(controller.state, EngineState.STOPPED)
                self.assertFalse(controller.owns_devices)
                self.assertTrue(controller.request_start())
                self.wait_for(lambda: controller.state is EngineState.FAULT and not controller.busy)
                self.assertIn("Audio backend is not configured", str(errors[-1]))
                self.assertEqual(self.backend.events, [])
                self.assertFalse(controller.request_backend_prepare(config))
                self.assertTrue(controller.request_stop())
                self.wait_for(lambda: controller.state is EngineState.STOPPED and not controller.busy)
        self.assertTrue(controller.request_backend_prepare(config))
        self.wait_for(lambda: not controller.busy)
        self.assertEqual(prepared, [config])
        self.assertEqual(self.backend.events, [])

    def test_prepare_busy_constructor_keeps_gui_live_rejects_repeats_and_closes(self):
        from voice_changer.audio import sounddevice_backend
        self.device_library()
        controller = self.controller()
        config = self.selected_config(controller)
        constructor = sounddevice_backend.SoundDeviceBackend
        entered, release = threading.Event(), threading.Event()
        constructions, prepared, heartbeat = [], [], []
        controller.backendPrepared.connect(prepared.append)
        def construct(value, *, output_confirmed):
            constructions.append(value)
            entered.set()
            if not release.wait(timeout=3):
                raise RuntimeError("Test did not release backend constructor")
            return constructor(value, output_confirmed=output_confirmed)
        with patch.object(sounddevice_backend, "SoundDeviceBackend", side_effect=construct):
            try:
                self.assertTrue(controller.request_backend_prepare(config))
                self.wait_for(entered.is_set)
                for _ in range(20):
                    self.assertFalse(controller.request_backend_prepare(config))
                    self.assertFalse(controller.request_start())
                    self.assertFalse(controller.select_device("input", None))
                    self.assertFalse(controller.request_stream_config(**self.config_fields))
                    self.assertFalse(controller.request_device_refresh())
                QTimer.singleShot(0, lambda: heartbeat.append(True))
                self.wait_for(lambda: bool(heartbeat))
                self.assertTrue(controller.request_shutdown())
                self.assertTrue(controller.closing)
                self.assertFalse(controller.request_backend_prepare(config))
                self.assertFalse(controller.is_closed)
            finally:
                release.set()
            self.wait_for(lambda: controller.is_closed)
        self.assertEqual(constructions, [config])
        self.assertEqual(prepared, [])
        self.assertIsNone(controller._thread)

    def test_prepare_owned_stream_and_fault_require_cleanup_and_explicit_stop(self):
        self.device_library()
        controller = self.controller(self.backend)
        config = self.selected_config(controller)
        self.start_running(controller)
        self.assertFalse(controller.request_backend_prepare(config))
        self.backend.close_error = OSError("synthetic retained stream")
        controller.request_stop()
        self.wait_for(lambda: controller.state is EngineState.FAULT and not controller.busy)
        self.assertTrue(controller.owns_devices)
        self.assertFalse(controller.request_backend_prepare(config))
        self.backend.close_error = None
        controller.request_stop()
        self.wait_for(lambda: controller.state is EngineState.STOPPED and not controller.busy)
        self.assertTrue(controller.request_backend_prepare(config))
        self.wait_for(lambda: not controller.busy)
        self.assertFalse(controller.owns_devices)
        self.assertEqual(sum(action == "open" for action, _ in self.backend.events), 1)

    def test_prepare_notifications_can_reenter_shutdown_after_work_is_enqueued(self):
        from voice_changer.audio import sounddevice_backend
        self.device_library()
        controller = self.controller()
        config = self.selected_config(controller)
        constructor = sounddevice_backend.SoundDeviceBackend
        constructions, prepared = [], []
        def construct(value, *, output_confirmed):
            constructions.append(value)
            return constructor(value, output_confirmed=output_confirmed)
        def close_on_busy():
            if controller.busy and not controller.closing and not controller.is_closed:
                controller.request_shutdown()
        controller.deviceSelectionChanged.connect(close_on_busy)
        controller.backendPrepared.connect(prepared.append)
        with patch.object(sounddevice_backend, "SoundDeviceBackend", side_effect=construct):
            self.assertTrue(controller.request_backend_prepare(config))
            self.wait_for(lambda: controller.is_closed)
        self.assertEqual(constructions, [config])
        self.assertEqual(prepared, [])
        self.assertIsNone(controller._thread)

    def test_prepare_completion_notification_can_reenter_shutdown_without_audio(self):
        self.device_library()
        controller = self.controller(self.backend)
        config = self.selected_config(controller)
        prepared = []
        def close_on_prepared(value):
            prepared.append(value)
            self.assertTrue(controller.request_shutdown())
        controller.backendPrepared.connect(close_on_prepared)
        self.assertTrue(controller.request_backend_prepare(config))
        self.wait_for(lambda: controller.is_closed)
        self.assertEqual(prepared, [config])
        self.assertEqual(self.backend.events, [])

    def test_prepare_rejects_initializing_query_closing_and_closed_without_success(self):
        controller = self.controller(wait=False)
        errors, prepared = [], []
        controller.errorOccurred.connect(errors.append)
        controller.backendPrepared.connect(prepared.append)
        self.assertFalse(controller.request_backend_prepare(None))
        self.wait_for(lambda: controller.is_ready)
        library = self.device_library()
        config = self.selected_config(controller)
        library.release.clear()
        try:
            controller.request_device_refresh()
            self.wait_for(library.entered.is_set)
            self.assertFalse(controller.request_backend_prepare(config))
            self.assertTrue(controller.request_shutdown())
            self.assertFalse(controller.request_backend_prepare(config))
        finally:
            library.release.set()
        self.wait_for(lambda: controller.is_closed)
        self.assertFalse(controller.request_backend_prepare(config))
        self.assertEqual(len(errors), 4)
        self.assertEqual(prepared, [])
        self.assertEqual(len(library.calls), 4)

    def test_prepare_worker_rechecks_running_state_without_disarming_live_stream(self):
        library = self.device_library()
        controller = self.controller(self.backend)
        config = self.selected_config(controller)
        self.start_running(controller)
        errors, prepared = [], []
        controller.errorOccurred.connect(errors.append)
        controller.backendPrepared.connect(prepared.append)
        calls = list(library.calls)
        controller._prepareRequested.emit(config)
        self.wait_for(lambda: bool(errors))
        self.assertIsInstance(errors[0].__cause__, EngineError)
        self.assertIn("requires stopped audio and released devices", str(errors[0]))
        self.assertEqual(controller.state, EngineState.RUNNING)
        self.assertTrue(controller.owns_devices)
        self.assertEqual(prepared, [])
        self.assertEqual(library.calls, calls)
        self.assertNotIn("close", [action for action, _ in self.backend.events])
        controller.request_stop()
        self.wait_for(lambda: controller.state is EngineState.STOPPED and not controller.busy)
        self.assertEqual(
            [action for action, _ in self.backend.events if action != "health"],
            ["open", "start", "stop", "close"],
        )

    def test_prepare_in_isolated_process_loads_backend_only_and_blocks_audio_imports(self):
        script = """
import faulthandler, importlib.abc, sys
from PySide6.QtCore import QCoreApplication, QEvent
from qt_wait import wait_until
faulthandler.dump_traceback_later(8, exit=True)
class NoExternalAudio(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("sounddevice", "numpy", "voice_changer.dsp", "voice_changer.ai")):
            raise AssertionError("Unexpected external audio import: " + fullname)
sys.meta_path.insert(0, NoExternalAudio())
from voice_changer.app.engine_controller import EngineController
from voice_changer.audio.device_discovery import DeviceInfo
from voice_changer.audio.device_types import DeviceRef
from voice_changer.audio.engine import EngineState
app = QCoreApplication([])
controller = EngineController()
def wait_for(condition):
    assert wait_until(condition), "Timed out waiting for isolated preparation"
try:
    wait_for(lambda: controller.is_ready)
    assert "voice_changer.audio.sounddevice_backend" not in sys.modules
    ref = DeviceRef(0, "synthetic duplex", 0)
    controller.device_selection.refresh((DeviceInfo(ref, "synthetic API", 2, 2, 48000.0),))
    assert controller.select_device("input", ref) and controller.select_device("output", ref)
    config = controller.device_selection.build_stream_config(samplerate=48000, blocksize=512,
                                                              channels=2, output_ceiling=0.1)
    prepared, errors = [], []
    controller.backendPrepared.connect(prepared.append)
    controller.errorOccurred.connect(errors.append)
    assert controller.request_backend_prepare(config)
    wait_for(lambda: not controller.busy)
    assert prepared == [config] and not errors
    assert controller.state is EngineState.STOPPED and not controller.owns_devices
    assert "voice_changer.audio.sounddevice_backend" in sys.modules
    assert "sounddevice" not in sys.modules and "numpy" not in sys.modules
    assert controller.request_start()
    wait_for(lambda: controller.state is EngineState.FAULT and not controller.busy)
    assert "Output requires explicit device and volume confirmation" in str(errors[-1])
    assert not controller.owns_devices
    assert "sounddevice" not in sys.modules and "numpy" not in sys.modules
finally:
    controller.request_shutdown()
    wait_for(lambda: controller.is_closed)
    assert controller._thread is None
    controller.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
faulthandler.cancel_dump_traceback_later()
print("prepared without external audio imports; output blocked")
"""
        root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root / "tests/integration")))
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=root, env=env,
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prepared without external audio imports; output blocked", result.stdout)

    def test_config_request_emits_frozen_value_on_owner_without_audio_or_queue(self):
        library = self.device_library()
        controller = self.controller(self.backend)
        self.refresh_devices(controller)
        self.select_pair(controller)
        configs, signal_threads, errors, operations, selections = [], [], [], [], []
        controller.streamConfigReady.connect(
            lambda config: (configs.append(config), signal_threads.append(threading.get_ident()))
        )
        controller.errorOccurred.connect(errors.append)
        controller.operationFinished.connect(lambda: operations.append(True))
        controller.deviceSelectionChanged.connect(lambda: selections.append(True))
        calls = list(library.calls)
        self.assertTrue(controller.request_stream_config(**self.config_fields))
        self.assertEqual(len(configs), 1)
        config = configs[0]
        self.assertIsInstance(config, StreamConfig)
        self.assertEqual(config.input_device, controller.device_selection.input_device)
        self.assertEqual(config.output_device, controller.device_selection.output_device)
        for field, value in self.config_fields.items():
            self.assertEqual(getattr(config, field), value)
        with self.assertRaises(FrozenInstanceError):
            config.blocksize = 256
        self.assertEqual(signal_threads, [threading.get_ident()])
        self.assertEqual((errors, operations, selections, self.backend.events), ([], [], [], []))
        self.assertEqual(library.calls, calls)
        self.assertFalse(controller.busy)
        self.assertFalse(controller.owns_devices)
        self.assertEqual(controller.state, EngineState.STOPPED)

    def test_config_errors_preserve_original_cause_and_state_without_success(self):
        library = self.device_library()
        controller = self.controller(self.backend)
        self.refresh_devices(controller)
        configs, errors = [], []
        controller.streamConfigReady.connect(configs.append)
        controller.errorOccurred.connect(errors.append)
        self.assertFalse(controller.request_stream_config(**self.config_fields))
        self.assertIsInstance(errors[-1].__cause__, DeviceSelectionError)
        self.select_pair(controller)
        before = (controller.device_selection.devices, controller.device_selection.input_device,
                  controller.device_selection.output_device)
        calls = list(library.calls)
        for field, value, cause_type in (
            ("samplerate", True, ValueError), ("blocksize", 0, ValueError),
            ("output_ceiling", float("nan"), ValueError),
            ("channels", 2, DeviceSelectionError),
        ):
            with self.subTest(field=field):
                self.assertFalse(controller.request_stream_config(
                    **{**self.config_fields, field: value},
                ))
                error = errors[-1]
                self.assertIsInstance(error, EngineError)
                self.assertIsInstance(error.__cause__, cause_type)
                self.assertIs(error.failures[0].error, error.__cause__)
                self.assertEqual(error.failures[0].operation, "configure")
                self.assertIn(str(error.__cause__), str(error))
                self.assertEqual(
                    (controller.device_selection.devices, controller.device_selection.input_device,
                     controller.device_selection.output_device), before,
                )
                self.assertFalse(controller.busy)
        self.assertEqual(configs, [])
        self.assertEqual(library.calls, calls)
        self.assertEqual(self.backend.events, [])
        cause = DeviceSelectionError("synthetic explicit configuration error")
        with patch.object(controller.device_selection, "build_stream_config", side_effect=cause):
            self.assertFalse(controller.request_stream_config(**self.config_fields))
        self.assertIs(errors[-1].__cause__, cause)
        self.assertIs(errors[-1].failures[0].error, cause)

    def test_config_request_requires_every_explicit_field(self):
        controller = self.controller()
        for field in self.config_fields:
            fields = self.config_fields.copy()
            del fields[field]
            with self.subTest(missing=field), self.assertRaises(TypeError):
                controller.request_stream_config(**fields)
        self.assertIsNone(controller._selection)
        self.assertFalse(controller.devices_loaded)

    def test_config_rejected_during_initialization_query_close_and_after_closed(self):
        controller = self.controller(wait=False)
        errors, configs = [], []
        controller.errorOccurred.connect(errors.append)
        controller.streamConfigReady.connect(configs.append)
        self.assertFalse(controller.request_stream_config(**self.config_fields))
        self.assertIsNone(controller._selection)
        self.wait_for(lambda: controller.is_ready)
        library = self.device_library()
        library.release.clear()
        controller.request_device_refresh()
        self.wait_for(library.entered.is_set)
        self.assertFalse(controller.request_stream_config(**self.config_fields))
        self.assertTrue(controller.busy)
        controller.request_shutdown()
        self.assertFalse(controller.request_stream_config(**self.config_fields))
        library.release.set()
        self.wait_for(lambda: controller.is_closed)
        self.assertFalse(controller.request_stream_config(**self.config_fields))
        self.assertEqual(len(errors), 4)
        self.assertEqual(configs, [])
        self.assertEqual([name for name, _ in library.calls], ["devices", "hostapis"])
        self.assertFalse(controller.devices_loaded)

    def test_config_ownership_and_failed_cleanup_reject_until_explicit_release(self):
        library = self.device_library()
        controller = self.controller(self.backend)
        self.refresh_devices(controller)
        self.select_pair(controller)
        configs, errors = [], []
        controller.streamConfigReady.connect(configs.append)
        controller.errorOccurred.connect(errors.append)
        self.start_running(controller)
        calls = list(library.calls)
        self.assertFalse(controller.request_stream_config(**self.config_fields))
        self.assertIsInstance(errors[-1].__cause__, DeviceSelectionError)
        self.backend.close_error = OSError("synthetic cleanup failure")
        controller.request_stop()
        self.wait_for(lambda: controller.state is EngineState.FAULT and not controller.busy)
        self.assertTrue(controller.owns_devices)
        self.assertFalse(controller.request_stream_config(**self.config_fields))
        self.assertEqual(configs, [])
        self.backend.close_error = None
        controller.request_stop()
        self.wait_for(lambda: controller.state is EngineState.STOPPED and not controller.busy)
        self.assertTrue(controller.request_stream_config(**self.config_fields))
        self.assertEqual(len(configs), 1)
        self.assertEqual(library.calls, calls)
        self.assertEqual(sum(action == "open" for action, _ in self.backend.events), 1)
        self.assertEqual(controller.state, EngineState.STOPPED)

    def test_config_requests_rebuild_from_latest_snapshot_without_reusing_old_value(self):
        library = self.device_library()
        library.devices[0]["max_input_channels"] = 2
        controller = self.controller()
        self.refresh_devices(controller)
        self.select_pair(controller)
        configs = []
        controller.streamConfigReady.connect(configs.append)
        self.assertTrue(controller.request_stream_config(**{**self.config_fields, "channels": 2}))
        library.devices[0]["max_input_channels"] = 1
        self.refresh_devices(controller)
        self.assertFalse(controller.request_stream_config(**{**self.config_fields, "channels": 2}))
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].channels, 2)
        self.assertTrue(controller.request_stream_config(**self.config_fields))
        self.assertEqual(configs[1].channels, 1)
        library.devices[0]["name"] = "replacement input"
        self.refresh_devices(controller)
        self.assertFalse(controller.request_stream_config(**self.config_fields))
        self.assertEqual(len(configs), 2)
        self.select_pair(controller)
        self.assertTrue(controller.request_stream_config(**self.config_fields))
        self.assertEqual(configs[2].input_device.name, "replacement input")
        self.assertEqual(configs[0].input_device.name, "test input")

    def test_config_success_notification_can_reenter_shutdown_without_auto_start(self):
        self.device_library()
        controller = self.controller(self.backend)
        self.refresh_devices(controller)
        self.select_pair(controller)
        configs = []
        def close_on_config(config):
            configs.append(config)
            self.assertFalse(controller.busy)
            self.assertTrue(controller.request_shutdown())
        controller.streamConfigReady.connect(close_on_config)
        self.assertTrue(controller.request_stream_config(**self.config_fields))
        self.wait_for(lambda: controller.is_closed)
        self.assertEqual(len(configs), 1)
        self.assertEqual(self.backend.events, [])
        self.assertIsNone(controller._thread)

    def test_config_error_notification_can_reenter_shutdown_without_success(self):
        controller = self.controller(self.backend)
        errors, configs = [], []
        controller.streamConfigReady.connect(configs.append)
        def close_on_error(error):
            errors.append(error)
            self.assertTrue(controller.request_shutdown())
        controller.errorOccurred.connect(close_on_error)
        self.assertFalse(controller.request_stream_config(**self.config_fields))
        self.wait_for(lambda: controller.is_closed)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0].__cause__, DeviceSelectionError)
        self.assertEqual(configs, [])
        self.assertEqual(self.backend.events, [])

    def test_config_request_in_isolated_process_never_loads_audio_libraries(self):
        script = """
import faulthandler, importlib.abc, sys
from PySide6.QtCore import QCoreApplication, QEvent
from qt_wait import wait_until
faulthandler.dump_traceback_later(8, exit=True)
class NoAudioImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("sounddevice", "numpy", "voice_changer.audio.sounddevice_backend",
                                "voice_changer.dsp", "voice_changer.ai")):
            raise AssertionError("Unexpected import: " + fullname)
sys.meta_path.insert(0, NoAudioImports())
from voice_changer.app.engine_controller import EngineController
from voice_changer.audio.device_discovery import DeviceInfo
from voice_changer.audio.device_types import DeviceRef
app = QCoreApplication([])
controller = EngineController()
def wait_for(condition):
    assert wait_until(condition), "Timed out waiting for isolated controller"
try:
    wait_for(lambda: controller.is_ready)
    ref = DeviceRef(0, "synthetic duplex", 0)
    controller.device_selection.refresh((DeviceInfo(ref, "synthetic API", 2, 2, 48000.0),))
    assert controller.select_device("input", ref)
    assert controller.select_device("output", ref)
    configs, errors = [], []
    controller.streamConfigReady.connect(configs.append)
    controller.errorOccurred.connect(errors.append)
    assert controller.request_stream_config(samplerate=44100, blocksize=256,
                                            channels=2, output_ceiling=0.1)
    assert len(configs) == 1 and configs[0].input_device == ref
    assert not controller.request_stream_config(samplerate=44100, blocksize=256,
                                                channels=3, output_ceiling=0.1)
    assert len(configs) == 1 and len(errors) == 1
    assert not controller.busy and not controller.owns_devices and not controller.devices_loaded
finally:
    controller.request_shutdown()
    wait_for(lambda: controller.is_closed)
    assert controller._thread is None
    controller.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
faulthandler.cancel_dump_traceback_later()
print("configured without audio imports")
"""
        root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root / "tests/integration")))
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=root, env=env,
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("configured without audio imports", result.stdout)

    def test_idle_construct_and_shutdown_never_open_backend(self):
        controller = self.controller(self.backend)
        self.assertEqual(controller.state, EngineState.STOPPED)
        self.assertEqual(self.backend.events, [])
        self.assertTrue(controller.request_shutdown())
        self.wait_for(lambda: controller.is_closed)
        self.assertEqual(self.backend.events, [])
        self.assertTrue(controller.request_shutdown())
        self.assertFalse(controller.is_ready)

    def test_all_backend_operations_are_serial_on_one_non_gui_thread(self):
        controller = self.controller(self.backend)
        signal_threads = []
        states = []
        controller.stateChanged.connect(
            lambda state: (states.append(state), signal_threads.append(threading.get_ident()))
        )
        self.start_running(controller)
        self.wait_for(lambda: any(action == "health" for action, _ in self.backend.events))
        self.assertTrue(controller.request_stop())
        self.wait_for(lambda: controller.state is EngineState.STOPPED and not controller.busy)
        worker_ids = {identity for _, identity in self.backend.events}
        self.assertEqual(len(worker_ids), 1)
        self.assertNotIn(threading.get_ident(), worker_ids)
        self.assertEqual(set(signal_threads), {threading.get_ident()})
        self.assertIn(EngineState.STARTING, states)
        self.assertIn(EngineState.STOPPING, states)
        self.assertEqual(
            [action for action, _ in self.backend.events if action != "health"],
            ["open", "start", "stop", "close"],
        )

    def test_busy_rejects_repeated_requests_without_queue_or_gui_block(self):
        self.backend.start_release.clear()
        controller = self.controller(self.backend)
        errors = []
        controller.errorOccurred.connect(errors.append)
        heartbeat = []
        timer = QTimer()
        timer.setInterval(5)
        timer.timeout.connect(lambda: heartbeat.append(1))
        timer.start()
        self.addCleanup(timer.stop)
        self.assertTrue(controller.request_start())
        self.assertTrue(controller.busy)
        self.wait_for(self.backend.start_entered.is_set)
        for _ in range(10):
            self.assertFalse(controller.request_start())
            self.assertFalse(controller.request_stop())
        self.wait_for(lambda: len(heartbeat) >= 3)
        self.assertEqual(len(errors), 20)
        self.assertEqual([action for action, _ in self.backend.events], ["open", "start"])
        self.backend.start_release.set()
        self.wait_for(lambda: not controller.busy)
        self.assertEqual(controller.state, EngineState.RUNNING)

    def test_missing_backend_is_visible_fault_not_running(self):
        controller = self.controller()
        errors = []
        controller.errorOccurred.connect(errors.append)
        self.assertTrue(controller.request_start())
        self.wait_for(lambda: controller.state is EngineState.FAULT and not controller.busy)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], EngineError)
        self.assertIn("not configured", str(errors[0]))
        self.assertFalse(controller.request_start())
        self.assertTrue(controller.request_stop())
        self.wait_for(lambda: controller.state is EngineState.STOPPED and not controller.busy)

    def test_runtime_health_fault_is_notified_and_stops_without_restart(self):
        controller = self.controller(self.backend)
        errors = []
        controller.errorOccurred.connect(errors.append)
        self.start_running(controller)
        self.backend.health_fault.set()
        self.wait_for(lambda: controller.state is EngineState.FAULT)
        self.assertEqual(len(errors), 1)
        self.assertIn("disconnected", str(errors[0]))
        self.assertEqual([a for a, _ in self.backend.events][-2:], ["stop", "close"])
        self.assertEqual(sum(a == "open" for a, _ in self.backend.events), 1)
        self.assertTrue(controller.request_stop())
        self.wait_for(lambda: controller.state is EngineState.STOPPED and not controller.busy)

    def test_start_failure_preserves_cause_and_stops_polling(self):
        cause = OSError("permission denied")
        self.backend.start_error = cause
        controller = self.controller(self.backend)
        errors = []
        controller.errorOccurred.connect(errors.append)
        controller.request_start()
        self.wait_for(lambda: controller.state is EngineState.FAULT and not controller.busy)
        self.assertIs(errors[0].__cause__, cause)
        self.assertEqual([a for a, _ in self.backend.events], ["open", "start", "stop", "close"])

    def test_shutdown_while_starting_is_queued_once_and_releases_after_start(self):
        self.backend.start_release.clear()
        controller = self.controller(self.backend)
        completed = []
        controller.shutdownComplete.connect(lambda: completed.append(True))
        controller.request_start()
        self.wait_for(self.backend.start_entered.is_set)
        self.assertTrue(controller.request_shutdown())
        self.assertTrue(controller.closing)
        self.assertFalse(controller.request_shutdown())
        self.assertFalse(controller.request_start())
        self.assertFalse(controller.request_stop())
        self.assertFalse(controller.is_closed)
        self.backend.start_release.set()
        self.wait_for(lambda: controller.is_closed)
        self.assertEqual(completed, [True])
        self.assertEqual([a for a, _ in self.backend.events if a != "health"],
                         ["open", "start", "stop", "close"])
        self.assertIsNone(controller._thread)

    def test_shutdown_before_ready_still_exits_without_opening_stream(self):
        controller = self.controller(self.backend, wait=False)
        self.assertTrue(controller.request_shutdown())
        self.wait_for(lambda: controller.is_closed)
        self.assertEqual(self.backend.events, [])

    def test_failed_shutdown_keeps_thread_for_explicit_cleanup_retry(self):
        controller = self.controller(self.backend)
        errors, failed, completed = [], [], []
        controller.errorOccurred.connect(errors.append)
        controller.shutdownFailed.connect(lambda: failed.append(True))
        controller.shutdownComplete.connect(lambda: completed.append(True))
        self.start_running(controller)
        self.backend.close_error = OSError("close failed")
        self.assertTrue(controller.request_shutdown())
        self.wait_for(lambda: bool(failed))
        self.assertEqual(controller.state, EngineState.FAULT)
        self.assertFalse(controller.is_closed)
        self.assertFalse(controller.closing)
        self.assertTrue(controller._thread.isRunning())
        self.assertEqual(controller.receivers(SIGNAL("_shutdownRequested()")), 1)
        self.assertEqual(controller._worker.timer.receivers(SIGNAL("timeout()")), 1)
        self.assertEqual(completed, [])
        self.assertIn("close failed", str(errors[0]))
        self.backend.close_error = None
        self.assertTrue(controller.request_shutdown())
        self.wait_for(lambda: controller.is_closed)
        self.assertEqual(completed, [True])
        self.assertEqual([a for a, _ in self.backend.events][-4:], ["stop", "close", "stop", "close"])

    def test_requests_from_foreign_thread_raise_without_enqueue(self):
        controller = self.controller(self.backend)
        errors = []
        def request_from_other_thread():
            try:
                controller.request_start()
            except RuntimeError as error:
                errors.append(error)
        thread = threading.Thread(target=request_from_other_thread)
        thread.start()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn("owning thread", str(errors[0]))
        self.assertEqual(self.backend.events, [])

    def test_construct_without_application_or_on_foreign_thread_is_rejected(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             "from voice_changer.app.engine_controller import EngineController; "
             "EngineController()"],
            env=env, capture_output=True, text=True, timeout=15,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("existing Qt application GUI thread", result.stderr)
        errors = []
        def construct_from_other_thread():
            try:
                EngineController(self.backend)
            except RuntimeError as error:
                errors.append(error)
        thread = threading.Thread(target=construct_from_other_thread)
        thread.start()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertEqual(self.backend.events, [])

    def test_worker_is_destroyed_before_thread_finishes_and_completion(self):
        controller = self.controller(self.backend)
        order = []
        finished = QSignalSpy(controller._thread.finished)
        # Native observation avoids queued cancellation and Python at thread teardown.
        controller._worker.destroyed.connect(
            lambda: order.append(("worker destroyed", finished.count())),
            Qt.ConnectionType.DirectConnection,
        )
        controller.shutdownComplete.connect(lambda: order.append(("complete", finished.count())))
        self.assertTrue(controller.request_shutdown())
        self.wait_for(lambda: controller.is_closed)
        self.assertEqual(order, [("worker destroyed", 0), ("complete", 1)])
        self.assertIsNone(controller._thread)

    def test_incoming_worker_connections_are_detached_before_destruction(self):
        controller = self.controller()
        thread = controller._thread
        requests = (
            "_startRequested()", "_stopRequested()", "_shutdownRequested()",
            "_devicesRequested()", "_prepareRequested(PyObject)",
        )
        self.assertEqual([controller.receivers(SIGNAL(name)) for name in requests], [1] * 5)
        self.assertEqual(thread.receivers(SIGNAL("started()")), 1)
        timer_counts = []
        request_counts = []
        controller._worker.shutdownSucceeded.connect(
            lambda: timer_counts.append(controller._worker.timer.receivers(SIGNAL("timeout()"))),
            Qt.ConnectionType.DirectConnection,
        )
        controller._worker.destroyed.connect(
            lambda: request_counts.append((
                [controller.receivers(SIGNAL(name)) for name in requests],
                thread.receivers(SIGNAL("started()")),
            )),
            Qt.ConnectionType.DirectConnection,
        )
        self.assertTrue(controller.request_shutdown())
        self.wait_for(lambda: controller.is_closed)
        self.assertEqual(timer_counts, [0])
        self.assertEqual(request_counts, [([0] * 5, 0)])

    def test_lazy_selection_before_ready_unlocks_without_querying(self):
        library = self.device_library()
        controller = self.controller(wait=False)
        selection = controller.device_selection
        self.assertTrue(selection.locked)
        self.wait_for(lambda: controller.is_ready)
        self.assertFalse(selection.locked)
        self.assertEqual(selection.devices, ())
        self.assertFalse(controller.devices_loaded)
        self.assertIsNone(selection.input_device)
        self.assertEqual(library.calls, [])

    def test_device_queries_are_serial_off_gui_and_selection_notifies_gui(self):
        library = self.device_library()
        controller = self.controller(self.backend)
        notifications = []
        controller.deviceSelectionChanged.connect(
            lambda: notifications.append(threading.get_ident()),
        )
        self.refresh_devices(controller)
        devices = controller.device_selection.devices
        self.assertEqual([d.ref.index for d in devices], [0, 1])
        self.assertIsNone(controller.device_selection.input_device)
        self.assertTrue(controller.select_device("input", devices[0].ref))
        self.assertTrue(controller.select_device("output", devices[1].ref))
        self.assertEqual(self.backend.events, [])
        self.assertEqual([name for name, _ in library.calls], ["devices", "hostapis"])
        identities = {identity for _, identity in library.calls}
        self.assertEqual(len(identities), 1)
        self.assertNotIn(threading.get_ident(), identities)
        self.assertEqual(set(notifications), {threading.get_ident()})

    def test_blocked_query_rejects_accumulation_and_shutdown_keeps_gui_responsive(self):
        library = self.device_library()
        library.release.clear()
        controller = self.controller()
        self.assertTrue(controller.request_device_refresh())
        self.wait_for(library.entered.is_set)
        for _ in range(20):
            self.assertFalse(controller.request_device_refresh())
            self.assertFalse(controller.request_start())
            self.assertFalse(controller.select_device("input", None))
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        self.wait_for(lambda: bool(heartbeat))
        self.assertTrue(controller.request_shutdown())
        self.assertFalse(controller.is_closed)
        library.release.set()
        self.wait_for(lambda: controller.is_closed)
        self.assertEqual(controller.device_selection.devices, ())
        self.assertFalse(controller.devices_loaded)
        self.assertEqual([name for name, _ in library.calls], ["devices", "hostapis"])

    def test_query_failures_preserve_snapshot_and_original_cause_then_retry(self):
        library = self.device_library()
        controller = self.controller()
        errors = []
        controller.errorOccurred.connect(errors.append)
        self.refresh_devices(controller)
        previous = controller.device_selection.devices
        self.assertTrue(controller.devices_loaded)
        self.assertTrue(controller.select_device("input", previous[0].ref))
        for cause in (OSError("test query failed"), FakePortAudioError("test PA failed")):
            library.error = cause
            self.refresh_devices(controller)
            self.assertIs(errors[-1].failures[0].error, cause)
            self.assertIs(errors[-1].__cause__, cause)
            self.assertEqual(controller.device_selection.devices, previous)
            self.assertTrue(controller.devices_loaded)
            self.assertEqual(controller.device_selection.input_device, previous[0].ref)
            self.assertFalse(controller.device_selection.locked)
        library.error = None
        with patch.dict(sys.modules, {"sounddevice": None}):
            self.refresh_devices(controller)
            self.assertIsInstance(errors[-1].failures[0].error, ModuleNotFoundError)
        self.refresh_devices(controller)
        self.assertEqual(controller.device_selection.devices, previous)

    def test_refresh_invalidation_updates_model_and_surfaces_direction_error(self):
        library = self.device_library()
        controller = self.controller()
        errors = []
        controller.errorOccurred.connect(errors.append)
        self.refresh_devices(controller)
        devices = controller.device_selection.devices
        controller.select_device("input", devices[0].ref)
        controller.select_device("output", devices[1].ref)
        library.devices[0]["max_input_channels"] = 0
        self.refresh_devices(controller)
        self.assertEqual(errors[-1].failures[0].error.invalidated, ("input",))
        self.assertIsNone(controller.device_selection.input_device)
        self.assertEqual(controller.device_selection.output_device, devices[1].ref)
        self.assertEqual(controller.device_selection.devices[0].max_input_channels, 0)
        library.devices[0]["max_input_channels"] = True
        previous = controller.device_selection.devices
        self.refresh_devices(controller)
        self.assertIsInstance(errors[-1].failures[0].error, ValueError)
        self.assertEqual(controller.device_selection.devices, previous)

    def test_cleanup_failure_ownership_locks_devices_until_explicit_release(self):
        library = self.device_library()
        controller = self.controller(self.backend)
        self.refresh_devices(controller)
        self.start_running(controller)
        self.assertTrue(controller.owns_devices)
        self.assertTrue(controller.device_selection.locked)
        calls = len(library.calls)
        self.assertFalse(controller.request_device_refresh())
        self.assertFalse(controller.select_device("input", None))
        self.backend.close_error = OSError("test close failed")
        controller.request_stop()
        self.wait_for(lambda: controller.state is EngineState.FAULT and not controller.busy)
        self.assertTrue(controller.owns_devices)
        self.assertTrue(controller.device_selection.locked)
        self.assertFalse(controller.request_device_refresh())
        self.assertEqual(len(library.calls), calls)
        self.backend.close_error = None
        controller.request_stop()
        self.wait_for(lambda: controller.state is EngineState.STOPPED and not controller.busy)
        self.assertFalse(controller.owns_devices)
        self.assertFalse(controller.device_selection.locked)
        self.refresh_devices(controller)

    def test_health_fault_clean_release_unlocks_without_auto_start_or_selection(self):
        self.device_library()
        controller = self.controller(self.backend)
        self.refresh_devices(controller)
        self.start_running(controller)
        self.backend.health_fault.set()
        self.wait_for(lambda: controller.state is EngineState.FAULT)
        self.assertFalse(controller.owns_devices)
        self.assertFalse(controller.device_selection.locked)
        self.refresh_devices(controller)
        self.assertIsNone(controller.device_selection.input_device)
        self.assertEqual(sum(action == "open" for action, _ in self.backend.events), 1)
        self.assertEqual(controller.state, EngineState.FAULT)

    def test_device_operations_from_foreign_thread_raise_without_query(self):
        library = self.device_library()
        controller = self.controller()
        errors = []
        def foreign():
            for operation in (
                lambda: controller.device_selection,
                controller.request_device_refresh,
                lambda: controller.select_device("input", None),
                lambda: controller.request_stream_config(**self.config_fields),
                lambda: controller.request_backend_prepare(None),
            ):
                try:
                    operation()
                except RuntimeError as error:
                    errors.append(error)
        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 5)
        self.assertEqual(library.calls, [])

    def test_reentrant_device_notification_cannot_enqueue_shutdown_before_start(self):
        controller = self.controller(self.backend)
        controller.device_selection
        def close_on_busy():
            if controller.busy and not controller.closing and not controller.is_closed:
                controller.request_shutdown()
        controller.deviceSelectionChanged.connect(close_on_busy)
        self.assertTrue(controller.request_start())
        self.wait_for(lambda: controller.is_closed)
        self.assertEqual(
            [action for action, _ in self.backend.events], ["open", "start", "stop", "close"],
        )


if __name__ == "__main__":
    unittest.main()
