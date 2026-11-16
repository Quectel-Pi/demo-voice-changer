"""Serialized Qt control thread with asynchronous resource and object cleanup."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from PySide6.QtCore import QCoreApplication, QObject, QThread, QTimer, Signal, Slot

from ..audio.engine import AudioBackend, AudioEngine, EngineError, EngineState, OperationFailure
from ..audio.device_types import CaptureConfig, OutputConfig, StreamConfig
from ..audio.media_status import MediaStatus, PlaybackTrack, RecordingState, Track
from ..audio import virtual_mic
from .preferences import AudioSettings, DeviceIdentity, Preferences, PreferencesStore, check_directory

if TYPE_CHECKING:
    from ..dsp.parameters import EffectParameters, LiveControl
    from .device_selection import DeviceKind, DeviceSelection
    from ..audio.device_discovery import DeviceInfo
    from ..audio.device_types import DeviceRef
    from ..audio.media_backend import CaptureBackend, CaptureStream, PlaybackBackend
    from ..audio.recording import RecordingGroup
    from ..audio.sounddevice_backend import ProcessorFactory


def _device_error(operation: str, cause: Exception) -> EngineError:
    error = EngineError("Device operation failed", (OperationFailure(operation, cause),))
    error.__cause__ = cause
    return error


@dataclass(frozen=True)
class _PolledStatus:
    media: MediaStatus
    live: tuple[tuple[float, float] | None, int, int, int] | None


class _LatestPolledStatus:
    """One replaceable snapshot shared by the control and GUI threads."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._status: _PolledStatus | None = None

    def publish(self, status: _PolledStatus) -> None:
        with self._lock:
            self._status = status

    def take(self) -> _PolledStatus | None:
        with self._lock:
            status = self._status
            self._status = None
            return status


class _EngineWorker(QObject):
    initialized = Signal()
    stateChanged = Signal(object)
    errorOccurred = Signal(object)
    operationFinished = Signal()
    shutdownSucceeded = Signal()
    shutdownFailed = Signal()
    resourcesChanged = Signal(bool)
    devicesDiscovered = Signal(object)
    backendPrepared = Signal(object)
    mediaStatus = Signal(object)
    preferencesChanged = Signal(object)
    saveDirectoryReady = Signal(str)

    def __init__(
        self, backend: AudioBackend | None, polled_status: _LatestPolledStatus,
        preferences_file: Path | None = None,
    ) -> None:
        super().__init__()
        self.engine = AudioEngine(backend)
        self._polled_status = polled_status
        self.live_control: LiveControl | None = None
        self.capture_backend: CaptureBackend | None = None
        self.playback_backend: PlaybackBackend | None = None
        self._meter_source_id = 0
        self.group: RecordingGroup | None = None
        self.recording_state = RecordingState.NONE
        self.monitoring = False
        self.playing: PlaybackTrack | None = None
        self.recoverable: tuple[str, ...] = ()
        self.virtual_mic: virtual_mic.VirtualMicSession | None = None
        self.preferences_store = PreferencesStore(preferences_file) if preferences_file is not None else None
        self.preferences = Preferences()
        self.preferences_dirty = False
        self.devices: tuple[DeviceInfo, ...] = ()
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.poll)

    @Slot()
    def initialize(self) -> None:
        if self.preferences_store is not None:
            try:
                self.preferences = self.preferences_store.load()
                if self.preferences.save_directory is not None:
                    try:
                        check_directory(self.preferences.save_directory)
                    except OSError as error:
                        self.preferences = replace(self.preferences, save_directory=None)
                        self.errorOccurred.emit(_device_error("restore save directory", error))
            except (OSError, ValueError, UnicodeError) as error:
                self.errorOccurred.emit(_device_error("load preferences; using safe defaults", error))
            self.preferencesChanged.emit(self.preferences)
        try:
            from ..audio.recording import recoverable_groups
            self.recoverable = tuple(str(path) for path in recoverable_groups())
        except (OSError, RuntimeError, ValueError) as error:
            self.errorOccurred.emit(_device_error("check recoverable recordings", error))
        self._emit_media()
        self.resourcesChanged.emit(self.engine.has_stream)
        self.initialized.emit()

    def _remember(self, config: StreamConfig | CaptureConfig | OutputConfig) -> None:
        if self.preferences_store is None:
            return
        changes: dict[str, AudioSettings | DeviceIdentity] = {"audio": AudioSettings(
            config.samplerate, config.blocksize, config.channels, config.output_ceiling,
        )}
        try:
            for kind in ("input", "output"):
                if kind == "input" and isinstance(config, (StreamConfig, CaptureConfig)):
                    ref = config.input_device
                elif kind == "output" and isinstance(config, (StreamConfig, OutputConfig)):
                    ref = config.output_device
                else:
                    continue
                info = next(info for info in self.devices if info.ref == ref)
                changes[f"{kind}_device"] = DeviceIdentity(info.ref.name, info.hostapi_name)
            self.preferences = replace(self.preferences, **changes)
        except (ValueError, StopIteration) as error:
            self.errorOccurred.emit(_device_error("remember successful audio configuration", error))
            return
        self.preferences_dirty = True
        self.preferencesChanged.emit(self.preferences)

    def _remember_saved_directory(self) -> None:
        if self.preferences_store is not None and self.group is not None:
            self.preferences = replace(
                self.preferences, save_directory=str(self.group.directory.parent.resolve()),
            )
            self.preferences_dirty = True
            self.preferencesChanged.emit(self.preferences)

    def _flush_preferences(self) -> EngineError | None:
        if self.preferences_store is None or not self.preferences_dirty or self.engine.has_stream:
            return None
        try:
            self.preferences_store.save(self.preferences)
        except (OSError, ValueError) as error:
            return _device_error("save preferences; retry after correcting the path", error)
        self.preferences_dirty = False
        return None

    @Slot()
    def check_save_directory(self) -> None:
        directory = self.preferences.save_directory or str(Path.home())
        try:
            check_directory(directory)
        except OSError as error:
            self.errorOccurred.emit(_device_error("restore save directory; choose a new location", error))
            directory = str(Path.home())
        self.saveDirectoryReady.emit(directory)

    def _execute(
        self, action: Callable[[], None], transition: EngineState,
    ) -> None:
        self.timer.stop()
        self.stateChanged.emit(transition)
        try:
            action()
        except EngineError as error:
            self.errorOccurred.emit(error)
        preference_error = self._flush_preferences()
        if preference_error is not None:
            self.errorOccurred.emit(preference_error)
        self.resourcesChanged.emit(self.engine.has_stream)
        self.stateChanged.emit(self.engine.state)
        self._emit_media()
        if self.engine.state is EngineState.RUNNING:
            self.timer.start()
        self.operationFinished.emit()

    @Slot()
    def start(self) -> None:
        self._execute(self.engine.start, EngineState.STARTING)

    @Slot()
    def stop(self) -> None:
        if (
            self.capture_backend is not None and self.capture_backend.stream is not None
            and self.capture_backend.stream.writer is not None
        ):
            self._media_action("stop monitoring", self._stop_monitor, EngineState.RUNNING)
            return
        transition = (
            EngineState.STOPPED if self.engine.state is EngineState.STOPPED
            else EngineState.STOPPING
        )
        def stop_and_disarm() -> None:
            self.engine.stop()
            if self.live_control is not None:
                self.engine = AudioEngine()
                self.live_control = None
                self.capture_backend = None
                self.playback_backend = None
                self.monitoring = False
                self.playing = None
        self._execute(stop_and_disarm, transition)

    @Slot()
    def poll(self) -> None:
        if self.engine.state is not EngineState.RUNNING:
            return
        try:
            if self.playback_backend is not None and self.playback_backend.stream is not None:
                playback = self.playback_backend.stream
                playback.fill()
                if playback.finished:
                    self._stop_all_impl()
                    self.timer.stop()
                    self.resourcesChanged.emit(self.engine.has_stream)
                    self.stateChanged.emit(self.engine.state)
                    self._emit_media()
                    return
            self.engine.check_health()
        except Exception as error:
            self.timer.stop()
            if not isinstance(error, EngineError) and self.engine.state is EngineState.RUNNING:
                try:
                    self.engine.handle_fault(error)
                except EngineError as fault:
                    error = fault
            self.errorOccurred.emit(error)
            self.resourcesChanged.emit(self.engine.has_stream)
            self.stateChanged.emit(self.engine.state)
            if self.group is not None and self.recording_state is RecordingState.RECORDING:
                self.recording_state = RecordingState.INCOMPLETE
            self._emit_media()
        else:
            live = None
            if self.live_control is not None:
                live = (
                    self.live_control.levels,
                    self.live_control.applied_revision,
                    self.live_control.requested[0],
                    self._meter_source_id,
                )
            self._polled_status.publish(_PolledStatus(self._media_snapshot(), live))

    @Slot(object, object, object)
    def listen(
        self, config: StreamConfig, control: "LiveControl",
        processor_factory: "ProcessorFactory | None" = None,
    ) -> None:
        def start() -> None:
            if self.engine.state is not EngineState.STOPPED or self.engine.has_stream:
                raise EngineError("Listening requires stopped audio and released devices")
            try:
                from ..audio.sounddevice_backend import SoundDeviceBackend
                self._start_virtual_mic(config.output_device)
                self.live_control = control
                if processor_factory is None:
                    backend = SoundDeviceBackend(config, output_confirmed=True, control=control)
                else:
                    backend = SoundDeviceBackend(
                        config, output_confirmed=True, control=control,
                        processor_factory=processor_factory,
                    )
                self.engine = AudioEngine(backend)
                self.capture_backend = None
                self.playback_backend = None
                self.monitoring = True
                self.playing = None
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                self._stop_virtual_mic()
                raise _device_error("listen", error) from error
            try:
                self.engine.start()
            except EngineError:
                self._stop_virtual_mic()
                raise
            self._meter_source_id += 1
            if processor_factory is None:
                self._remember(config)
        self._execute(start, EngineState.STARTING)

    def _stop_virtual_mic(self) -> None:
        session = self.virtual_mic
        self.virtual_mic = None
        if session is not None:
            session.stop()

    def _start_virtual_mic(self, output: "DeviceRef") -> None:
        """Create the PipeWire nodes this output needs, or fail visibly."""
        needed = output.name == virtual_mic.PCM_NAME
        if not needed:
            self._stop_virtual_mic()
            return
        capabilities = virtual_mic.capabilities()
        if not capabilities.available:
            raise RuntimeError(
                f"Virtual microphone is unavailable: {capabilities.reason}"
            )
        if self.virtual_mic is None:
            session = virtual_mic.VirtualMicSession()
            session.start()
            self.virtual_mic = session

    def _emit_media(self) -> None:
        self.mediaStatus.emit(self._media_snapshot())

    def _media_snapshot(self) -> MediaStatus:
        stream = self.capture_backend.stream if self.capture_backend is not None else None
        writer = stream.writer if stream is not None else None
        recording = writer is not None and self.engine.state is EngineState.RUNNING
        group = self.group
        frames = writer.frames_written if writer is not None else (group.frames if group is not None else 0)
        rate = stream.config.samplerate if writer is not None else (group.samplerate if group is not None else 1)
        playback = self.playback_backend.stream if self.playback_backend is not None else None
        playback_sample = playback.output_peak_sample if playback is not None else None
        analysis = (
            getattr(playback, "analysis", None) if playback is not None
            else self.live_control.analysis if self.live_control is not None else None
        )
        return MediaStatus(
            monitoring=(
                (stream.monitoring if stream is not None else self.monitoring)
                and self.engine.state is EngineState.RUNNING
            ),
            recording=recording,
            playback=self.playing if self.engine.state is EngineState.RUNNING else None,
            recording_state=self.recording_state, group=group, duration=frames / rate,
            free_bytes=writer.free_bytes if writer is not None else None,
            output_peak=playback_sample[1] if playback_sample is not None else None,
            output_peak_sequence=playback_sample[0] if playback_sample is not None else 0,
            meter_source_id=self._meter_source_id,
            analysis=analysis.snapshot if analysis is not None else None,
            recoverable=self.recoverable,
        )

    def _media_action(
        self, operation: str, action: Callable[[], None], transition: EngineState,
    ) -> None:
        def guarded() -> None:
            try:
                action()
            except EngineError:
                raise
            except Exception as error:
                raise _device_error(operation, error) from error
        self._execute(guarded, transition)

    def _capture_stream(self) -> "CaptureStream":
        if self.capture_backend is None or self.capture_backend.stream is None:
            raise RuntimeError("There is no running input-owned session")
        return self.capture_backend.stream

    @Slot(object, object, object)
    def record(
        self, config: CaptureConfig, control: "LiveControl", monitoring: OutputConfig | None,
    ) -> None:
        def start_recording() -> None:
            from ..audio.recording import RecordingWriter
            from ..audio.media_backend import CaptureBackend

            if self.group is not None and not self.group.saved:
                raise RuntimeError("Save or explicitly discard the current recording first")
            writer = RecordingWriter(
                config.samplerate, config.channels, config.blocksize,
                alignment={
                    "method": "paired pre-gain input and DSP output FIFO; native startup delay trimmed",
                    "boundaries": "DSP acknowledges next captured source boundary; finalization drains that interval",
                    "parameter_events": "DSP snapshots, not measured audible response times",
                },
            )
            self.group = writer.group
            self.recording_state = RecordingState.RECORDING
            try:
                if self.capture_backend is None:
                    self.engine.stop()
                    if monitoring is not None:
                        self._start_virtual_mic(monitoring.output_device)
                    elif self.virtual_mic is not None:
                        self._stop_virtual_mic()
                    control.levels = None
                    self.live_control = control
                    self.capture_backend = CaptureBackend(
                        config, control, monitoring=monitoring,
                        output_confirmed=monitoring is not None,
                    )
                    self.playback_backend = None
                    self.playing = None
                    self.engine = AudioEngine(self.capture_backend)
                    try:
                        self.engine.start()
                    except EngineError:
                        self._stop_virtual_mic()
                        raise
                    self._meter_source_id += 1
                stream = self._capture_stream()
                if stream.config != config:
                    raise RuntimeError("Recording settings differ from the active input session")
                stream.start_recording(writer)
                self._remember(config)
                if monitoring is not None:
                    self._remember(monitoring)
            except Exception as error:
                self.recording_state = RecordingState.INCOMPLETE
                failures = [OperationFailure("start recording", error)]
                stream = self.capture_backend.stream if self.capture_backend is not None else None
                if stream is None or stream.writer is not writer:
                    try:
                        writer.finish(complete=False, reason=f"Recording startup failed: {error}")
                    except Exception as cleanup:
                        failures.append(OperationFailure("finalize unstarted recording", cleanup))
                if self.engine.state is EngineState.RUNNING:
                    try:
                        self.engine.handle_fault(error)
                    except EngineError as cleanup:
                        failures.append(OperationFailure("stop capture after startup failure", cleanup))
                raise EngineError("Recording startup failed", tuple(failures)) from error
        self._media_action(
            "start recording", start_recording,
            EngineState.RUNNING if self.engine.state is EngineState.RUNNING else EngineState.STARTING,
        )

    def _stop_recording_impl(self) -> None:
        self.recording_state = RecordingState.FINALIZING
        self._emit_media()
        stream = self._capture_stream()
        self.group = stream.stop_recording()
        self.recording_state = (
            RecordingState.UNSAVED if self.group.complete else RecordingState.INCOMPLETE
        )
        if not stream.monitoring:
            self.engine.stop()
            self.capture_backend = None
            self.live_control = None
            self.monitoring = False
            self._stop_virtual_mic()

    @Slot()
    def stop_recording(self) -> None:
        def finish() -> None:
            try:
                self._stop_recording_impl()
            except Exception as error:
                self.recording_state = RecordingState.INCOMPLETE
                if self.engine.state is EngineState.RUNNING:
                    self.engine.handle_fault(error)
                raise
        self._media_action("finalize recording", finish, EngineState.RUNNING)

    def _stop_monitor(self) -> None:
        self._capture_stream().disable_monitor()
        self.monitoring = False
        self._stop_virtual_mic()

    @Slot(object)
    def monitor(self, config: OutputConfig) -> None:
        def enable() -> None:
            self._start_virtual_mic(config.output_device)
            try:
                self._capture_stream().enable_monitor(config, output_confirmed=True)
            except Exception:
                if not self.capture_backend or not self.capture_backend.stream.monitoring:
                    self._stop_virtual_mic()
                raise
            self.monitoring = True
            self._remember(config)
        self._media_action("start monitoring", enable, EngineState.RUNNING)

    def _stop_all_impl(self) -> None:
        failures = []
        if (
            self.capture_backend is not None and self.capture_backend.stream is not None
            and self.capture_backend.stream.writer is not None
        ):
            try:
                self._stop_recording_impl()
            except Exception as error:
                self.recording_state = RecordingState.INCOMPLETE
                failures.append(OperationFailure("finalize recording", error))
        try:
            self.engine.stop()
        except EngineError as error:
            failures.append(OperationFailure("stop audio", error))
        if not self.engine.has_stream:
            self.capture_backend = None
            self.playback_backend = None
            self.live_control = None
            self.monitoring = False
            self.playing = None
            self._stop_virtual_mic()
        if failures:
            raise EngineError("Media shutdown failed", tuple(failures))

    @Slot()
    def stop_all(self) -> None:
        self._media_action("stop all audio", self._stop_all_impl, EngineState.STOPPING)

    @Slot(object, str)
    def play(self, config: OutputConfig, track: Track) -> None:
        def start_playback() -> None:
            from ..audio.media_backend import PlaybackBackend

            if self.group is None or self.recording_state in (
                RecordingState.RECORDING, RecordingState.FINALIZING, RecordingState.SAVING,
            ):
                raise RuntimeError("Stop recording before playback")
            if track not in ("original", "effect"):
                raise ValueError("Choose the original or effect track")
            self._stop_all_impl()
            path = self.group.original_path if track == "original" else self.group.effect_path
            self.playback_backend = PlaybackBackend(
                config, path, self.group.frames, output_confirmed=True,
            )
            self.engine = AudioEngine(self.playback_backend)
            self.playing = track
            self.engine.start()
            self._meter_source_id += 1
            self._remember(config)
        self._media_action("play recording", start_playback, EngineState.STARTING)

    @Slot(object, str)
    def play_generated(self, config: OutputConfig, filename: str) -> None:
        def start_playback() -> None:
            import wave
            from ..audio.media_backend import PlaybackBackend

            if self.engine.state is not EngineState.STOPPED or self.engine.has_stream:
                raise RuntimeError("Stop other audio before playing generated speech")
            path = Path(filename)
            if path.is_symlink() or not path.is_file():
                raise ValueError("Generated audio must be a regular WAV file")
            with wave.open(str(path), "rb") as reader:
                if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (
                    config.samplerate, config.channels, 2,
                ):
                    raise ValueError("Generated WAV format changed")
                frames = reader.getnframes()
            self.playback_backend = PlaybackBackend(config, path, frames, output_confirmed=True)
            self.engine = AudioEngine(self.playback_backend)
            self.playing = "generated"
            self.engine.start()
            self._meter_source_id += 1
            self._remember(config)
        self._media_action("play generated speech", start_playback, EngineState.STARTING)

    @Slot(str, str)
    def save_recording(self, parent: str, name: str) -> None:
        def save() -> None:
            from ..audio.recording import RecordingSaveError
            if self.group is None or self.recording_state in (
                RecordingState.RECORDING, RecordingState.FINALIZING,
            ):
                raise RuntimeError("Finalize the current recording before saving")
            self.recording_state = RecordingState.SAVING
            self._emit_media()
            try:
                self.group = self.group.save(Path(parent), name)
            except RecordingSaveError as error:
                if error.saved_group is not None:
                    self.group = error.saved_group
                    self.recording_state = RecordingState.SAVED
                    self._remember_saved_directory()
                else:
                    self.recording_state = RecordingState.SAVE_FAILED
                raise
            except (OSError, RuntimeError, ValueError):
                self.recording_state = RecordingState.SAVE_FAILED
                raise
            self.recording_state = RecordingState.SAVED
            self._remember_saved_directory()
        self._media_action("save recording group", save, self.engine.state)

    @Slot()
    def discard_recording(self) -> None:
        def discard() -> None:
            if self.engine.state is not EngineState.STOPPED or self.engine.has_stream:
                raise RuntimeError("Stop audio before discarding a recording")
            if self.group is None:
                raise RuntimeError("There is no recording to discard")
            if not self.group.saved:
                self.group.discard()
            self.group = None
            self.recording_state = RecordingState.NONE
        self._media_action("discard recording", discard, self.engine.state)

    @Slot(str)
    def recover_recording(self, path: str) -> None:
        def recover() -> None:
            from ..audio.recording import RecordingGroup, recoverable_groups

            if self.engine.state is not EngineState.STOPPED or self.engine.has_stream:
                raise RuntimeError("Stop audio before recovery")
            if self.group is not None and not self.group.saved:
                raise RuntimeError("Save or discard the current recording before recovery")
            if path not in self.recoverable:
                raise RuntimeError("The selected recoverable recording is no longer listed")
            self.group = RecordingGroup.recover(Path(path))
            self.recording_state = RecordingState.UNSAVED if self.group.complete else RecordingState.INCOMPLETE
            self.recoverable = tuple(
                str(item) for item in recoverable_groups() if item != self.group.directory
            )
        self._media_action("recover recording", recover, self.engine.state)

    @Slot(str)
    def delete_recoverable(self, path: str) -> None:
        def delete() -> None:
            from ..audio.recording import discard_recoverable, recoverable_groups

            if self.engine.state is not EngineState.STOPPED or self.engine.has_stream:
                raise RuntimeError("Release devices before deleting leftover recordings")
            if path not in self.recoverable or self.group is not None and Path(path) == self.group.directory:
                raise RuntimeError("Choose a listed leftover, not the current recording")
            discard_recoverable(Path(path))
            self.recoverable = tuple(str(item) for item in recoverable_groups())
        self._media_action("delete leftover recording", delete, self.engine.state)

    @Slot()
    def discover(self) -> None:
        try:
            import sounddevice as sd
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            self.errorOccurred.emit(_device_error("discover", error))
        else:
            from ..audio.device_discovery import discover_devices
            try:
                devices = discover_devices()
            except (sd.PortAudioError, OSError, RuntimeError, ValueError) as error:
                self.errorOccurred.emit(_device_error("discover", error))
            else:
                self.devices = devices
                self.devicesDiscovered.emit(devices)
        self.operationFinished.emit()

    @Slot(object)
    def prepare_backend(self, config: StreamConfig) -> None:
        try:
            if self.engine.state is not EngineState.STOPPED or self.engine.has_stream:
                raise EngineError("Backend preparation requires stopped audio and released devices")
            self.timer.stop()
            self.live_control = None
            self.engine = AudioEngine()
            from ..audio.sounddevice_backend import SoundDeviceBackend
            self.engine = AudioEngine(SoundDeviceBackend(config, output_confirmed=False))
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            self.errorOccurred.emit(_device_error("prepare", error))
        else:
            self.backendPrepared.emit(config)
        self.resourcesChanged.emit(self.engine.has_stream)
        self.stateChanged.emit(self.engine.state)
        self.operationFinished.emit()

    @Slot()
    def shutdown(self) -> None:
        self.timer.stop()
        self.stateChanged.emit(EngineState.STOPPING)
        try:
            self._stop_all_impl()
            self._emit_media()
            if self.group is not None and not self.group.saved:
                raise EngineError("Save or explicitly discard the recording before exiting")
            preference_error = self._flush_preferences()
            if preference_error is not None:
                raise preference_error
        except EngineError as error:
            self.errorOccurred.emit(error)
            self.resourcesChanged.emit(self.engine.has_stream)
            self.stateChanged.emit(self.engine.state)
            self.shutdownFailed.emit()
            return
        finally:
            if not self.engine.has_stream:
                self._stop_virtual_mic()
        self.resourcesChanged.emit(self.engine.has_stream)
        self.stateChanged.emit(self.engine.state)
        self.timer.timeout.disconnect(self.poll)
        self.shutdownSucceeded.emit()


class EngineController(QObject):
    """Use only on its owning GUI thread and retain until shutdownComplete.

    Backend operations run on one worker. Failed cleanup keeps that worker alive
    for an explicit shutdown retry; destroying it early is not safe. After
    shutdownComplete, dispose with deleteLater while the owner event loop lives.
    """

    ready = Signal()
    stateChanged = Signal(object)
    errorOccurred = Signal(object)
    operationFinished = Signal()
    shutdownFailed = Signal()
    shutdownComplete = Signal()
    deviceSelectionChanged = Signal()
    streamConfigReady = Signal(object)
    backendPrepared = Signal(object)
    liveStatus = Signal(object)
    mediaStatus = Signal(object)
    preferencesChanged = Signal(object)
    saveDirectoryReady = Signal(str)
    _saveDirectoryRequested = Signal()
    _startRequested = Signal()
    _stopRequested = Signal()
    _shutdownRequested = Signal()
    _devicesRequested = Signal()
    _prepareRequested = Signal(object)
    _listenRequested = Signal(object, object, object)
    _recordRequested = Signal(object, object, object)
    _recordStopRequested = Signal()
    _monitorRequested = Signal(object)
    _stopAllRequested = Signal()
    _playRequested = Signal(object, str)
    _playGeneratedRequested = Signal(object, str)
    _saveRecordingRequested = Signal(str, str)
    _discardRecordingRequested = Signal()
    _recoverRecordingRequested = Signal(str)
    _deleteRecoveryRequested = Signal(str)

    def __init__(
        self, backend: AudioBackend | None = None, parent: QObject | None = None,
        *, preferences_file: Path | None = None,
    ) -> None:
        super().__init__(parent)
        application = QCoreApplication.instance()
        if application is None or QThread.currentThread() != application.thread():
            raise RuntimeError("EngineController requires an existing Qt application GUI thread")
        self._state = EngineState.STOPPED
        self._ready = False
        self._busy = True
        self._closing = False
        self._closed = False
        self._owns_devices = False
        self._devices_loaded = False
        self._selection: DeviceSelection | None = None
        self._live_control: LiveControl | None = None
        self._capture_config: CaptureConfig | None = None
        self._media_status = MediaStatus()
        self.preferences_enabled = preferences_file is not None
        self.preferences = Preferences()
        self._restore_preferences_pending = self.preferences_enabled
        self._thread: QThread | None = QThread()
        self._polled_status = _LatestPolledStatus()
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(50)
        self._status_timer.timeout.connect(self._receive_polled_status)
        self._worker = _EngineWorker(backend, self._polled_status, preferences_file)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.initialize)
        self._startRequested.connect(self._worker.start)
        self._stopRequested.connect(self._worker.stop)
        self._shutdownRequested.connect(self._worker.shutdown)
        self._devicesRequested.connect(self._worker.discover)
        self._prepareRequested.connect(self._worker.prepare_backend)
        self._listenRequested.connect(self._worker.listen)
        self._recordRequested.connect(self._worker.record)
        self._recordStopRequested.connect(self._worker.stop_recording)
        self._monitorRequested.connect(self._worker.monitor)
        self._stopAllRequested.connect(self._worker.stop_all)
        self._playRequested.connect(self._worker.play)
        self._playGeneratedRequested.connect(self._worker.play_generated)
        self._saveRecordingRequested.connect(self._worker.save_recording)
        self._discardRecordingRequested.connect(self._worker.discard_recording)
        self._recoverRecordingRequested.connect(self._worker.recover_recording)
        self._deleteRecoveryRequested.connect(self._worker.delete_recoverable)
        self._saveDirectoryRequested.connect(self._worker.check_save_directory)
        self._worker.preferencesChanged.connect(self._on_preferences)
        self._worker.saveDirectoryReady.connect(self._on_save_directory_ready)
        self._worker.initialized.connect(self._on_ready)
        self._worker.stateChanged.connect(self._on_state)
        self._worker.errorOccurred.connect(self._on_error)
        self._worker.operationFinished.connect(self._on_operation_finished)
        self._worker.shutdownFailed.connect(self._on_shutdown_failed)
        self._worker.resourcesChanged.connect(self._on_resources)
        self._worker.devicesDiscovered.connect(self._on_devices)
        self._worker.backendPrepared.connect(self._on_backend_prepared)
        self._worker.mediaStatus.connect(self._on_media_status)
        self._worker.shutdownSucceeded.connect(self._on_shutdown_succeeded)
        self._worker.destroyed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.destroyed.connect(self._on_thread_destroyed)
        self._thread.start()

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._ready and not self._closing and not self._closed

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def closing(self) -> bool:
        return self._closing

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def owns_devices(self) -> bool:
        """Last worker-reported stream ownership; busy protects pending transitions."""
        return self._owns_devices

    @property
    def media_status(self) -> MediaStatus:
        return self._media_status

    @property
    def devices_loaded(self) -> bool:
        return self._devices_loaded

    @property
    def device_selection(self) -> "DeviceSelection":
        self._require_owner()
        if self._selection is None:
            from .device_selection import DeviceSelection
            self._selection = DeviceSelection()
            self._selection.set_activity(busy=self._busy, owns_devices=self._owns_devices)
        return self._selection

    def _sync_selection(self) -> None:
        if self._selection is not None:
            self._selection.set_activity(busy=self._busy, owns_devices=self._owns_devices)
            self.deviceSelectionChanged.emit()

    def _require_owner(self) -> None:
        if QThread.currentThread() != self.thread():
            raise RuntimeError("EngineController requests must use its owning thread")

    def _reject(self, message: str) -> bool:
        self.errorOccurred.emit(EngineError(message))
        return False

    def _can_request(self) -> bool:
        self._require_owner()
        if not self.is_ready or self._busy:
            return self._reject("Controller is busy, initializing, closing or closed")
        return True

    @Slot(result=bool)
    def request_start(self) -> bool:
        if not self._can_request():
            return False
        if self._state is not EngineState.STOPPED:
            return self._reject(f"Cannot start while engine is {self._state.value}")
        self._busy = True
        self._startRequested.emit()
        self._sync_selection()
        return True

    @Slot(result=bool)
    def request_stop(self) -> bool:
        if not self._can_request():
            return False
        self._busy = True
        self._stopRequested.emit()
        self._sync_selection()
        return True

    @Slot(result=bool)
    def request_shutdown(self) -> bool:
        self._require_owner()
        if self._closed:
            return True
        if self._closing:
            return self._reject("Controller shutdown is already pending")
        self._closing = True
        self._busy = True
        self._status_timer.stop()
        self._polled_status.take()
        self._shutdownRequested.emit()
        self._sync_selection()
        return True

    def select_device(self, kind: "DeviceKind", ref: "DeviceRef | None") -> bool:
        if not self._can_request():
            return False
        from .device_selection import DeviceSelectionError
        try:
            self.device_selection.select(kind, ref)
        except (ValueError, DeviceSelectionError) as error:
            self.errorOccurred.emit(_device_error("select", error))
            return False
        self.deviceSelectionChanged.emit()
        return True

    def request_stream_config(
        self, *, samplerate: int, blocksize: int, channels: int, output_ceiling: float,
    ) -> bool:
        if not self._can_request():
            return False
        from .device_selection import DeviceSelectionError
        try:
            config = self.device_selection.build_stream_config(
                samplerate=samplerate, blocksize=blocksize, channels=channels,
                output_ceiling=output_ceiling,
            )
        except (ValueError, DeviceSelectionError) as error:
            self.errorOccurred.emit(_device_error("configure", error))
            return False
        self.streamConfigReady.emit(config)
        return True

    def request_device_refresh(self) -> bool:
        if not self._can_request():
            return False
        if self.device_selection.locked:
            return self._reject("Device refresh requires released devices")
        self._busy = True
        self._devicesRequested.emit()
        self._sync_selection()
        return True

    def request_save_directory(self) -> bool:
        if not self._can_request():
            return False
        self._busy = True
        self._saveDirectoryRequested.emit()
        self._sync_selection()
        return True

    @Slot(str)
    def _on_save_directory_ready(self, directory: str) -> None:
        if not self._closing:
            self._busy = False
            self._sync_selection()
            self.saveDirectoryReady.emit(directory)

    @Slot(object)
    def _on_preferences(self, preferences: Preferences) -> None:
        self.preferences = preferences
        self.preferencesChanged.emit(preferences)

    def request_backend_prepare(self, config: StreamConfig) -> bool:
        if not self._can_request():
            return False
        if self._state is not EngineState.STOPPED or self._owns_devices:
            return self._reject("Backend preparation requires stopped audio and released devices")
        from .device_selection import DeviceSelectionError
        try:
            if not isinstance(config, StreamConfig):
                raise ValueError("Backend preparation requires an explicit StreamConfig")
            current = self.device_selection.build_stream_config(
                samplerate=config.samplerate, blocksize=config.blocksize,
                channels=config.channels, output_ceiling=config.output_ceiling,
            )
            if current != config:
                raise DeviceSelectionError("Configuration changed; rebuild before preparing backend")
        except (ValueError, DeviceSelectionError) as error:
            self.errorOccurred.emit(_device_error("prepare", error))
            return False
        self._busy = True
        self._prepareRequested.emit(current)
        self._sync_selection()
        return True

    def request_listen(
        self, config: StreamConfig, parameters: "EffectParameters", *,
        output_confirmed: bool = False,
        processor_factory: "ProcessorFactory | None" = None,
    ) -> bool:
        if not self._can_request():
            return False
        if output_confirmed is not True:
            return self._reject("Confirm the selected output and safe headphone volume before listening")
        if self._state is not EngineState.STOPPED or self._owns_devices:
            return self._reject("Listening requires stopped audio and released devices")
        from .device_selection import DeviceSelectionError
        from ..dsp.parameters import LiveControl
        try:
            if not isinstance(config, StreamConfig):
                raise ValueError("Listening requires an explicit StreamConfig")
            current = self.device_selection.build_stream_config(
                samplerate=config.samplerate, blocksize=config.blocksize,
                channels=config.channels, output_ceiling=config.output_ceiling,
            )
            if current != config:
                raise DeviceSelectionError("Configuration changed; validate it again")
            control = LiveControl(parameters)
        except (ValueError, DeviceSelectionError) as error:
            self.errorOccurred.emit(_device_error("listen", error))
            return False
        self._live_control = control
        self._capture_config = CaptureConfig(
            config.input_device, config.samplerate, config.blocksize,
            config.channels, config.output_ceiling,
        )
        self._busy = True
        self._listenRequested.emit(current, control, processor_factory)
        self._sync_selection()
        return True

    def request_record(
        self, parameters: "EffectParameters", *, samplerate: int,
        blocksize: int, channels: int, output_ceiling: float,
    ) -> bool:
        if not self._can_request():
            return False
        status = self._media_status
        if self._state not in (EngineState.STOPPED, EngineState.RUNNING) or status.playback is not None:
            return self._reject("Stop playback or reset the fault before recording")
        if status.recording or (status.group is not None and not status.group.saved):
            return self._reject("Stop, save or explicitly discard the current recording first")
        from .device_selection import DeviceSelectionError
        from ..dsp.parameters import LiveControl
        try:
            config = self.device_selection.build_capture_config(
                samplerate=samplerate, blocksize=blocksize,
                channels=channels, output_ceiling=output_ceiling,
            )
            if self._state is EngineState.RUNNING and self._capture_config != config:
                raise ValueError("Recording settings must match the running input session")
            control = self._live_control if self._live_control is not None else LiveControl(parameters)
            if self._live_control is not None:
                control.submit(parameters)
            output = (
                self.device_selection.build_output_config(
                    samplerate=samplerate, blocksize=blocksize,
                    channels=channels, output_ceiling=output_ceiling,
                ) if status.monitoring else None
            )
        except (ValueError, DeviceSelectionError) as error:
            self.errorOccurred.emit(_device_error("record", error))
            return False
        self._capture_config = config
        self._live_control = control
        self._busy = True
        self._recordRequested.emit(config, control, output)
        self._sync_selection()
        return True

    def request_stop_recording(self) -> bool:
        if not self._can_request():
            return False
        if not self._media_status.recording:
            return self._reject("There is no active recording")
        self._busy = True
        self._recordStopRequested.emit()
        self._sync_selection()
        return True

    def request_monitor(self, *, output_confirmed: bool = False) -> bool:
        if not self._can_request():
            return False
        if output_confirmed is not True:
            return self._reject("Confirm safe output before monitoring")
        if not self._media_status.recording or self._media_status.monitoring or self._capture_config is None:
            return self._reject("Monitoring can only be added to a running recording")
        from .device_selection import DeviceSelectionError
        config = self._capture_config
        try:
            output = self.device_selection.build_output_config(
                samplerate=config.samplerate, blocksize=config.blocksize,
                channels=config.channels, output_ceiling=config.output_ceiling,
            )
        except (ValueError, DeviceSelectionError) as error:
            self.errorOccurred.emit(_device_error("monitor", error))
            return False
        self._busy = True
        self._monitorRequested.emit(output)
        self._sync_selection()
        return True

    def request_stop_all(self) -> bool:
        if not self._can_request():
            return False
        self._busy = True
        self._stopAllRequested.emit()
        self._sync_selection()
        return True

    def request_play(
        self, track: Track, *, blocksize: int, output_ceiling: float,
        output_confirmed: bool = False,
    ) -> bool:
        if not self._can_request():
            return False
        if output_confirmed is not True:
            return self._reject("Confirm safe output before playback")
        status = self._media_status
        if (
            status.recording or status.group is None
            or self._state not in (EngineState.STOPPED, EngineState.RUNNING)
            or track not in ("original", "effect")
        ):
            return self._reject("Choose a finalized recording and stop recording before playback")
        from .device_selection import DeviceSelectionError
        try:
            config = self.device_selection.build_output_config(
                samplerate=status.group.samplerate, blocksize=blocksize,
                channels=status.group.channels, output_ceiling=output_ceiling,
            )
        except (ValueError, DeviceSelectionError) as error:
            self.errorOccurred.emit(_device_error("play", error))
            return False
        self._live_control = None
        self._busy = True
        self._playRequested.emit(config, track)
        self._sync_selection()
        return True

    def request_play_generated(
        self, path: Path, sample_rate: int, *, output_ceiling: float,
        output_confirmed: bool = False,
    ) -> bool:
        if not self._can_request():
            return False
        if not output_confirmed:
            return self._reject("Confirm safe output before playback")
        if self._state is not EngineState.STOPPED or self.owns_devices or self.media_status.recording:
            return self._reject("Stop other audio before playing generated speech")
        from .device_selection import DeviceSelectionError
        try:
            config = self.device_selection.build_output_config(
                samplerate=sample_rate, blocksize=1024, channels=1,
                output_ceiling=output_ceiling,
            )
        except (ValueError, DeviceSelectionError) as error:
            self.errorOccurred.emit(_device_error("play generated speech", error))
            return False
        self._live_control = None
        self._busy = True
        self._playGeneratedRequested.emit(config, str(path))
        self._sync_selection()
        return True

    def request_save_recording(self, parent: str, name: str) -> bool:
        if not self._can_request():
            return False
        status = self._media_status
        if status.group is None or status.recording or status.playback is not None:
            return self._reject("Stop recording and playback before saving")
        if status.group.saved:
            return self._reject("This recording is already saved")
        self._busy = True
        self._saveRecordingRequested.emit(parent, name)
        self._sync_selection()
        return True

    def request_discard_recording(self) -> bool:
        if not self._can_request():
            return False
        if self._state is not EngineState.STOPPED or self._owns_devices or self._media_status.group is None:
            return self._reject("Stop audio before discarding the current recording")
        self._busy = True
        self._discardRecordingRequested.emit()
        self._sync_selection()
        return True

    def request_recover_recording(self, path: str) -> bool:
        if not self._can_request():
            return False
        if (
            self._state is not EngineState.STOPPED or self._owns_devices
            or path not in self._media_status.recoverable
        ):
            return self._reject("Choose a listed recording and release devices before recovery")
        self._busy = True
        self._recoverRecordingRequested.emit(path)
        self._sync_selection()
        return True

    def request_delete_recoverable(self, path: str) -> bool:
        if not self._can_request():
            return False
        if self._state is not EngineState.STOPPED or self._owns_devices or path not in self._media_status.recoverable:
            return self._reject("Choose a listed leftover and release devices before deleting")
        self._busy = True
        self._deleteRecoveryRequested.emit(path)
        self._sync_selection()
        return True

    @Slot(object)
    def _on_media_status(self, status: MediaStatus) -> None:
        if not self._closed and status.meter_source_id >= self._media_status.meter_source_id:
            self._media_status = status
            self.mediaStatus.emit(status)

    @Slot()
    def _receive_polled_status(self) -> None:
        snapshot = self._polled_status.take()
        if snapshot is None or self._closing or self._state is not EngineState.RUNNING:
            return
        media, current = snapshot.media, self._media_status
        if (
            media.meter_source_id != current.meter_source_id
            or media.monitoring != current.monitoring
            or media.recording != current.recording
            or media.playback != current.playback
            or media.recording_state != current.recording_state
            or media.group is not current.group
            or media.recoverable != current.recoverable
        ):
            return
        self._on_media_status(media)
        if snapshot.live is not None:
            self._on_live_status(snapshot.live)

    def request_parameters(self, parameters: "EffectParameters") -> bool:
        self._require_owner()
        if not self.is_ready or self._busy or self._closing:
            return self._reject("Effect parameters cannot change while closing or initializing")
        if self._state is not EngineState.RUNNING or self._live_control is None:
            return self._reject("Start a live listening session before submitting parameters")
        try:
            self._live_control.submit(parameters)
        except ValueError as error:
            self.errorOccurred.emit(_device_error("parameters", error))
            return False
        return True

    @Slot(object)
    def _on_live_status(self, status: tuple[tuple[float, float] | None, int, int, int]) -> None:
        if not self._closing and self._state is EngineState.RUNNING and self._live_control is not None:
            levels, applied, _, source_id = status
            if source_id == self._media_status.meter_source_id and self._media_status.playback is None:
                self.liveStatus.emit((levels, applied, self._live_control.requested[0]))

    @Slot(object)
    def _on_backend_prepared(self, config: StreamConfig) -> None:
        if not self._closing and not self._closed:
            self.backendPrepared.emit(config)

    @Slot(bool)
    def _on_resources(self, owns_devices: bool) -> None:
        self._owns_devices = owns_devices
        self._sync_selection()

    @Slot(object)
    def _on_devices(self, devices: "tuple[DeviceInfo, ...]") -> None:
        if self._closing:
            return
        from .device_selection import DeviceSelectionError, SelectionInvalidatedError
        selection = self.device_selection
        # Query completion is still busy, but this committed worker result is idle audio.
        selection.set_activity(busy=False, owns_devices=self._owns_devices)
        failure = None
        try:
            selection.refresh(devices)
        except SelectionInvalidatedError as error:
            self._devices_loaded = True
            failure = _device_error("refresh", error)
        except (ValueError, DeviceSelectionError) as error:
            failure = _device_error("refresh", error)
        else:
            self._devices_loaded = True
            if self._restore_preferences_pending:
                self._restore_preferences_pending = False
                channels = self.preferences.audio.channels if self.preferences.audio is not None else 1
                for kind in ("input", "output"):
                    identity = (
                        self.preferences.input_device if kind == "input"
                        else self.preferences.output_device
                    )
                    if identity is not None:
                        try:
                            selection.select(kind, identity.match(devices, kind, channels))
                        except ValueError as error:
                            self.errorOccurred.emit(_device_error("restore device", error))
        finally:
            self._sync_selection()
        if failure is not None:
            self.errorOccurred.emit(failure)

    @Slot()
    def _on_ready(self) -> None:
        self._ready = True
        if not self._closing:
            self._busy = False
            self._sync_selection()
            self.ready.emit()

    @Slot(object)
    def _on_state(self, state: EngineState) -> None:
        self._state = state
        if state is EngineState.RUNNING and not self._closing:
            if not self._status_timer.isActive():
                self._status_timer.start()
        else:
            self._status_timer.stop()
            self._polled_status.take()
        if state is EngineState.STOPPED:
            self._live_control = None
            self._capture_config = None
        self.stateChanged.emit(state)

    @Slot(object)
    def _on_error(self, error: EngineError) -> None:
        self.errorOccurred.emit(error)

    @Slot()
    def _on_operation_finished(self) -> None:
        if not self._closing:
            self._busy = False
        self._sync_selection()
        self.operationFinished.emit()

    @Slot()
    def _on_shutdown_failed(self) -> None:
        self._closing = False
        self._busy = False
        self._sync_selection()
        self.shutdownFailed.emit()

    @Slot()
    def _on_shutdown_succeeded(self) -> None:
        # Detach Python-wrapped senders before QObject destruction takes Qt locks.
        assert self._thread is not None
        self._thread.started.disconnect(self._worker.initialize)
        self._startRequested.disconnect(self._worker.start)
        self._stopRequested.disconnect(self._worker.stop)
        self._shutdownRequested.disconnect(self._worker.shutdown)
        self._devicesRequested.disconnect(self._worker.discover)
        self._prepareRequested.disconnect(self._worker.prepare_backend)
        self._listenRequested.disconnect(self._worker.listen)
        self._recordRequested.disconnect(self._worker.record)
        self._recordStopRequested.disconnect(self._worker.stop_recording)
        self._monitorRequested.disconnect(self._worker.monitor)
        self._stopAllRequested.disconnect(self._worker.stop_all)
        self._playRequested.disconnect(self._worker.play)
        self._saveRecordingRequested.disconnect(self._worker.save_recording)
        self._discardRecordingRequested.disconnect(self._worker.discard_recording)
        self._recoverRecordingRequested.disconnect(self._worker.recover_recording)
        self._deleteRecoveryRequested.disconnect(self._worker.delete_recoverable)
        self._saveDirectoryRequested.disconnect(self._worker.check_save_directory)
        self._worker.deleteLater()

    @Slot()
    def _on_thread_destroyed(self) -> None:
        self._thread = None
        self._ready = False
        self._busy = False
        self._closing = False
        self._closed = True
        self._owns_devices = False
        self._sync_selection()
        self.shutdownComplete.emit()
