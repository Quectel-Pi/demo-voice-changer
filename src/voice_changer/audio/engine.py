"""Lifecycle control for a single stream, without Qt or an audio dependency.

Call from one serialized, non-GUI control thread, never an audio callback.
Transition guards prevent reentry, not concurrent access from multiple threads.
"""

from dataclasses import dataclass
from enum import Enum
from typing import NoReturn, Protocol


class EngineState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAULT = "fault"


class AudioStream(Protocol):
    """A stream returned inactive; successful close releases all its resources.

    stop and close must tolerate a failed or partial start and cleanup retries.
    """

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def check_health(self) -> None:
        """Raise on callback faults or unexpected inactivity; control thread only."""
        ...


class AudioBackend(Protocol):
    def open_stream(self) -> AudioStream:
        """Return an inactive stream; release partial allocations before raising."""
        ...


@dataclass(frozen=True)
class OperationFailure:
    operation: str
    error: Exception


class EngineError(RuntimeError):
    def __init__(
        self, message: str, failures: tuple[OperationFailure, ...] = (),
    ) -> None:
        self.failures = failures
        details = "; ".join(
            f"{failure.operation}: {failure.error}" for failure in failures
        )
        super().__init__(f"{message}: {details}" if details else message)


class AudioEngine:
    def __init__(self, backend: AudioBackend | None = None) -> None:
        self._backend = backend
        self._stream: AudioStream | None = None
        self._state = EngineState.STOPPED
        self._last_error: EngineError | None = None

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def has_stream(self) -> bool:
        """Whether a stream handle remains owned, including failed cleanup."""
        return self._stream is not None

    @property
    def last_error(self) -> EngineError | None:
        return self._last_error

    def start(self) -> None:
        if self._state is not EngineState.STOPPED:
            raise EngineError(f"Cannot start while engine is {self._state.value}")
        if self._backend is None:
            self._raise_failure("Audio backend is not configured", [])

        self._state = EngineState.STARTING
        operation = "open"
        try:
            self._stream = self._backend.open_stream()
            if self._stream is None:
                raise RuntimeError("Audio backend returned no stream")
            operation = "start"
            self._stream.start()
        except Exception as error:
            failures = [OperationFailure(operation, error)]
            self._state = EngineState.STOPPING
            failures.extend(self._release_stream())
            self._raise_failure("Audio startup failed", failures)
        self._state = EngineState.RUNNING

    def stop(self) -> None:
        """Stop and release; a successful explicit stop also acknowledges a fault."""
        if self._state in (EngineState.STARTING, EngineState.STOPPING):
            raise EngineError(f"Cannot stop while engine is {self._state.value}")
        if self._state is EngineState.STOPPED:
            return

        self._state = EngineState.STOPPING
        failures = self._release_stream()
        if failures:
            self._raise_failure("Audio shutdown failed", failures)
        self._state = EngineState.STOPPED
        self._last_error = None

    def handle_fault(self, error: Exception) -> NoReturn:
        """Consume a runtime fault on the control thread, release, and raise."""
        if self._state is not EngineState.RUNNING:
            raise EngineError(
                f"Cannot handle a runtime fault while engine is {self._state.value}"
            )
        self._state = EngineState.STOPPING
        failures = [OperationFailure("runtime", error)]
        failures.extend(self._release_stream())
        self._raise_failure("Audio runtime failed", failures)

    def check_health(self) -> None:
        """Poll backend health outside the callback and stop on a reported fault."""
        if self._state is not EngineState.RUNNING:
            raise EngineError(f"Cannot check health while engine is {self._state.value}")
        stream = self._stream
        if stream is None:
            self.handle_fault(RuntimeError("Running engine has no stream"))
        try:
            stream.check_health()
        except Exception as error:
            self.handle_fault(error)

    def _release_stream(self) -> list[OperationFailure]:
        stream = self._stream
        if stream is None:
            return []

        failures = []
        try:
            stream.stop()
        except Exception as error:
            failures.append(OperationFailure("stop", error))
        try:
            stream.close()
        except Exception as error:
            failures.append(OperationFailure("close", error))
        else:
            self._stream = None
        return failures

    def _raise_failure(
        self, message: str, failures: list[OperationFailure],
    ) -> NoReturn:
        error = EngineError(message, tuple(failures))
        self._last_error = error
        self._state = EngineState.FAULT
        if failures:
            raise error from failures[0].error
        raise error
