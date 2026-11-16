"""Shared test-only backend; no real device library or simulated app mode."""

import threading


class FakeStream:
    def __init__(self, backend):
        self.backend = backend

    def start(self):
        self.backend.record("start")
        self.backend.start_entered.set()
        if not self.backend.start_release.wait(timeout=3):
            raise RuntimeError("Test did not release start")
        if self.backend.start_error:
            raise self.backend.start_error

    def stop(self):
        self.backend.record("stop")
        if self.backend.stop_error:
            raise self.backend.stop_error

    def close(self):
        self.backend.record("close")
        if self.backend.close_error:
            raise self.backend.close_error

    def check_health(self):
        self.backend.record("health")
        if self.backend.health_fault.is_set():
            raise RuntimeError("test stream disconnected")


class FakeBackend:
    def __init__(self):
        self.events = []
        self.start_entered = threading.Event()
        self.start_release = threading.Event()
        self.start_release.set()
        self.health_fault = threading.Event()
        self.start_error = None
        self.stop_error = None
        self.close_error = None

    def record(self, action):
        self.events.append((action, threading.get_ident()))

    def open_stream(self):
        self.record("open")
        return FakeStream(self)
