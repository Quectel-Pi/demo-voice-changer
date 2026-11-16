"""Test-only read-only device library with a bounded query barrier."""

import threading
from types import ModuleType


class FakePortAudioError(Exception):
    pass


class FakeDeviceLibrary(ModuleType):
    PortAudioError = FakePortAudioError

    def __init__(self):
        super().__init__("sounddevice")
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.error = None
        self.devices = [
            dict(name="test input", hostapi=0, max_input_channels=1,
                 max_output_channels=0, default_samplerate=48000.0),
            dict(name="test output", hostapi=0, max_input_channels=0,
                 max_output_channels=2, default_samplerate=48000.0),
        ]
        self.hostapis = [dict(name="test API", devices=[0, 1])]

    def query_devices(self):
        self.calls.append(("devices", threading.get_ident()))
        self.entered.set()
        if not self.release.wait(timeout=3):
            raise RuntimeError("Test did not release device query")
        if self.error is not None:
            raise self.error
        return self.devices

    def query_hostapis(self):
        self.calls.append(("hostapis", threading.get_ident()))
        return self.hostapis

    @property
    def default(self):
        raise AssertionError("Do not select defaults")

    def Stream(self, **options):
        raise AssertionError("Device discovery must not open streams")
