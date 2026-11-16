"""Synthetic PortAudio substitutes; no actual devices or user audio."""

import threading

import numpy as np

from device_fakes import FakeDeviceLibrary, FakePortAudioError


class CallbackAbort(Exception):
    pass


class CallbackStop(Exception):
    pass


class MediaRawStream:
    def __init__(self, library, kind, options):
        self.library, self.kind, self.options = library, kind, options
        self.device = options["device"]
        self.samplerate = options["samplerate"]
        self.blocksize = options["blocksize"]
        self.channels = options["channels"]
        self.dtype = ("float32", "float32") if kind == "duplex" else "float32"
        self.active = False
        self.closed = False
        self.close_error = None
        self._stop = threading.Event()
        self._thread = None
        self.cursor = 0
        self.output_peak = 0.0
        self.output_nonfinite = False

    def start(self):
        self.library.operations.append(("start", self.kind, threading.get_ident()))
        self.active = True
        if self.library.auto_callbacks:
            self._thread = threading.Thread(target=self._pump, name="synthetic-media-device")
            self._thread.start()

    def tick(self):
        channels = self.channels[0] if self.kind == "duplex" else self.channels
        phase = (np.arange(self.blocksize) + self.cursor) / self.samplerate
        samples = np.repeat(
            (self.library.amplitude * np.sin(2 * np.pi * 220 * phase))[:, None],
            channels, axis=1,
        ).astype(np.float32)
        output = np.empty((self.blocksize, channels), dtype=np.float32)
        try:
            if self.kind == "input":
                self.options["callback"](samples, self.blocksize, None, False)
            elif self.kind == "output":
                self.options["callback"](output, self.blocksize, None, False)
            else:
                self.options["callback"](samples, output, self.blocksize, None, False)
        except (CallbackAbort, CallbackStop):
            self.active = False
            self._stop.set()
        if self.kind != "input":
            if not np.isfinite(output).all():
                self.output_nonfinite = True
            else:
                self.output_peak = max(self.output_peak, float(np.abs(output).max()))
        self.cursor += self.blocksize
        return output

    def _pump(self):
        while not self._stop.is_set():
            self.tick()
            self._stop.wait(self.blocksize / self.samplerate)

    def stop(self):
        self.library.operations.append(("stop", self.kind, threading.get_ident()))
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("Synthetic callback failed to stop")
        self.active = False

    def close(self):
        self.library.operations.append(("close", self.kind, threading.get_ident()))
        if self.close_error is not None:
            raise self.close_error
        self.stop()
        self.closed = True


class MediaLibrary(FakeDeviceLibrary):
    CallbackAbort = CallbackAbort
    CallbackStop = CallbackStop

    def __init__(self, *, auto_callbacks=True):
        super().__init__()
        self.auto_callbacks = auto_callbacks
        self.amplitude = 0.02
        self.streams = []
        self.operations = []
        self.output_error = None

    def query_devices(self, index=None, kind=None):
        if index is None:
            return super().query_devices()
        self.operations.append(("query", kind, threading.get_ident()))
        return self.devices[index]

    def check_input_settings(self, **options):
        self.operations.append(("settings", "input", threading.get_ident()))

    def check_output_settings(self, **options):
        self.operations.append(("settings", "output", threading.get_ident()))

    def _stream(self, kind, options):
        self.operations.append(("create", kind, threading.get_ident()))
        if kind == "output" and self.output_error is not None:
            raise FakePortAudioError(self.output_error)
        raw = MediaRawStream(self, kind, options)
        self.streams.append(raw)
        return raw

    def InputStream(self, **options):
        return self._stream("input", options)

    def OutputStream(self, **options):
        return self._stream("output", options)

    def Stream(self, **options):
        return self._stream("duplex", options)
