"""Bounded streaming C API adapter; instantiate only on a processing/control thread."""

import ctypes
from ctypes.util import find_library
import os

import numpy as np


class RubberBand:
    def __init__(self, samplerate: int, channels: int, blocksize: int) -> None:
        library = os.environ.get("VOICE_CHANGER_RUBBERBAND_LIBRARY") or find_library("rubberband")
        if not library:
            raise RuntimeError("Rubber Band 3.3+ library is missing; install/configure it before listening")
        self._api = ctypes.CDLL(library)
        state = ctypes.c_void_p
        uint = ctypes.c_uint
        ptr = ctypes.POINTER(ctypes.c_float)
        vector = ctypes.POINTER(ptr)
        signatures = {
            "new": ([uint, uint, ctypes.c_int, ctypes.c_double, ctypes.c_double], state),
            "delete": ([state], None),
            "get_engine_version": ([state], ctypes.c_int),
            "set_pitch_scale": ([state, ctypes.c_double], None),
            "set_formant_scale": ([state, ctypes.c_double], None),
            "get_preferred_start_pad": ([state], uint),
            "get_start_delay": ([state], uint),
            "set_max_process_size": ([state, uint], None),
            "process": ([state, vector, uint, ctypes.c_int], None),
            "available": ([state], ctypes.c_int),
            "retrieve": ([state, vector, uint], uint),
        }
        try:
            for name, (args, result) in signatures.items():
                function = getattr(self._api, f"rubberband_{name}")
                function.argtypes = args
                function.restype = result
        except AttributeError as error:
            raise RuntimeError("Rubber Band lacks the required R3/pitch/formant streaming API") from error
        self.blocksize = blocksize
        self.channels = channels
        self._input = np.zeros((channels, blocksize), dtype=np.float32)
        self._output = np.empty_like(self._input)
        self._input_ptrs = (ptr * channels)(
            *(self._input[channel].ctypes.data_as(ptr) for channel in range(channels))
        )
        self._output_ptrs = (ptr * channels)(
            *(self._output[channel].ctypes.data_as(ptr) for channel in range(channels))
        )
        # Same real-time/R3/threading-never/FormantPreserved baseline as the M0 probe.
        options = 0x00000001 | 0x00010000 | 0x20000000 | 0x01000000
        self._state = self._api.rubberband_new(samplerate, channels, options, 1.0, 1.0)
        if not self._state:
            raise RuntimeError("Rubber Band could not create a streaming processor")
        try:
            if self._api.rubberband_get_engine_version(self._state) != 3:
                raise RuntimeError("Rubber Band did not select the R3 engine")
            self._api.rubberband_set_max_process_size(self._state, blocksize)
            self.start_pad = int(self._api.rubberband_get_preferred_start_pad(self._state))
            self.start_delay = int(self._api.rubberband_get_start_delay(self._state))
            if not 0 <= self.start_pad <= samplerate or not 0 <= self.start_delay <= samplerate:
                raise RuntimeError("Rubber Band reported an unsupported startup delay")
        except Exception:
            self.close()
            raise

    def set_scales(self, pitch: float, formant: float) -> None:
        """Apply a pitch ratio and a formant ratio relative to the original voice.

        R3 applies its formant scale on top of the pitch shift (its automatic
        value is 1 / pitch), so the net envelope shift is formant_scale * pitch.
        """
        self._api.rubberband_set_pitch_scale(self._state, pitch)
        self._api.rubberband_set_formant_scale(self._state, formant / pitch)

    def process(self, samples: np.ndarray) -> None:
        if samples.ndim != 2 or samples.shape[1] != self.channels or not 0 < len(samples) <= self.blocksize:
            raise ValueError("Rubber Band input must fit the configured block/channels")
        self._input[:, :len(samples)] = samples.T
        self._api.rubberband_process(self._state, self._input_ptrs, len(samples), 0)

    @property
    def available(self) -> int:
        available = int(self._api.rubberband_available(self._state))
        if available < 0:
            raise RuntimeError("Rubber Band unexpectedly ended a live stream")
        return available

    def retrieve(self, output: np.ndarray, frames: int) -> int:
        if not 0 < frames <= min(len(output), self.blocksize):
            raise ValueError("Rubber Band retrieve exceeds its preallocated block")
        received = int(self._api.rubberband_retrieve(self._state, self._output_ptrs, frames))
        if not 0 < received <= frames:
            raise RuntimeError("Rubber Band reported available samples but returned an invalid count")
        output[:received] = self._output[:, :received].T
        return received

    def close(self) -> None:
        if self._state:
            self._api.rubberband_delete(self._state)
            self._state = None
