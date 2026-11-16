"""Streaming spectral-gate noise reduction for the demo DSP chain.

The gate works on a fixed 512-frame sqrt-Hann STFT with 50 % overlap.  It
always runs so the noise profile is warm even while the control is off, but
with ``strength == 0`` the output comes from a PCM delay line, which keeps the
bypass path bit-exact.  Every path emits exactly ``blocksize`` frames per call
and introduces a fixed ``hop`` (256-frame, 5.3 ms at 48 kHz) delay.
"""

import math

import numpy as np


class StreamingDenoiser:
    """Bounded, allocation-free-after-setup spectral gate for one audio stream."""

    frame_size = 512

    def __init__(self, samplerate: int, channels: int, blocksize: int) -> None:
        if (
            samplerate != 48000
            or channels not in (1, 2)
            or blocksize not in (256, 512, 1024)
        ):
            raise ValueError(
                "D1 demo denoise supports 48000 Hz, 256/512/1024 frames and "
                "1 or 2 matching channels"
            )
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.hop = self.frame_size // 2
        index = np.arange(self.frame_size)
        self._window = np.sqrt(
            0.5 - 0.5 * np.cos(2.0 * np.pi * index / self.frame_size)
        )[:, None]
        self._pending = np.zeros(
            (self.frame_size + blocksize, channels), dtype=np.float64,
        )
        self._pending_count = 0
        self._overlap = np.zeros((self.frame_size, channels), dtype=np.float64)
        self._out = np.zeros(
            (self.frame_size + blocksize, channels), dtype=np.float64,
        )
        self._out_count = 0
        # A fixed PCM delay keeps the strength=0 path sample-exact.
        self._delay = np.zeros((self.hop, channels), dtype=np.float32)
        self._delayed = np.empty((blocksize, channels), dtype=np.float32)
        self._rendered = np.empty((blocksize, channels), dtype=np.float32)
        # Both output paths start with the same hop of silence.
        self._warmup = self.hop
        self._noise = np.zeros((self.frame_size // 2 + 1, channels), dtype=np.float64)
        self._power = np.zeros_like(self._noise)
        self._primed = False
        self._gain = np.ones_like(self._noise)
        self._strength = 0.0
        self._target = 0.0
        self._smoothing = 1 - math.exp(-blocksize / (samplerate * 0.02))

    @property
    def latency(self) -> int:
        return self.hop

    @property
    def settled(self) -> bool:
        return abs(self._target - self._strength) < 0.001

    def process(
        self,
        samples: np.ndarray,
        output: np.ndarray,
        strength: float,
    ) -> None:
        """Render one block; ``strength`` is 0..1 and smoothed per block."""
        if (
            not isinstance(strength, (int, float))
            or isinstance(strength, bool)
            or not math.isfinite(strength)
            or not 0.0 <= strength <= 1.0
        ):
            raise ValueError("denoise strength must be finite and in [0, 1]")
        if samples.shape != (self.blocksize, self.channels):
            raise ValueError("denoise samples must match the configured block shape")
        if output.shape != samples.shape or not output.flags.writeable:
            raise ValueError("denoise output must be a writable matching block")
        self._target = float(strength)
        self._strength += (self._target - self._strength) * self._smoothing
        if abs(self._target - self._strength) < 0.001:
            self._strength = self._target

        # PCM delay: hop frames of history, then the start of this block.
        self._delayed[: self.hop] = self._delay
        self._delayed[self.hop:] = samples[: self.blocksize - self.hop]
        self._delay[:] = samples[self.blocksize - self.hop:]

        self._pending[self._pending_count : self._pending_count + self.blocksize] = samples
        self._pending_count += self.blocksize
        while self._pending_count >= self.frame_size:
            self._render_frame(self._pending[: self.frame_size])
            keep = self._pending_count - self.hop
            self._pending[:keep] = self._pending[self.hop : self._pending_count]
            self._pending_count = keep

        self._emit(self._rendered)
        if self._strength == 0.0:
            np.copyto(output, self._delayed)
        else:
            np.copyto(output, self._rendered)

    def _render_frame(self, frame: np.ndarray) -> None:
        spectrum = np.fft.rfft(frame * self._window, axis=0)
        power = spectrum.real ** 2 + spectrum.imag ** 2
        # Short smoothing then slow upward minimum tracking: sustained speech
        # does not get learned as noise, while a steady floor does.
        self._power = 0.8 * self._power + 0.2 * power
        if not self._primed:
            # The first observed spectrum becomes the profile; starting from
            # silence would gate all speech until the estimate caught up.
            self._noise[:] = self._power
            self._primed = True
        else:
            self._noise = np.where(
                self._power < self._noise, self._power, self._noise * 1.004,
            )
        strength = self._strength
        if strength == 0.0:
            gain = np.ones_like(power)
        else:
            noise = (1.0 + 1.2 * strength) * 2.5 * self._noise
            snr = np.maximum(power - noise, 0.0) / np.maximum(power, 1e-12)
            gain = np.maximum(np.sqrt(snr), 1.0 - 0.9 * strength)
            gain = 0.5 * self._gain + 0.5 * gain
            smoothed = gain.copy()
            smoothed[1:-1] = (
                0.25 * gain[:-2] + 0.5 * gain[1:-1] + 0.25 * gain[2:]
            )
            gain = smoothed
            spectrum *= gain
        self._gain = gain
        time = np.fft.irfft(spectrum, n=self.frame_size, axis=0) * self._window
        self._overlap += time
        self._out[self._out_count : self._out_count + self.hop] = self._overlap[: self.hop]
        self._out_count += self.hop
        self._overlap[: -self.hop] = self._overlap[self.hop :]
        self._overlap[-self.hop :] = 0.0

    def _emit(self, output: np.ndarray) -> None:
        silent = min(self._warmup, self.blocksize)
        output[:silent] = 0.0
        self._warmup -= silent
        take = self.blocksize - silent
        if take > self._out_count:
            raise RuntimeError("Denoise output underrun; the bounded buffers are inconsistent")
        if take:
            np.copyto(output[silent:], self._out[:take])
            keep = self._out_count - take
            self._out[:keep] = self._out[take : self._out_count]
            self._out_count = keep
