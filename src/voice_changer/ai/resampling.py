"""Stateful NumPy polyphase-style resampling for the optional AI worker."""

from __future__ import annotations

import math

import numpy as np


class StreamingResampler:
    """Windowed-sinc resampler that preserves phase across input blocks.

    The symmetric FIR waits for its right-hand context before publishing
    samples. It is intended for worker-thread audio, not a PortAudio callback.
    """

    def __init__(self, input_rate: int, output_rate: int, *, taps: int = 63) -> None:
        for name, value in (("input_rate", input_rate), ("output_rate", output_rate)):
            if type(value) is not int or value < 8000 or value > 192000:
                raise ValueError(f"{name} must be an integer from 8000 to 192000 Hz")
        if type(taps) is not int or taps < 15 or taps > 255 or taps % 2 == 0:
            raise ValueError("Resampler taps must be an odd integer from 15 to 255")
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.taps = taps
        self._half = taps // 2
        self._ratio = min(1.0, output_rate / input_rate)
        self._offsets = np.arange(-self._half, self._half + 1, dtype=np.int64)
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0
        self._total_input = 0
        self._next_output = 0
        self._divisor = math.gcd(input_rate, output_rate)
        self._phases = output_rate // self._divisor
        self._weights = (
            self._filter(np.arange(self._phases, dtype=np.float64) / self._phases)
            .astype(np.float32)
            if self._phases <= 1024 else None
        )

    def _filter(self, fractions: np.ndarray) -> np.ndarray:
        relative = self._offsets[None, :] - fractions[:, None]
        window = (
            0.42 + 0.5 * np.cos(np.pi * relative / self._half)
            + 0.08 * np.cos(2 * np.pi * relative / self._half)
        )
        window[np.abs(relative) > self._half] = 0
        weights = self._ratio * np.sinc(self._ratio * relative) * window
        sums = weights.sum(axis=1, keepdims=True)
        if not np.isfinite(weights).all() or np.any(np.abs(sums) < 1e-12):
            raise RuntimeError("Resampler produced invalid filter coefficients")
        return weights / sums

    @property
    def group_delay_seconds(self) -> float:
        if self.input_rate == self.output_rate:
            return 0.0
        return self._half / self.input_rate

    def process(self, samples: np.ndarray) -> np.ndarray:
        if (not isinstance(samples, np.ndarray) or samples.ndim != 1
                or samples.dtype != np.float32
                or not np.isfinite(samples).all()
                or np.any(np.abs(samples) > 1)):
            raise ValueError("Resampler input must be finite normalized mono float32")
        if samples.size == 0:
            return np.empty(0, dtype=np.float32)
        if self.input_rate == self.output_rate:
            self._total_input += samples.size
            return samples.copy()

        self._buffer = np.concatenate((self._buffer, samples))
        total = self._total_input + samples.size
        self._total_input = total
        last_center = total - 1 - self._half
        last_output = (last_center * self.output_rate) // self.input_rate
        if last_output < self._next_output:
            return np.empty(0, dtype=np.float32)

        output_indices = np.arange(self._next_output, last_output + 1, dtype=np.int64)
        numerators = output_indices * self.input_rate
        centers = numerators // self.output_rate
        remainder = numerators % self.output_rate
        weights = (
            self._weights[remainder // self._divisor] if self._weights is not None
            else self._filter(remainder.astype(np.float64) / self.output_rate)
            .astype(np.float32)
        )

        absolute = centers[:, None] + self._offsets[None, :]
        local = absolute - self._buffer_start
        valid = (absolute >= 0) & (absolute < total) & (local >= 0) & (local < len(self._buffer))
        # float32 keeps the worker fast enough for the 21.3 ms board budget.
        # Each output row is summed independently, so chunk boundaries cannot
        # change the result.
        values = np.zeros(absolute.shape, dtype=np.float32)
        values[valid] = self._buffer[local[valid]]
        output = np.sum(values * weights, axis=1, dtype=np.float32)
        if not np.isfinite(output).all():
            raise RuntimeError("Resampler produced non-finite audio")

        self._next_output = last_output + 1
        next_center = (self._next_output * self.input_rate) // self.output_rate
        keep_from = min(total, max(0, next_center - self._half))
        discard = keep_from - self._buffer_start
        if discard > 0:
            self._buffer = self._buffer[discard:].copy()
            self._buffer_start = keep_from
        return output
