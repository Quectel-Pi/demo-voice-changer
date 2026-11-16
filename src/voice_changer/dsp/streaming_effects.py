"""Bounded streaming robot, echo, and short-reverb effects."""

import math

import numpy as np

from .parameters import EffectParameters


_ROBOT_PERIOD = 600
_SMOOTHING_SECONDS = 0.02
_SETTLE_EPSILON = 1e-4


class StreamingEffects:
    """Apply smooth, block-streamed effects to fixed-size 48 kHz audio blocks."""

    def __init__(self, samplerate: int, channels: int, blocksize: int) -> None:
        if type(samplerate) is not int or samplerate != 48000:
            raise ValueError("StreamingEffects requires a 48000 Hz sample rate")
        if type(channels) is not int or channels not in (1, 2):
            raise ValueError("StreamingEffects supports one or two channels")
        if type(blocksize) is not int or blocksize not in (256, 512, 1024):
            raise ValueError("StreamingEffects supports 256, 512, or 1024 frame blocks")

        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        shape = (blocksize, channels)

        self._finite = np.empty(shape, dtype=np.bool_)
        self._stage_a = np.empty(shape, dtype=np.float64)
        self._stage_b = np.empty(shape, dtype=np.float64)
        self._stage_c = np.empty(shape, dtype=np.float64)
        self._work = np.empty(shape, dtype=np.float64)
        self._echo_delayed = np.empty(shape, dtype=np.float64)
        self._reverb_delayed = np.empty((4, blocksize, channels), dtype=np.float64)
        self._reverb_wet = np.empty(shape, dtype=np.float64)

        phase = np.arange(0, _ROBOT_PERIOD, 8, dtype=np.float64)[:, None]
        frames = np.arange(blocksize, dtype=np.float64)[None, :]
        self._robot_table = np.empty((phase.shape[0], blocksize), dtype=np.float32)
        np.sin((2.0 * math.pi / _ROBOT_PERIOD) * (phase + frames),
               out=self._robot_table)
        self._robot_block = np.empty(blocksize, dtype=np.float32)
        self._robot_phase = 0

        decay = np.exp(
            -np.arange(blocksize, dtype=np.float64)
            / (samplerate * _SMOOTHING_SECONDS)
        )
        self._smoothing = decay[:, None]
        self._end_smoothing = math.exp(
            -blocksize / (samplerate * _SMOOTHING_SECONDS)
        )
        self._curves = np.empty((3, blocksize, 1), dtype=np.float64)
        self._dry_weights = np.empty((blocksize, 1), dtype=np.float64)
        self._wet_weights = np.empty((blocksize, 1), dtype=np.float64)

        self._echo_delay = round(0.2 * samplerate)
        self._echo_line = np.zeros((self._echo_delay, channels), dtype=np.float64)
        self._echo_position = 0
        reverb_delays = tuple(round(delay * samplerate) for delay in (
            0.031, 0.037, 0.041, 0.047,
        ))
        self._reverb_delays = reverb_delays
        self._reverb_lines = tuple(
            np.zeros((delay, channels), dtype=np.float64) for delay in reverb_delays
        )
        self._reverb_positions = [0, 0, 0, 0]
        self._reverb_feedback = tuple(
            10.0 ** (-3.0 * delay / samplerate / 0.8)
            for delay in reverb_delays
        )

        self._robot_mix = 0.0
        self._echo_mix = 0.0
        self._reverb_mix = 0.0
        self._settled = True

    @property
    def settled(self) -> bool:
        return self._settled

    def process(
        self,
        samples: np.ndarray,
        output: np.ndarray,
        parameters: EffectParameters,
    ) -> None:
        """Process exactly one configured block into a caller-owned output array."""
        if not isinstance(parameters, EffectParameters):
            raise ValueError("An explicit EffectParameters snapshot is required")
        targets = self._validated_targets(parameters)
        expected_shape = (self.blocksize, self.channels)
        if not isinstance(samples, np.ndarray) or samples.shape != expected_shape:
            raise ValueError(f"samples must have shape {expected_shape}")
        if not isinstance(output, np.ndarray) or output.shape != expected_shape:
            raise ValueError(f"output must have shape {expected_shape}")
        if samples.dtype.kind != "f" or output.dtype != samples.dtype:
            raise ValueError("samples and output must use the same floating-point dtype")
        if not output.flags.writeable:
            raise ValueError("output must be writable")
        np.isfinite(samples, out=self._finite)
        if not bool(np.all(self._finite)):
            raise ValueError("samples must contain only finite values")

        current = (self._robot_mix, self._echo_mix, self._reverb_mix)
        self._fill_curve(0, current[0], targets[0])
        self._fill_curve(1, current[1], targets[1])
        self._fill_curve(2, current[2], targets[2])

        if not any(current) and not any(targets):
            np.copyto(output, samples, casting="no")
            self._advance_robot_phase()
            return

        robot_active = current[0] != 0.0 or targets[0] != 0.0
        echo_active = current[1] != 0.0 or targets[1] != 0.0
        reverb_active = current[2] != 0.0 or targets[2] != 0.0

        source = samples
        if robot_active:
            np.copyto(self._stage_a, samples, casting="unsafe")
            np.copyto(
                self._robot_block,
                self._robot_table[self._robot_phase // 8],
            )
            np.multiply(self._stage_a, self._robot_block[:, None], out=self._stage_b)
            self._fill_weights(0)
            np.multiply(self._stage_a, self._dry_weights, out=self._stage_c)
            np.multiply(self._stage_b, self._wet_weights, out=self._work)
            np.add(self._stage_c, self._work, out=self._stage_c)
            source = self._stage_c

        if echo_active:
            self._read_delay(
                self._echo_line, self._echo_position, self._echo_delayed,
            )
            self._fill_weights(1)
            np.multiply(source, self._dry_weights, out=self._stage_a)
            np.multiply(self._echo_delayed, self._wet_weights, out=self._work)
            np.add(self._stage_a, self._work, out=self._stage_a)
            self._write_feedback(
                self._echo_line, self._echo_position, source,
                self._echo_delayed, 0.25,
            )
            self._echo_position = (
                self._echo_position + self.blocksize
            ) % self._echo_delay
            source = self._stage_a

        if reverb_active:
            self._reverb_wet.fill(0.0)
            for index, line in enumerate(self._reverb_lines):
                position = self._reverb_positions[index]
                delayed = self._reverb_delayed[index]
                self._read_delay(line, position, delayed)
                np.multiply(delayed, 0.25, out=self._work)
                np.add(self._reverb_wet, self._work, out=self._reverb_wet)
                self._write_feedback(
                    line, position, source, delayed,
                    self._reverb_feedback[index],
                )
                self._reverb_positions[index] = (
                    position + self.blocksize
                ) % self._reverb_delays[index]

            self._fill_weights(2)
            np.multiply(source, self._dry_weights, out=self._stage_b)
            np.multiply(self._reverb_wet, self._wet_weights, out=self._work)
            np.add(self._stage_b, self._work, out=self._stage_b)
            source = self._stage_b

        np.copyto(output, source, casting="same_kind")
        self._update_amounts(targets)
        self._advance_robot_phase()

    @staticmethod
    def _validated_targets(parameters: EffectParameters) -> tuple[float, float, float]:
        values = (
            parameters.robot_mix,
            parameters.echo_mix,
            parameters.reverb_mix,
        )
        limits = (1.0, 0.4, 0.4)
        for value, limit in zip(values, limits):
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0.0 <= value <= limit
            ):
                raise ValueError("Effect amounts must be finite and within their validated bounds")
        if type(parameters.bypass) is not bool:
            raise ValueError("bypass must be a boolean")
        if parameters.bypass and any(values):
            raise ValueError("bypass snapshots must have zero effect amounts")
        if parameters.bypass:
            return (0.0, 0.0, 0.0)
        return tuple(float(value) for value in values)

    def _fill_curve(self, index: int, current: float, target: float) -> None:
        curve = self._curves[index]
        np.multiply(self._smoothing, current - target, out=curve)
        np.add(curve, target, out=curve)

    def _fill_weights(self, index: int) -> None:
        curve = self._curves[index]
        np.subtract(1.0, curve, out=self._dry_weights)
        np.copyto(self._wet_weights, curve)

    def _update_amounts(self, targets: tuple[float, float, float]) -> None:
        values = []
        for current, target in zip(
            (self._robot_mix, self._echo_mix, self._reverb_mix), targets,
        ):
            updated = target + (current - target) * self._end_smoothing
            if abs(updated - target) <= _SETTLE_EPSILON:
                updated = target
            values.append(updated)
        self._robot_mix, self._echo_mix, self._reverb_mix = values
        self._settled = all(
            current == target
            for current, target in zip(values, targets)
        )
        if self._echo_mix == 0.0 and targets[1] == 0.0:
            self._echo_line.fill(0.0)
            self._echo_position = 0
        if self._reverb_mix == 0.0 and targets[2] == 0.0:
            for line in self._reverb_lines:
                line.fill(0.0)
            self._reverb_positions[:] = (0, 0, 0, 0)

    def _advance_robot_phase(self) -> None:
        self._robot_phase = (self._robot_phase + self.blocksize) % _ROBOT_PERIOD

    def _read_delay(
        self,
        line: np.ndarray,
        position: int,
        destination: np.ndarray,
    ) -> None:
        first = min(self.blocksize, line.shape[0] - position)
        np.copyto(destination[:first], line[position:position + first])
        remainder = self.blocksize - first
        if remainder:
            np.copyto(destination[first:], line[:remainder])

    def _write_feedback(
        self,
        line: np.ndarray,
        position: int,
        source: np.ndarray,
        delayed: np.ndarray,
        feedback: float,
    ) -> None:
        first = min(self.blocksize, line.shape[0] - position)
        self._write_feedback_slice(
            line[position:position + first], source[:first], delayed[:first], feedback,
        )
        remainder = self.blocksize - first
        if remainder:
            self._write_feedback_slice(
                line[:remainder], source[first:], delayed[first:], feedback,
            )

    @staticmethod
    def _write_feedback_slice(
        destination: np.ndarray,
        source: np.ndarray,
        delayed: np.ndarray,
        feedback: float,
    ) -> None:
        np.multiply(delayed, feedback, out=destination)
        np.add(destination, source, out=destination)
