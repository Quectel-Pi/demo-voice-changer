import math
import unittest

import numpy as np

from voice_changer.dsp.parameters import EffectParameters
from voice_changer.dsp.streaming_effects import StreamingEffects


def process_block(
    effects: StreamingEffects,
    samples: np.ndarray,
    parameters: EffectParameters,
) -> np.ndarray:
    output = np.empty_like(samples)
    effects.process(samples, output, parameters)
    return output


def settle(
    effects: StreamingEffects,
    parameters: EffectParameters,
    *,
    limit: int = 64,
) -> int:
    silence = np.zeros((effects.blocksize, effects.channels), dtype=np.float32)
    for blocks in range(1, limit + 1):
        process_block(effects, silence, parameters)
        if effects.settled:
            return blocks
    raise AssertionError("effect parameters did not settle within the test limit")


class StreamingEffectsTests(unittest.TestCase):
    def test_default_parameters_are_exact_passthrough_for_supported_shapes(self) -> None:
        for channels in (1, 2):
            for blocksize in (256, 512, 1024):
                with self.subTest(channels=channels, blocksize=blocksize):
                    effects = StreamingEffects(48000, channels, blocksize)
                    samples = np.linspace(
                        -0.75, 0.75, blocksize * channels, dtype=np.float32,
                    ).reshape(blocksize, channels)
                    output = np.full_like(samples, np.nan)

                    effects.process(samples, output, EffectParameters())

                    np.testing.assert_array_equal(output, samples)
                    self.assertTrue(effects.settled)

    def test_robot_modulation_has_expected_80_hz_sidebands(self) -> None:
        effects = StreamingEffects(48000, 1, 1024)
        parameters = EffectParameters(bypass=False, robot_mix=1.0)
        settle(effects, parameters)
        rendered = []
        frame = 0
        for _ in range(24):
            indices = frame + np.arange(effects.blocksize)
            samples = np.sin(2.0 * math.pi * 600.0 * indices / 48000.0)
            samples = samples.astype(np.float32).reshape(-1, 1)
            rendered.append(process_block(effects, samples, parameters)[:, 0])
            frame += effects.blocksize

        signal = np.concatenate(rendered)[576:]
        amplitudes = 2.0 * np.abs(np.fft.rfft(signal)) / signal.size
        frequencies = np.fft.rfftfreq(signal.size, 1.0 / 48000.0)

        for sideband in (520.0, 680.0):
            amplitude = amplitudes[np.argmin(np.abs(frequencies - sideband))]
            self.assertAlmostEqual(float(amplitude), 0.5, delta=0.01)
        carrier = amplitudes[np.argmin(np.abs(frequencies - 600.0))]
        self.assertLess(float(carrier), 0.01)

    def test_robot_phase_continues_across_blocks_and_disabled_audio(self) -> None:
        effects = StreamingEffects(48000, 1, 512)
        default = EffectParameters()
        silence = np.zeros((512, 1), dtype=np.float32)
        disabled_frames = 0
        for _ in range(3):
            process_block(effects, silence, default)
            disabled_frames += 512

        robot = EffectParameters(bypass=False, robot_mix=1.0)
        settled_blocks = settle(effects, robot)
        enabled_frames = 0
        actual = []
        for _ in range(4):
            actual.append(process_block(effects, np.ones_like(silence), robot)[:, 0])
            enabled_frames += 512

        observed = np.concatenate(actual)
        absolute_frames = (
            disabled_frames + settled_blocks * 512 + np.arange(enabled_frames)
        )
        expected = np.sin(2.0 * math.pi * absolute_frames / 600.0)
        np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)

    def test_echo_has_exact_200ms_delay_and_quarter_feedback(self) -> None:
        effects = StreamingEffects(48000, 1, 512)
        parameters = EffectParameters(bypass=False, echo_mix=0.25)
        settle(effects, parameters)
        rendered = []
        for block_index in range(50):
            samples = np.zeros((512, 1), dtype=np.float32)
            if block_index == 0:
                samples[0, 0] = 1.0
            rendered.append(process_block(effects, samples, parameters)[:, 0])
        signal = np.concatenate(rendered)

        self.assertAlmostEqual(float(signal[0]), 0.75, places=6)
        self.assertAlmostEqual(float(signal[9600]), 0.25, places=6)
        self.assertAlmostEqual(float(signal[19200]), 0.0625, places=6)
        np.testing.assert_array_equal(signal[1:9600], np.zeros(9599, dtype=np.float32))

    def test_echo_history_survives_mix_adjustments(self) -> None:
        effects = StreamingEffects(48000, 1, 512)
        initial = EffectParameters(bypass=False, echo_mix=0.2)
        settle(effects, initial)
        impulse = np.zeros((512, 1), dtype=np.float32)
        impulse[0, 0] = 1.0
        process_block(effects, impulse, initial)

        adjusted = EffectParameters(bypass=False, echo_mix=0.35)
        echo_frame = None
        cursor = 512
        for _ in range(19):
            samples = np.zeros((512, 1), dtype=np.float32)
            output = process_block(effects, samples, adjusted)[:, 0]
            if cursor <= 9600 < cursor + 512:
                echo_frame = output[9600 - cursor]
            cursor += 512

        self.assertIsNotNone(echo_frame)
        self.assertGreater(float(echo_frame), 0.25)

    def test_reverb_has_four_first_reflections_and_a_decaying_tail(self) -> None:
        effects = StreamingEffects(48000, 1, 256)
        parameters = EffectParameters(bypass=False, reverb_mix=0.4)
        settle(effects, parameters)
        rendered = []
        block_count = math.ceil(2.5 * 48000 / effects.blocksize)
        for block_index in range(block_count):
            samples = np.zeros((256, 1), dtype=np.float32)
            if block_index == 0:
                samples[0, 0] = 1.0
            rendered.append(process_block(effects, samples, parameters)[:, 0])
        signal = np.concatenate(rendered)[:120000]

        self.assertAlmostEqual(float(signal[0]), 0.6, places=6)
        self.assertAlmostEqual(
            float(np.sum(np.abs(signal[1:1488]))), 0.0, places=7,
        )
        for delay in (1488, 1776, 1968, 2256):
            self.assertAlmostEqual(float(signal[delay]), 0.1, places=6)
        self.assertNotEqual(float(signal[2 * 1488]), 0.0)
        early_tail = float(np.sum(np.square(signal[12000:16800])))
        late_tail = float(np.sum(np.square(signal[96000:100800])))
        self.assertGreater(early_tail, late_tail)
        self.assertGreater(float(np.max(np.abs(signal[96000:]))), 0.0)

    def test_reverb_31ms_comb_matches_nominal_08s_decay(self) -> None:
        effects = StreamingEffects(48000, 1, 256)
        parameters = EffectParameters(bypass=False, reverb_mix=0.4)
        settle(effects, parameters)

        measured = {}
        target_frames = {1488, 2976, 40176}
        for block_start in range(0, 40448, effects.blocksize):
            samples = np.zeros((256, 1), dtype=np.float32)
            if block_start == 0:
                samples[0, 0] = 1.0
            output = process_block(effects, samples, parameters)[:, 0]
            for frame in target_frames:
                if block_start <= frame < block_start + effects.blocksize:
                    measured[frame] = float(output[frame - block_start])

        first = measured[1488]
        second = measured[2976]
        last = measured[40176]
        feedback = 10.0 ** (-3.0 * 0.031 / 0.8)
        self.assertAlmostEqual(second / first, feedback, delta=1e-6)
        last_to_first_db = 20.0 * math.log10(last / first)
        expected_last_to_first_db = 20.0 * math.log10(feedback ** 26)
        self.assertAlmostEqual(
            last_to_first_db, expected_last_to_first_db, delta=0.1,
        )
        self.assertAlmostEqual(last_to_first_db, -60.45, delta=0.1)

    def test_bypass_smooths_and_clears_stale_delay_history(self) -> None:
        effects = StreamingEffects(48000, 1, 512)
        echo = EffectParameters(bypass=False, echo_mix=0.4)
        settle(effects, echo)
        impulse = np.zeros((512, 1), dtype=np.float32)
        impulse[0, 0] = 1.0
        process_block(effects, impulse, echo)

        bypass = EffectParameters()
        ones = np.ones_like(impulse)
        first_fade_block = process_block(effects, ones, bypass)
        self.assertFalse(effects.settled)
        self.assertAlmostEqual(float(first_fade_block[0, 0]), 0.6, places=6)
        settle(effects, bypass)

        silence = np.zeros_like(impulse)
        for _ in range(20):
            np.testing.assert_array_equal(
                process_block(effects, silence, bypass), silence,
            )
        np.testing.assert_array_equal(process_block(effects, ones, bypass), ones)
        self.assertTrue(effects.settled)

    def test_stereo_channels_remain_independent_and_storage_is_bounded(self) -> None:
        effects = StreamingEffects(48000, 2, 512)
        parameters = EffectParameters(bypass=False, echo_mix=0.4)
        settle(effects, parameters)
        samples = np.zeros((512, 2), dtype=np.float32)
        samples[0, 0] = 1.0
        process_block(effects, samples, parameters)
        for _ in range(19):
            output = process_block(
                effects, np.zeros_like(samples), parameters,
            )
            np.testing.assert_array_equal(output[:, 1], np.zeros(512, dtype=np.float32))
        self.assertEqual(effects._echo_line.shape, (9600, 2))
        self.assertEqual(
            tuple(line.shape[0] for line in effects._reverb_lines),
            (1488, 1776, 1968, 2256),
        )
        self.assertLess(
            effects._robot_table.nbytes
            + effects._echo_line.nbytes
            + effects._reverb_delayed.nbytes
            + sum(line.nbytes for line in effects._reverb_lines),
            1_000_000,
        )

    def test_configuration_snapshot_and_buffers_are_validated(self) -> None:
        for config in (
            (44100, 1, 256),
            (48000, 3, 256),
            (48000, 1, 128),
            (True, 1, 256),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                StreamingEffects(*config)
        for parameters in (
            {"bypass": False, "robot_mix": 1.01},
            {"bypass": False, "echo_mix": float("inf")},
            {"bypass": False, "reverb_mix": -0.01},
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                EffectParameters(**parameters)

        effects = StreamingEffects(48000, 1, 256)
        samples = np.zeros((256, 1), dtype=np.float32)
        output = np.empty_like(samples)
        with self.assertRaises(ValueError):
            effects.process(samples, output, object())
        with self.assertRaises(ValueError):
            effects.process(samples[:-1], output, EffectParameters())
        with self.assertRaises(ValueError):
            effects.process(samples, output.astype(np.float64), EffectParameters())
        readonly_output = np.empty_like(samples)
        readonly_output.flags.writeable = False
        with self.assertRaises(ValueError):
            effects.process(samples, readonly_output, EffectParameters())

        invalid = EffectParameters()
        object.__setattr__(invalid, "robot_mix", float("nan"))
        with self.assertRaises(ValueError):
            effects.process(samples, output, invalid)
        samples[0, 0] = np.inf
        with self.assertRaises(ValueError):
            effects.process(samples, output, EffectParameters())


if __name__ == "__main__":
    unittest.main()
