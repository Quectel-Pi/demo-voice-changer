"""Synthetic streaming tests for the demo spectral-gate noise reduction."""

import math
import unittest

import numpy as np

from voice_changer.dsp.denoise import StreamingDenoiser


def render(denoiser, samples, strengths):
    outputs = []
    size = denoiser.blocksize
    for index, start in enumerate(range(0, len(samples) - size + 1, size)):
        block = np.empty((size, denoiser.channels), dtype=np.float32)
        strength = strengths[index] if isinstance(strengths, (list, tuple)) else strengths
        denoiser.process(samples[start:start + size], block, strength)
        outputs.append(block)
    return np.concatenate(outputs)


def noise(seconds, channels, amplitude, seed=0):
    rng = np.random.default_rng(seed)
    frames = int(48000 * seconds) // 512 * 512
    return (rng.standard_normal((frames, channels)) * amplitude).astype(np.float32)


class DenoiseFormatTests(unittest.TestCase):
    def test_supported_formats_and_strength_validation(self):
        for samplerate, channels, blocksize in (
            (44100, 1, 512), (48000, 3, 512), (48000, 1, 128),
        ):
            with self.subTest(samplerate=samplerate, channels=channels,
                              blocksize=blocksize):
                with self.assertRaisesRegex(ValueError, "D1 demo denoise"):
                    StreamingDenoiser(samplerate, channels, blocksize)
        denoiser = StreamingDenoiser(48000, 1, 512)
        block = np.zeros((512, 1), dtype=np.float32)
        for strength in (True, "1", math.nan, math.inf, -0.01, 1.01):
            with self.subTest(strength=strength):
                with self.assertRaisesRegex(ValueError, "denoise strength"):
                    denoiser.process(block, block.copy(), strength)

    def test_output_must_be_a_writable_matching_block(self):
        denoiser = StreamingDenoiser(48000, 1, 512)
        block = np.zeros((512, 1), dtype=np.float32)
        frozen = np.zeros((512, 1), dtype=np.float32)
        frozen.flags.writeable = False
        with self.assertRaisesRegex(ValueError, "writable matching block"):
            denoiser.process(block, frozen, 0.0)
        with self.assertRaisesRegex(ValueError, "writable matching block"):
            denoiser.process(block, np.zeros((256, 1), dtype=np.float32), 0.0)
        with self.assertRaisesRegex(ValueError, "configured block shape"):
            denoiser.process(np.zeros((512, 2), dtype=np.float32), block, 0.0)


class DenoiseBypassTests(unittest.TestCase):
    def test_zero_strength_is_bit_exact_delayed_copy(self):
        for channels in (1, 2):
            for blocksize in (256, 512, 1024):
                with self.subTest(channels=channels, blocksize=blocksize):
                    denoiser = StreamingDenoiser(48000, channels, blocksize)
                    samples = noise(1.0, channels, 0.1, seed=channels + blocksize)
                    rendered = render(denoiser, samples, 0.0)
                    hop = denoiser.latency
                    self.assertEqual(hop, 256)
                    self.assertTrue(np.all(rendered[:hop] == 0))
                    np.testing.assert_array_equal(
                        rendered[hop:], samples[:len(rendered) - hop],
                    )
                    self.assertTrue(np.isfinite(rendered).all())

    def test_zero_strength_pcm_path_stays_exact_after_gate_activity(self):
        denoiser = StreamingDenoiser(48000, 1, 512)
        samples = noise(2.0, 1, 0.05, seed=7)
        render(denoiser, samples, 1.0)
        follow_up = noise(1.0, 1, 0.05, seed=8)
        rendered = render(denoiser, follow_up, 0.0)
        hop = denoiser.latency
        settle = 64 * 512
        np.testing.assert_array_equal(
            rendered[settle:], follow_up[settle - hop:len(rendered) - hop],
        )


class DenoiseBehaviourTests(unittest.TestCase):
    def test_strength_change_settles_within_a_few_blocks(self):
        denoiser = StreamingDenoiser(48000, 1, 512)
        block = np.zeros((512, 1), dtype=np.float32)
        for strength in (1.0, 0.5, 0.0):
            denoiser.process(block, block.copy(), strength)
            self.assertFalse(denoiser.settled)
            for _ in range(64):
                denoiser.process(block, block.copy(), strength)
            self.assertTrue(denoiser.settled, strength)

    def test_noise_floor_is_attenuated_at_full_strength(self):
        denoiser = StreamingDenoiser(48000, 1, 512)
        samples = noise(3.0, 1, 0.03, seed=3)
        rendered = render(denoiser, samples, 1.0)
        tail = slice(48000, len(rendered))
        ratio = (np.sqrt(np.mean(rendered[tail] ** 2))
                 / np.sqrt(np.mean(samples[:len(rendered)][tail] ** 2)))
        self.assertLess(ratio, 0.6)
        self.assertGreater(ratio, 0.02)

    def test_strength_controls_how_much_noise_is_removed(self):
        samples = noise(3.0, 1, 0.03, seed=5)
        tail = slice(48000, len(samples))
        ratios = {}
        for strength in (0.25, 1.0):
            denoiser = StreamingDenoiser(48000, 1, 512)
            rendered = render(denoiser, samples, strength)
            ratios[strength] = (
                np.sqrt(np.mean(rendered[tail] ** 2))
                / np.sqrt(np.mean(samples[tail] ** 2))
            )
        self.assertLess(ratios[1.0], ratios[0.25])

    def test_both_stereo_channels_are_denoised_independently(self):
        denoiser = StreamingDenoiser(48000, 2, 512)
        samples = noise(3.0, 2, 0.03, seed=11)
        rendered = render(denoiser, samples, 1.0)
        tail = slice(48000, len(rendered))
        for channel in (0, 1):
            ratio = (np.sqrt(np.mean(rendered[tail, channel] ** 2))
                     / np.sqrt(np.mean(samples[:len(rendered)][tail, channel] ** 2)))
            self.assertLess(ratio, 0.6)

    def test_rendered_path_keeps_signal_alignment_and_has_fixed_latency(self):
        blocksize = 512
        denoiser = StreamingDenoiser(48000, 1, blocksize)
        samples = np.zeros((blocksize * 20, 1), dtype=np.float32)
        samples[1000] = 1.0
        rendered = render(denoiser, samples, 0.01)
        hop = denoiser.latency
        peak = int(np.argmax(np.abs(rendered[:, 0])))
        self.assertEqual(peak, 1000 + hop)
        self.assertGreater(float(rendered[peak, 0]), 0.99)
        self.assertLess(float(np.abs(rendered[: hop + 900, 0]).max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
