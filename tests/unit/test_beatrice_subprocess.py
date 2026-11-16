import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from voice_changer.ai import beatrice, beatrice_subprocess
from voice_changer.audio.device_types import DeviceRef, StreamConfig


def stream_config(blocksize=1024, channels=1):
    device = DeviceRef(0, "test", 0)
    return StreamConfig(device, device, 48000, blocksize, channels, 0.9)


class _FakeClient:
    def __init__(self, blocksize=1024, gain=1.0):
        self.blocksize = blocksize
        self.gain = gain
        self.calls = 0
        self.closed = False
        self.arguments = []

    def convert(self, samples):
        self.calls += 1
        self.arguments.append(np.asarray(samples).copy())
        return np.asarray(samples, dtype=np.float32) * self.gain

    def close(self):
        self.closed = True


class _FakeStage:
    def __init__(self, factor=2.0):
        self.factor = factor
        self.calls = 0
        self.closed = False

    def process(self, block):
        self.calls += 1
        return np.asarray(block, dtype=np.float32) * self.factor

    def close(self):
        self.closed = True


class _FailingClient(_FakeClient):
    def __init__(self, fail_on=2, **kwargs):
        super().__init__(**kwargs)
        self.fail_on = fail_on

    def convert(self, samples):
        if self.calls + 1 >= self.fail_on:
            self.calls += 1
            raise RuntimeError("helper stalled")
        return super().convert(samples)


class SubprocessEnabledTests(unittest.TestCase):
    def test_only_truthy_values_enable_the_experiment(self):
        for value, expected in (("1", True), ("true", True), ("YES", True), (" on ", True),
                                ("", False), ("0", False), ("no", False), ("off", False)):
            with mock.patch.dict(os.environ, {beatrice.SUBPROCESS_ENV: value}):
                self.assertIs(beatrice.subprocess_enabled(), expected, value)

    def test_missing_variable_means_disabled(self):
        env = dict(os.environ)
        env.pop(beatrice.SUBPROCESS_ENV, None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(beatrice.subprocess_enabled())

    def test_worker_script_and_python_are_resolvable(self):
        self.assertTrue(beatrice_subprocess.worker_script().is_file())
        with mock.patch.dict(os.environ, {"VOICE_CHANGER_BEATRICE_PYTHON": "/opt/py13"}):
            self.assertEqual(beatrice_subprocess.worker_python(), "/opt/py13")
        env = dict(os.environ)
        env.pop("VOICE_CHANGER_BEATRICE_PYTHON", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(beatrice_subprocess.worker_python(), __import__("sys").executable)


class ClientConverterTests(unittest.TestCase):
    def test_mono_block_is_forwarded_and_clipped(self):
        client = _FakeClient(gain=2.0)
        converter = beatrice_subprocess._ClientConverter(client, type("C", (), {"channels": 1})())
        block = np.full((1024, 1), 0.4, dtype=np.float32)
        output = converter(block)
        self.assertEqual(output.shape, (1024, 1))
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.all(output <= 0.8 + 1e-6))
        self.assertEqual(client.calls, 1)
        # The mono input is clipped before it reaches the helper.
        self.assertLessEqual(float(np.max(np.abs(client.arguments[0]))), 1.0)

    def test_stereo_is_averaged_and_repeated(self):
        client = _FakeClient(gain=1.0)
        converter = beatrice_subprocess._ClientConverter(client, type("C", (), {"channels": 2})())
        block = np.zeros((1024, 2), dtype=np.float32)
        block[:, 0] = 0.25
        block[:, 1] = -0.25
        output = converter(block)
        self.assertEqual(output.shape, (1024, 2))
        self.assertTrue(np.allclose(output, 0.0))

    def test_wrong_shape_is_rejected(self):
        client = _FakeClient()
        converter = beatrice_subprocess._ClientConverter(client, type("C", (), {"channels": 1})())
        with self.assertRaises(ValueError):
            converter(np.zeros((512, 1), dtype=np.float32))

    def test_close_closes_the_helper(self):
        client = _FakeClient()
        converter = beatrice_subprocess._ClientConverter(client, type("C", (), {"channels": 1})())
        converter.close()
        self.assertTrue(client.closed)

    def test_formant_pipeline_keeps_block_order(self):
        client = _FakeClient(gain=1.0)
        stage = _FakeStage(factor=3.0)
        converter = beatrice_subprocess._ClientConverter(
            client, type("C", (), {"channels": 1})(), stage=stage)
        lag = beatrice_subprocess._ClientConverter.PIPELINE_LAG
        self.assertGreaterEqual(lag, 1)
        blocks, outputs = [], []
        try:
            for index in range(lag + 4):
                block = np.full((1024, 1), (index + 1) / 10.0, dtype=np.float32)
                blocks.append(block)
                outputs.append(converter(block))
        finally:
            converter.close()
        for index in range(lag):
            self.assertTrue(np.all(outputs[index] == 0.0))
        for index in range(len(blocks) - lag):
            self.assertEqual(outputs[lag + index].shape, (1024, 1))
            self.assertEqual(outputs[lag + index].dtype, np.float32)
            self.assertTrue(np.allclose(outputs[lag + index], blocks[index] * 3.0))
        self.assertGreaterEqual(stage.calls, len(blocks) - lag)
        self.assertTrue(client.closed)
        self.assertTrue(stage.closed)

    def test_formant_pipeline_surfaces_helper_failures(self):
        client = _FailingClient(fail_on=1, gain=1.0)
        stage = _FakeStage()
        converter = beatrice_subprocess._ClientConverter(
            client, type("C", (), {"channels": 1})(), stage=stage)
        lag = beatrice_subprocess._ClientConverter.PIPELINE_LAG
        try:
            for _ in range(lag):
                self.assertEqual(
                    converter(np.full((1024, 1), 0.1, dtype=np.float32)).shape,
                    (1024, 1))
            with self.assertRaises(RuntimeError):
                for _ in range(4):
                    converter(np.full((1024, 1), 0.2, dtype=np.float32))
        finally:
            converter.close()

    def test_closed_formant_converter_rejects_blocks(self):
        client = _FakeClient()
        converter = beatrice_subprocess._ClientConverter(
            client, type("C", (), {"channels": 1})(), stage=_FakeStage())
        converter.close()
        with self.assertRaises(RuntimeError):
            converter(np.zeros((1024, 1), dtype=np.float32))


class ProcessorTests(unittest.TestCase):
    def test_helper_processor_uses_the_worker_queue_reserve(self):
        client = _FakeClient()
        with mock.patch.object(beatrice_subprocess, "BeatriceWorkerClient", return_value=client):
            processor = beatrice_subprocess.subprocess_processor(
                stream_config(), beatrice.BeatriceConfig(Path("/api"), Path("/model"), 5))
        self.assertEqual(processor.conversion_frames, 1024)
        self.assertEqual(processor.channels, 1)
        processor.close()
        self.assertTrue(client.closed)

    def test_unsupported_device_config_is_rejected(self):
        with self.assertRaises(ValueError):
            beatrice_subprocess.subprocess_processor(
                stream_config(blocksize=512), beatrice.BeatriceConfig(Path("/api"), Path("/model")))

    def test_measure_block_ms_averages_warm_round_trips(self):
        client = _FakeClient()
        times = iter([0.0, 0.08])
        with mock.patch.object(beatrice_subprocess, "BeatriceWorkerClient", return_value=client), \
                mock.patch.object(beatrice_subprocess.time, "monotonic", side_effect=lambda: next(times)):
            measured = beatrice_subprocess.measure_block_ms(
                stream_config(), beatrice.BeatriceConfig(Path("/api"), Path("/model")),
                blocks=4, warmup=2)
        self.assertAlmostEqual(measured, 20.0, places=3)  # 80 ms / 4 blocks
        self.assertEqual(client.calls, 6)
        self.assertTrue(client.closed)

    def test_factory_switches_to_the_helper_when_enabled(self):
        factory = beatrice.BeatriceProcessorFactory(beatrice.BeatriceConfig(Path("/api"), Path("/model")))
        sentinel = object()
        with mock.patch.object(beatrice, "subprocess_enabled", return_value=True), \
                mock.patch.object(beatrice_subprocess, "subprocess_processor", return_value=sentinel) as create:
            result = factory(stream_config(), None)
        self.assertIs(result, sentinel)
        create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
