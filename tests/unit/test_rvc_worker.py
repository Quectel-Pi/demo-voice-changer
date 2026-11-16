import threading
import time
import unittest

import numpy as np

from voice_changer.ai.rvc_worker import RvcWorkerProcessor
from voice_changer.audio.device_types import CaptureConfig, DeviceRef, StreamConfig


def stream_config(blocksize=16):
    device = DeviceRef(0, "test", 0)
    return StreamConfig(device, device, 48000, blocksize, 1, 0.9)


def wait_until(predicate, timeout=1.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


class RvcWorkerTests(unittest.TestCase):
    def test_nondivisible_device_and_conversion_blocks_preserve_sample_order(self):
        calls = []

        def convert(samples):
            calls.append(samples.copy())
            return samples.copy()

        worker = RvcWorkerProcessor(
            stream_config(16), convert, conversion_frames=40, startup_blocks=1,
        )
        worker.start()
        outputs = []
        source = np.arange(320, dtype=np.float32)[:, None] / 1000
        try:
            for offset in range(0, 320, 16):
                output = np.empty((16, 1), dtype=np.float32)
                self.assertTrue(worker.exchange(source[offset:offset+16], output))
                outputs.append(output.copy())
                target = ((offset+16)//40)*40
                self.assertTrue(wait_until(lambda: worker._processed >= target))
            np.testing.assert_array_equal(np.concatenate(calls), source)
            played = np.concatenate(outputs)[48:]
            np.testing.assert_array_equal(played, source[:len(played)])
            self.assertEqual(worker.input.capacity, 40*8)
        finally:
            worker.close()

    def test_converter_runs_off_callback_and_output_is_ordered(self):
        thread_names = []

        def convert(samples):
            thread_names.append(threading.current_thread().name)
            return samples * np.float32(0.5)

        worker = RvcWorkerProcessor(stream_config(), convert, queue_blocks=4, startup_blocks=1)
        worker.start()
        output = np.empty((16, 1), dtype=np.float32)
        try:
            block = np.full((16, 1), 0.1, dtype=np.float32)
            self.assertTrue(worker.exchange(block, output))
            np.testing.assert_array_equal(output, 0)
            self.assertTrue(wait_until(lambda: worker.output.available >= 16))
            block = np.full((16, 1), 0.2, dtype=np.float32)
            self.assertTrue(worker.exchange(block, output))
            np.testing.assert_array_equal(output, np.full((16, 1), 0.05, dtype=np.float32))
            self.assertTrue(wait_until(lambda: worker.output.available >= 16))
            block = np.full((16, 1), 0.3, dtype=np.float32)
            self.assertTrue(worker.exchange(block, output))
            np.testing.assert_array_equal(output, np.full((16, 1), 0.1, dtype=np.float32))
            self.assertTrue(thread_names)
            self.assertEqual(set(thread_names), {"voice-changer-rvc-inference"})
            self.assertIsNone(worker.error)
        finally:
            worker.close()

    def test_converter_resource_is_closed_after_worker_stops(self):
        class Converter:
            def __init__(self):
                self.closed = 0

            def __call__(self, samples):
                return samples.copy()

            def close(self):
                self.closed += 1

        converter = Converter()
        worker = RvcWorkerProcessor(
            stream_config(), converter, queue_blocks=4, startup_blocks=1,
        )
        worker.start()
        worker.close()
        worker.close()
        self.assertEqual(converter.closed, 1)

    def test_startup_is_silence_then_processing_error_never_falls_back(self):
        entered = threading.Event()
        release = threading.Event()

        def convert(_samples):
            entered.set()
            release.wait(1)
            raise RuntimeError("model failed")

        worker = RvcWorkerProcessor(stream_config(), convert, queue_blocks=4, startup_blocks=1)
        worker.start()
        output = np.full((16, 1), 1, dtype=np.float32)
        try:
            self.assertTrue(worker.exchange(np.ones((16, 1), dtype=np.float32) * 0.2, output))
            np.testing.assert_array_equal(output, 0)
            self.assertTrue(entered.wait(1))
            release.set()
            self.assertTrue(wait_until(lambda: worker.error is not None))
            self.assertFalse(worker.exchange(np.ones((16, 1), dtype=np.float32), output))
            np.testing.assert_array_equal(output, 0)
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                worker.check_health()
        finally:
            release.set()
            worker.close()

    def test_bad_converter_output_is_reported_and_not_played(self):
        worker = RvcWorkerProcessor(
            stream_config(), lambda samples: samples.astype(np.float64),
            queue_blocks=4, startup_blocks=1,
        )
        worker.start()
        output = np.empty((16, 1), dtype=np.float32)
        try:
            self.assertTrue(worker.exchange(np.ones((16, 1), dtype=np.float32) * 0.1, output))
            self.assertTrue(wait_until(lambda: worker.error is not None))
            self.assertFalse(worker.exchange(np.ones((16, 1), dtype=np.float32), output))
            np.testing.assert_array_equal(output, 0)
            self.assertIn("invalid audio", str(worker.error))
        finally:
            worker.close()

    def test_startup_queue_bound_fails_closed(self):
        entered = threading.Event()
        release = threading.Event()

        def convert(samples):
            entered.set()
            release.wait(1)
            return samples.copy()

        worker = RvcWorkerProcessor(
            stream_config(), convert, queue_blocks=2, startup_blocks=1,
        )
        worker.start()
        output = np.empty((16, 1), dtype=np.float32)
        block = np.full((16, 1), 0.1, dtype=np.float32)
        try:
            self.assertTrue(worker.exchange(block, output))
            self.assertTrue(entered.wait(1))
            self.assertTrue(worker.exchange(block, output))
            self.assertFalse(worker.exchange(block, output))
            self.assertIn("startup", str(worker.error))
        finally:
            release.set()
            worker.close()

    def test_lifecycle_and_configuration_validation(self):
        config = CaptureConfig(DeviceRef(0, "test", 0), 48000, 256, 1, 0.9)
        worker = RvcWorkerProcessor(config, lambda samples: samples.copy())
        with self.assertRaises(ValueError):
            RvcWorkerProcessor(stream_config(), lambda samples: samples, queue_blocks=2, startup_blocks=2)
        worker.start()
        with self.assertRaises(RuntimeError):
            worker.start()
        worker.close()
        with self.assertRaises(RuntimeError):
            worker.start()
        with self.assertRaises(ValueError):
            worker.close(timeout=0)

    def test_invalid_callback_blocks_fail_and_zero_output(self):
        worker = RvcWorkerProcessor(stream_config(), lambda samples: samples.copy())
        worker.start()
        output = np.full((16, 1), 1, dtype=np.float32)
        try:
            self.assertFalse(worker.exchange(np.zeros((8, 1), dtype=np.float32), output))
            np.testing.assert_array_equal(output, 0)
            with self.assertRaisesRegex(RuntimeError, "invalid block"):
                worker.check_health()
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
