import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from voice_changer.ai.llvc import (
    INPUTS, OUTPUTS, MODEL_SHA256, LlvcConverter, configured_model, unavailable_reason,
    LlvcProcessorFactory, inspect_local_onnx_model,
    STABLE_MODEL_SHA256,
)
from voice_changer.audio.device_types import DeviceRef, StreamConfig


def config(channels=1):
    device = DeviceRef(0, "test", 0)
    return StreamConfig(device, device, 48000, 2496, channels, 0.5)


class Session:
    frames = 832
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.invalid = False

    def get_inputs(self):
        return [
            SimpleNamespace(name=name, type="tensor(float)",
                            shape=(
                                [1, 1, self.frames + 32],
                                [1, 512, 510],
                                [1, 2, 13, 256],
                                [1, 512, 4],
                                [1, 1, 24],
                            )[i])
            for i, name in enumerate(INPUTS)
        ]

    def get_outputs(self):
        return [
            SimpleNamespace(name=name, type="tensor(float)",
                            shape=(
                                [1, 1, self.frames],
                                [1, 512, 510],
                                [1, 2, 13, 256],
                                [1, 512, 4],
                                [1, 1, 24],
                            )[i])
            for i, name in enumerate(OUTPUTS)
        ]

    def run(self, names, feed):
        self.calls.append({name: value.copy() for name, value in feed.items()})
        audio = feed["audio"][:, :, -self.frames:] * np.float32(0.5)
        if self.invalid:
            audio.fill(np.nan)
        return [audio, *[feed[name] + np.float32(1) for name in INPUTS[1:]]]


class LlvcTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = Path(temp.name) / "model.onnx"
        self.model.write_bytes(b"test")
        runtime = SimpleNamespace(
            SessionOptions=SimpleNamespace, InferenceSession=Session,
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        )
        self.runtime_patch = patch.dict(sys.modules, {"onnxruntime": runtime})
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)
        self.hash_patch = patch.object(LlvcConverter, "_hash", return_value=MODEL_SHA256)
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)

    def test_device_rate_stream_is_bounded_and_keeps_context_and_states(self):
        converter = LlvcConverter(config(2), self.model)
        nonzero = False
        for i in range(120):
            t = (np.arange(2496) + i * 2496) / 48000
            samples = np.repeat((0.2 * np.sin(2*np.pi*180*t)).astype(np.float32)[:, None], 2, axis=1)
            output = converter(samples)
            self.assertEqual(output.shape, samples.shape)
            self.assertEqual(output.dtype, np.float32)
            self.assertTrue(np.isfinite(output).all())
            np.testing.assert_array_equal(output[:, 0], output[:, 1])
            nonzero |= bool(np.any(output))
            self.assertLess(converter.pending.size, 832)
            self.assertLessEqual(converter.ready.size, 4992)
        self.assertTrue(nonzero)
        calls = converter.session.calls
        self.assertEqual(len(calls), 119)
        np.testing.assert_array_equal(calls[0]["audio"][0, 0, :32], 0)
        np.testing.assert_array_equal(calls[1]["audio"][0, 0, :32], calls[0]["audio"][0, 0, -32:])
        for name in INPUTS[1:]:
            np.testing.assert_array_equal(calls[1][name], calls[0][name] + 1)

    def test_new_session_resets_audio_and_every_state(self):
        a = LlvcConverter(config(), self.model)
        b = LlvcConverter(config(), self.model)
        samples = np.full((2496, 1), 0.1, dtype=np.float32)
        for _ in range(8):
            np.testing.assert_array_equal(a(samples), b(samples))
        for left, right in zip(a.state, b.state):
            np.testing.assert_array_equal(left, right)

    def test_stable_graph_uses_104ms_model_blocks_and_1024_device_callbacks(self):
        from dataclasses import replace
        with patch.object(Session, "frames", 1664), patch.object(
            LlvcConverter, "_hash", return_value=STABLE_MODEL_SHA256,
        ):
            worker = LlvcProcessorFactory(self.model)(replace(config(), blocksize=1024), None)
            self.assertEqual(worker.block_frames, 1024)
            self.assertEqual(worker.conversion_frames, 4992)
            self.assertEqual(worker.convert.config.blocksize, 4992)
            self.assertEqual(worker.convert.model_frames, 1664)
            samples = np.full((4992, 1), 0.1, dtype=np.float32)
            heard = False
            for _ in range(12):
                output = worker.convert(samples)
                self.assertEqual(output.shape, samples.shape)
                heard |= bool(np.any(output))
            self.assertTrue(heard)
            self.assertLess(worker.convert.pending.size, 1664)
            self.assertEqual(worker.input.capacity, 4992*8)

    def test_invalid_model_output_is_not_passthrough(self):
        converter = LlvcConverter(config(), self.model)
        converter.session.invalid = True
        samples = np.full((2496, 1), 0.1, dtype=np.float32)
        converter(samples)
        with self.assertRaisesRegex(RuntimeError, "invalid audio"):
            converter(samples)

    def test_wrong_hash_and_device_format_fail_explicitly(self):
        with patch.object(LlvcConverter, "_hash", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                LlvcConverter(config(), self.model)
        device = DeviceRef(0, "test", 0)
        with self.assertRaisesRegex(ValueError, "2496"):
            LlvcConverter(StreamConfig(device, device, 48000, 512, 1, 0.5), self.model)

    def test_schema_mismatch_is_rejected(self):
        wrong = Session().get_inputs()
        wrong[1].name = "missing_state"
        with patch.object(Session, "get_inputs", return_value=wrong):
            with self.assertRaisesRegex(ValueError, "schema"):
                LlvcConverter(config(), self.model)

    def test_missing_state_output_is_rejected(self):
        converter = LlvcConverter(config(), self.model)
        samples = np.zeros((2496, 1), dtype=np.float32)
        converter(samples)
        with patch.object(converter.session, "run", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "omitted"):
                converter(samples)

    def test_incomplete_output_shape_metadata_is_rejected(self):
        outputs = Session().get_outputs()
        for item in outputs:
            item.shape = ["unknown"]
        with patch.object(Session, "get_outputs", return_value=outputs):
            with self.assertRaisesRegex(ValueError, "fixed-shape"):
                LlvcConverter(config(), self.model)

    def test_custom_onnx_must_match_exact_state_shapes(self):
        wrong = Session().get_inputs()
        wrong[2].shape = [1, 2, 13, 128]
        with patch.object(Session, "get_inputs", return_value=wrong):
            with self.assertRaisesRegex(ValueError, "fixed audio/state contract"):
                LlvcConverter(config(), self.model, allow_custom_model=True)

    def test_custom_onnx_must_match_every_state_output_shape(self):
        wrong = Session().get_outputs()
        wrong[3].shape = [1, 512, 3]
        with patch.object(Session, "get_outputs", return_value=wrong):
            with self.assertRaisesRegex(ValueError, "fixed audio/state contract"):
                LlvcConverter(config(), self.model, allow_custom_model=True)

    def test_local_custom_model_is_inspected_without_hash_allowlisting(self):
        with patch.object(LlvcConverter, "_hash", return_value="custom-model"):
            self.assertEqual(
                LlvcConverter(config(), self.model, allow_custom_model=True).model_frames,
                832,
            )
        self.assertEqual(inspect_local_onnx_model(self.model), 832)

    def test_local_model_symlinks_are_rejected(self):
        alias = self.model.with_name("alias.onnx")
        alias.symlink_to(self.model)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            inspect_local_onnx_model(alias)

    def test_configuration_never_uses_an_implicit_model(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(configured_model())
            self.assertIn("VOICE_CHANGER_LLVC_MODEL", unavailable_reason())
        with patch.dict(os.environ, {"VOICE_CHANGER_LLVC_MODEL": str(self.model / "missing")}):
            self.assertIn("Model not found", unavailable_reason())

    def test_product_factory_runs_through_callback_worker_and_output_ceiling(self):
        from test_sounddevice_backend import FakeSoundDevice
        from voice_changer.audio.sounddevice_backend import SoundDeviceBackend

        library = FakeSoundDevice()
        stream_config = StreamConfig(
            DeviceRef(3, "selected input", 1), DeviceRef(7, "selected output", 1),
            48000, 2496, 1, 0.05,
        )
        with patch.dict(sys.modules, {"sounddevice": library}):
            stream = SoundDeviceBackend(
                stream_config, output_confirmed=True,
                processor_factory=LlvcProcessorFactory(self.model),
            ).open_stream()
            stream.start()
            try:
                samples = np.full((2496, 1), 0.2, dtype=np.float32)
                output = np.empty_like(samples)
                heard = False
                for i in range(10):
                    library.streams[0].options["callback"](samples, output, 2496, None, False)
                    self.assertLessEqual(float(np.abs(output).max()), 0.05)
                    heard |= bool(np.any(output))
                    deadline = time.monotonic() + 1
                    while stream._processor._processed < (i+1)*2496 and time.monotonic() < deadline:
                        time.sleep(0.001)
                    self.assertEqual(stream._processor._processed, (i+1)*2496)
                    stream.check_health()
                self.assertTrue(heard)
            finally:
                stream.stop()
                stream.close()


if __name__ == "__main__":
    unittest.main()
