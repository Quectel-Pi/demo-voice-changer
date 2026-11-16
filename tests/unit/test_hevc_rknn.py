"""Hermetic tests for the HEVC RKNN NPU live path (no rknn, no board, no network).

The client is exercised against a small fake helper process written into a temp
directory; it speaks the same binary protocol as ``scripts/hevc_rknn_worker.py``
but computes a deterministic echo with NumPy. The converter is exercised with a
fake client object, so no subprocess is involved there.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from voice_changer.ai import hevc_rknn
from voice_changer.audio.device_types import DeviceRef, StreamConfig

# The three published HEEVoicechanger ONNX source graphs the manifest may name.
KNOWN_SOURCE_ONNX = (
    "e0084cf7eaa0956f284037dad6ac8344563c936132c47d8ed3cb74202d0633b6",
    "069819e0589fe5bc7253576daa30ab37abf3c2d7bc82f89e08c2d089f3310f55",
    "60d900e8b13523a816bb1b235c9c8614e81f3d0ff96554a9c6ee24d2d2946c84",
)

_FAKE_WORKER = '''\
"""Deterministic stand-in for scripts/hevc_rknn_worker.py; no rknn imports."""
import struct
import sys

import numpy as np

HEADER = struct.Struct("<4sI")
MAGIC = b"HEV1"
READY = b"HVRD"
ERROR = b"HVER"
CHUNK = 3072
WINDOW = 11264


def write_frame(magic, count, payload):
    sys.stdout.buffer.write(HEADER.pack(magic, count))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def write_samples(magic, samples):
    payload = np.asarray(samples, dtype="<f4").tobytes()
    write_frame(magic, len(payload) // 4, payload)


def write_error(message):
    payload = message.encode("utf-8")
    write_frame(ERROR, len(payload), payload)


def read_frame():
    """The length field counts float32 samples, matching HevcRknnClient."""
    header = sys.stdin.buffer.read(HEADER.size)
    if len(header) < HEADER.size:
        return None
    magic, count = HEADER.unpack(header)
    payload = sys.stdin.buffer.read(count * 4)
    if len(payload) < count * 4:
        return None
    return magic, count, payload


mode = sys.argv[1] if len(sys.argv) > 1 else "echo"
if mode == "exit":
    sys.exit(3)
if mode == "startup-error":
    write_error("rknn-toolkit-lite2 is missing")
    sys.exit(2)
if mode == "bad-handshake":
    write_samples(READY, np.zeros(3))
    sys.exit(0)

write_samples(READY, np.zeros(0))
while True:
    frame = read_frame()
    if frame is None:
        break
    magic, count, payload = frame
    if mode == "infer-error":
        write_error("NPU inference failed")
        continue
    window = np.frombuffer(payload, dtype="<f4").astype(np.float32)
    if magic != MAGIC or count != WINDOW or window.size != WINDOW:
        write_error("bad request")
        continue
    write_samples(MAGIC, np.float32(0.5) * window[-CHUNK:])
'''


def valid_manifest(digest, **overrides):
    manifest = {
        "format": hevc_rknn.MANIFEST_FORMAT,
        "target_platform": hevc_rknn.TARGET_PLATFORM,
        "toolkit_version": hevc_rknn.TOOLKIT_VERSION,
        "model_sha256": digest,
        "frames": hevc_rknn.WINDOW_SAMPLES,
        "history": hevc_rknn.HISTORY_SAMPLES,
        "chunk": hevc_rknn.CHUNK_SAMPLES,
        "source_onnx_sha256": KNOWN_SOURCE_ONNX[0],
        "simulator_snr_db": 60.0,
    }
    manifest.update(overrides)
    return manifest


def write_manifest(model, digest, **overrides):
    Path(f"{model}.json").write_text(
        json.dumps(valid_manifest(digest, **overrides)), encoding="utf-8",
    )


def make_package(root, name="female_voice_hfg_new_rknn.rknn",
                 content=b"rknn package fixture"):
    model = Path(root) / name
    model.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    write_manifest(model, digest)
    return model, digest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model, self.digest = make_package(self.root)

    def test_valid_sidecar_is_accepted(self):
        self.assertEqual(hevc_rknn.manifest_path(self.model), Path(f"{self.model}.json"))
        for source in KNOWN_SOURCE_ONNX:
            with self.subTest(source=source):
                write_manifest(self.model, self.digest, source_onnx_sha256=source)
                manifest = hevc_rknn.read_manifest(self.model)
                self.assertEqual(manifest["format"], hevc_rknn.MANIFEST_FORMAT)
                self.assertEqual(manifest["source_onnx_sha256"], source)
                self.assertEqual(manifest["frames"], hevc_rknn.WINDOW_SAMPLES)

    def test_missing_or_broken_sidecars_are_rejected(self):
        plain = self.root / "no-sidecar.rknn"
        plain.write_bytes(b"rknn without sidecar")
        with self.assertRaisesRegex(ValueError, "sidecar manifest"):
            hevc_rknn.read_manifest(plain)

        Path(f"{plain}.json").write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sidecar manifest"):
            hevc_rknn.read_manifest(plain)

        Path(f"{plain}.json").unlink()
        Path(f"{plain}.json").mkdir()
        with self.assertRaisesRegex(ValueError, "sidecar manifest"):
            hevc_rknn.read_manifest(plain)

        Path(f"{plain}.json").rmdir()
        Path(f"{plain}.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            hevc_rknn.read_manifest(plain)

        missing = self.root / "missing.rknn"
        with self.assertRaisesRegex(ValueError, "Cannot access"):
            hevc_rknn.read_manifest(missing)

    def test_symbolic_links_and_directories_are_rejected(self):
        directory = self.root / "directory.rknn"
        directory.mkdir()
        with self.assertRaisesRegex(ValueError, "regular model file"):
            hevc_rknn.read_manifest(directory)

        target = self.root / "target.rknn"
        target.write_bytes(b"rknn target")
        alias = self.root / "alias.rknn"
        alias.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "regular model file"):
            hevc_rknn.read_manifest(alias)

    def test_manifest_identity_rejections(self):
        cases = (
            ({"format": "voice-changer-llvc-rknn-v1"}, "Unsupported RKNN package"),
            ({"target_platform": "rk3588"}, "another platform"),
            ({"toolkit_version": "2.0.0"}, "rknn-toolkit2"),
            ({"model_sha256": "0" * 64}, "SHA-256"),
        )
        for overrides, pattern in cases:
            with self.subTest(overrides=overrides):
                write_manifest(self.model, self.digest, **overrides)
                with self.assertRaisesRegex(ValueError, pattern):
                    hevc_rknn.read_manifest(self.model)

    def test_window_geometry_rejections(self):
        cases = (
            ({"frames": hevc_rknn.WINDOW_SAMPLES - 1}, "window"),
            ({"history": hevc_rknn.HISTORY_SAMPLES // 2}, "window split"),
            ({"chunk": hevc_rknn.CHUNK_SAMPLES // 3}, "window split"),
            ({"history": hevc_rknn.HISTORY_SAMPLES - 1}, "window split"),
        )
        for overrides, pattern in cases:
            with self.subTest(overrides=overrides):
                write_manifest(self.model, self.digest, **overrides)
                with self.assertRaisesRegex(ValueError, pattern):
                    hevc_rknn.read_manifest(self.model)

    def test_source_onnx_must_be_a_known_hevc_graph(self):
        write_manifest(self.model, self.digest, source_onnx_sha256="f" * 64)
        with self.assertRaisesRegex(ValueError, "Unknown source ONNX"):
            hevc_rknn.read_manifest(self.model)

        write_manifest(self.model, self.digest, source_onnx_sha256=None)
        with self.assertRaisesRegex(ValueError, "Unknown source ONNX"):
            hevc_rknn.read_manifest(self.model)

    def test_simulator_accuracy_threshold_is_enforced(self):
        cases = (
            (hevc_rknn.MIN_SIMULATOR_SNR_DB - 0.1, "below 40 dB"),
            (39.0, "below 40 dB"),
            ("60", "below 40 dB"),
            (None, "below 40 dB"),
            (math.nan, "below 40 dB"),
        )
        for value, pattern in cases:
            with self.subTest(value=value):
                write_manifest(self.model, self.digest, simulator_snr_db=value)
                with self.assertRaisesRegex(ValueError, pattern):
                    hevc_rknn.read_manifest(self.model)

        write_manifest(self.model, self.digest,
                       simulator_snr_db=hevc_rknn.MIN_SIMULATOR_SNR_DB)
        self.assertEqual(
            hevc_rknn.read_manifest(self.model)["simulator_snr_db"],
            hevc_rknn.MIN_SIMULATOR_SNR_DB,
        )


class UnavailableReasonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model, self.digest = make_package(self.root)

    def test_no_model_and_invalid_manifest_report_a_reason(self):
        self.assertIn("Select a converted", hevc_rknn.unavailable_reason(None))

        write_manifest(self.model, self.digest, toolkit_version="2.0.0")
        reason = hevc_rknn.unavailable_reason(self.model)
        self.assertIn("rknn-toolkit2", reason)

    def test_missing_worker_python_env_is_named(self):
        with patch.dict(os.environ, {hevc_rknn.WORKER_PYTHON_ENV: ""}):
            reason = hevc_rknn.unavailable_reason(self.model)
        self.assertIn(hevc_rknn.WORKER_PYTHON_ENV, reason)

        with patch.dict(os.environ, {}):
            os.environ.pop(hevc_rknn.WORKER_PYTHON_ENV, None)
            reason = hevc_rknn.unavailable_reason(self.model)
        self.assertIn(hevc_rknn.WORKER_PYTHON_ENV, reason)

    def test_complete_prerequisites_make_the_package_available(self):
        with patch.dict(os.environ, {hevc_rknn.WORKER_PYTHON_ENV: sys.executable}), \
                patch.object(hevc_rknn, "worker_script", return_value=Path(sys.executable)):
            self.assertIsNone(hevc_rknn.unavailable_reason(self.model))

    def test_missing_helper_script_or_interpreter_is_reported(self):
        with patch.dict(os.environ, {hevc_rknn.WORKER_PYTHON_ENV: sys.executable}), \
                patch.object(hevc_rknn, "worker_script",
                             return_value=self.root / "missing_worker.py"):
            reason = hevc_rknn.unavailable_reason(self.model)
        self.assertIn("Missing NPU helper script", reason)

        with patch.dict(os.environ, {
            hevc_rknn.WORKER_PYTHON_ENV: str(self.root / "missing-python"),
        }), patch.object(hevc_rknn, "worker_script", return_value=Path(sys.executable)):
            reason = hevc_rknn.unavailable_reason(self.model)
        self.assertIn("interpreter does not exist", reason)

    def test_worker_python_defaults_to_this_interpreter(self):
        with patch.dict(os.environ, {hevc_rknn.WORKER_PYTHON_ENV: ""}):
            self.assertEqual(hevc_rknn.worker_python(), sys.executable)
        with patch.dict(os.environ, {hevc_rknn.WORKER_PYTHON_ENV: f"  {sys.executable}  "}):
            self.assertEqual(hevc_rknn.worker_python(), sys.executable)


class HevcRknnClientTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model, self.digest = make_package(self.root)
        self.worker = self.root / "fake_hevc_worker.py"
        self.worker.write_text(textwrap.dedent(_FAKE_WORKER), encoding="utf-8")
        self._processes = []

    def tearDown(self):
        # HevcRknnClient.close() never closes the helper's stdout pipe, so the
        # tests close it here to keep the suite free of ResourceWarnings (the
        # leak itself is reported as a product issue).
        for process in self._processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5.0)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()

    def _client(self, mode="echo"):
        real_popen = hevc_rknn.subprocess.Popen

        def recording_popen(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            self._processes.append(process)
            return process

        with patch.object(hevc_rknn.subprocess, "Popen", side_effect=recording_popen):
            return hevc_rknn.HevcRknnClient(
                self.model,
                command=[sys.executable, str(self.worker), mode],
                startup_timeout=10.0,
                inference_timeout=10.0,
            )

    def test_echo_helper_round_trip_and_idempotent_close(self):
        window = np.linspace(-1.0, 1.0, hevc_rknn.WINDOW_SAMPLES, dtype=np.float32)
        client = self._client()
        try:
            output = client.infer(window)
            self.assertEqual(output.shape, (hevc_rknn.CHUNK_SAMPLES,))
            self.assertEqual(output.dtype, np.float32)
            self.assertTrue(np.isfinite(output).all())
            np.testing.assert_allclose(
                output, 0.5 * window[-hevc_rknn.CHUNK_SAMPLES:], rtol=0, atol=1e-6,
            )
            reversed_output = client.infer(window[::-1].copy())
            np.testing.assert_allclose(
                reversed_output, 0.5 * window[:hevc_rknn.CHUNK_SAMPLES][::-1],
                rtol=0, atol=1e-6,
            )
        finally:
            client.close()
        client.close()
        with self.assertRaisesRegex(RuntimeError, "not running"):
            client.infer(window)

    def test_error_frame_raises_runtime_error(self):
        client = self._client("infer-error")
        try:
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                client.infer(np.zeros(hevc_rknn.WINDOW_SAMPLES, dtype=np.float32))
        finally:
            client.close()

    def test_startup_failures_raise_runtime_error(self):
        cases = (
            ("exit", "exited"),
            ("startup-error", "rknn-toolkit-lite2 is missing"),
            ("bad-handshake", "handshake"),
        )
        for mode, pattern in cases:
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, pattern):
                self._client(mode)

    def test_window_validation_happens_before_the_helper_sees_it(self):
        client = self._client()
        try:
            with self.assertRaisesRegex(ValueError, "11264"):
                client.infer(np.zeros(hevc_rknn.WINDOW_SAMPLES - 1, dtype=np.float32))
            non_finite = np.zeros(hevc_rknn.WINDOW_SAMPLES, dtype=np.float32)
            non_finite[10] = np.inf
            with self.assertRaisesRegex(ValueError, "non-finite"):
                client.infer(non_finite)
        finally:
            client.close()


class _FakeConverterClient:
    """Stands in for ``HevcRknnClient`` without starting a subprocess."""

    def __init__(self):
        self.windows = []
        self.closed = 0

    def infer(self, window):
        self.windows.append(np.array(window, copy=True))
        return (np.float32(0.5) * window[-hevc_rknn.CHUNK_SAMPLES:]).astype(np.float32)

    def close(self):
        self.closed += 1


def stream_config(blocksize=hevc_rknn.CONVERSION_FRAMES, samplerate=48000, channels=1):
    device = DeviceRef(0, "test device", 0)
    return StreamConfig(device, device, samplerate, blocksize, channels, 0.5)


class HevcRknnConverterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "female_voice_hfg_new_rknn.rknn"
        self.model.write_bytes(b"rknn fixture")

    def test_three_device_blocks_use_full_windows_without_discontinuities(self):
        frames = hevc_rknn.CONVERSION_FRAMES
        client = _FakeConverterClient()
        converter = hevc_rknn.HevcRknnConverter(
            stream_config(), self.model, client=client,
        )
        # A linear ramp makes any dropped or duplicated sample visible as a step
        # in the concatenated NPU chunks.
        total = frames * 4
        signal = np.linspace(-1.0, 1.0, total, dtype=np.float32)
        for index in range(3):
            block = signal[index * frames:(index + 1) * frames, None]
            output = converter(block)
            self.assertEqual(output.shape, (frames, 1))
            self.assertEqual(output.dtype, np.float32)
            self.assertTrue(np.isfinite(output).all())
            self.assertLessEqual(float(np.abs(output).max()), 1.0)
        # The 48 kHz -> 16 kHz filter needs 63 samples of context, so the first
        # block leaves 3046 pending samples: three device blocks feed two full
        # NPU windows.
        self.assertEqual(len(client.windows), 2)
        self._assert_continuous_windows(client.windows)

        converter(signal[3 * frames:4 * frames, None])
        self.assertGreaterEqual(len(client.windows), 3)
        self._assert_continuous_windows(client.windows)
        self.assertEqual(converter.client, client)

    def _assert_continuous_windows(self, windows):
        for window in windows:
            self.assertEqual(window.size, hevc_rknn.WINDOW_SAMPLES)
            self.assertEqual(window.dtype, np.float32)
            self.assertTrue(np.isfinite(window).all())
        for previous, current in zip(windows, windows[1:]):
            # The rolling history is carried over exactly: no seam in the input.
            np.testing.assert_array_equal(
                current[:hevc_rknn.HISTORY_SAMPLES],
                previous[hevc_rknn.CHUNK_SAMPLES:],
            )
        chunks = np.concatenate(
            [window[-hevc_rknn.CHUNK_SAMPLES:] for window in windows]
        )
        steps = np.diff(chunks)
        self.assertGreater(float(steps.min()), 0.0)
        typical = float(np.median(steps))
        for seam in range(1, len(windows)):
            step = float(
                chunks[seam * hevc_rknn.CHUNK_SAMPLES]
                - chunks[seam * hevc_rknn.CHUNK_SAMPLES - 1]
            )
            self.assertAlmostEqual(step / typical, 1.0, delta=0.5)

    def test_converter_rejects_wrong_device_configuration(self):
        for config in (
            stream_config(samplerate=44100),
            stream_config(blocksize=1024),
            stream_config(channels=3),
        ):
            with self.subTest(config=config), self.assertRaisesRegex(
                ValueError, r"48000 Hz, 9216",
            ):
                hevc_rknn.HevcRknnConverter(
                    config, self.model, client=_FakeConverterClient(),
                )

    def test_converter_rejects_invalid_device_audio(self):
        frames = hevc_rknn.CONVERSION_FRAMES
        client = _FakeConverterClient()
        converter = hevc_rknn.HevcRknnConverter(
            stream_config(), self.model, client=client,
        )
        nan_block = np.zeros((frames, 1), dtype=np.float32)
        nan_block[0, 0] = math.nan
        bad_blocks = (
            np.zeros((frames - 1, 1), dtype=np.float32),
            np.zeros((frames, 2), dtype=np.float32),
            np.zeros((frames, 1), dtype=np.float64),
            nan_block,
        )
        for block in bad_blocks:
            with self.subTest(shape=block.shape, dtype=block.dtype):
                with self.assertRaises(ValueError):
                    converter(block)
        self.assertEqual(client.windows, [])

    def test_close_delegates_to_the_client(self):
        client = _FakeConverterClient()
        converter = hevc_rknn.HevcRknnConverter(
            stream_config(), self.model, client=client,
        )
        converter.close()
        self.assertEqual(client.closed, 1)


class HevcRknnProcessorFactoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model = Path(temporary.name) / "female_voice_hfg_new_rknn.rknn"
        self.model.write_bytes(b"rknn fixture")

    def test_factory_upgrades_the_block_size_and_returns_a_worker(self):
        from voice_changer.ai.rvc_worker import RvcWorkerProcessor

        calls = []

        class RecordingConverter:
            def __init__(self, config, model_path):
                calls.append((config, model_path))
                self.samples = []

            def __call__(self, block):
                self.samples.append(block.shape)
                return np.zeros_like(block)

            def close(self):
                pass

        factory = hevc_rknn.HevcRknnProcessorFactory(self.model)
        self.assertEqual(factory.model_path, self.model)
        device_config = stream_config(blocksize=1024)
        with patch.object(hevc_rknn, "HevcRknnConverter", RecordingConverter):
            worker = factory(device_config, object())
        self.assertIsInstance(worker, RvcWorkerProcessor)
        self.assertEqual(worker.config, device_config)
        self.assertEqual(worker.config.blocksize, 1024)
        self.assertEqual(worker.block_frames, 1024)
        self.assertEqual(worker.conversion_frames, hevc_rknn.CONVERSION_FRAMES)
        self.assertEqual(worker.conversion_frames, 9216)
        self.assertEqual(len(calls), 1)
        conversion_config, model_path = calls[0]
        self.assertEqual(conversion_config.samplerate, 48000)
        self.assertEqual(conversion_config.blocksize, hevc_rknn.CONVERSION_FRAMES)
        self.assertEqual(conversion_config.channels, 1)
        self.assertEqual(model_path, self.model)
        worker.close()

    def test_factory_rejects_configurations_other_than_48k_1024(self):
        factory = hevc_rknn.HevcRknnProcessorFactory(self.model)
        for config in (stream_config(blocksize=1024, samplerate=44100),
                       stream_config(blocksize=2048)):
            with self.subTest(config=config), self.assertRaisesRegex(
                ValueError, r"48000 Hz and 1024",
            ):
                factory(config, object())


if __name__ == "__main__":
    unittest.main()
