"""Opt-in fixed-block CPU/MNN synthesis with split NPU F0, not a live backend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .rvc_frontend import RvcOnnxContentEncoder, upsample_content_features


class ExperimentalRvc:
    """Convert isolated 200 ms segments; no streaming-context quality claim."""

    def __init__(self, directory: Path) -> None:
        import MNN
        import onnxruntime as ort
        from rknnlite.api import RKNNLite

        self._mnn = MNN
        self._pitch_runtime = None
        manifest = json.loads((directory / "experiment-model.json").read_text())
        for name, digest in manifest["hashes"].items():
            with (directory / name).open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest() if hasattr(
                    hashlib, "file_digest") else self._hash(source)
            if actual != digest:
                raise ValueError(f"RVC experiment asset hash mismatch: {name}")
        self.manifest = manifest
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")

        def session(name):
            return ort.InferenceSession(
                str(directory / name), options, providers=["CPUExecutionProvider"],
            )

        self._content = RvcOnnxContentEncoder(
            session("content_vec_500.onnx"), input_name="audio", output_name="unit12",
        )
        self._pre = session("pitch-pre.onnx")
        self._post = session("pitch-post.onnx")
        self._upstream = session("upstream-fp32.onnx")
        self._pitch_manifest = json.loads(
            (directory / "pitch-cnn-toolkit232.rknn.json").read_text(),
        )
        config = {"backend": "CPU", "precision": "high", "numThread": 4}
        runtime, supported = MNN.Interpreter.createRuntime((config,))
        if not supported[0]:
            raise RuntimeError("MNN CPU runtime unavailable")
        self._decoder = MNN.Interpreter(str(directory / "decoder-fp32.mnn"))
        self._decoder_session = self._decoder.createSession(config, runtime)
        if self._decoder.getSessionInfo(self._decoder_session, 2) not in (0, 13):
            raise RuntimeError("Experimental synthesis requires MNN CPU high precision")
        pitch_runtime = RKNNLite()
        self._pitch_runtime = pitch_runtime
        try:
            if pitch_runtime.load_rknn(str(directory / "pitch-cnn-toolkit232.rknn")) != 0:
                raise RuntimeError("RVC F0 NPU model load failed")
            if pitch_runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
                raise RuntimeError("RVC F0 NPU initialization failed")
        except Exception:
            self.close()
            raise
        self._rng = np.random.default_rng(20261113)

    @staticmethod
    def _hash(source) -> str:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()

    def synthesize_feed(self, feed: dict[str, np.ndarray]) -> np.ndarray:
        MNN = self._mnn
        hidden, harmonic = self._upstream.run(self.manifest["upstream_outputs"], feed)
        for name, array in (("/Add_1_output_0", hidden), ("source_harmonic", harmonic)):
            tensor = self._decoder.getSessionInput(self._decoder_session, name)
            if tuple(tensor.getShape()) != array.shape:
                raise RuntimeError(f"RVC decoder input shape mismatch: {name}")
            host = MNN.Tensor(
                array.shape, MNN.Halide_Type_Float,
                np.ascontiguousarray(array, dtype=np.float32), MNN.Tensor_DimensionType_Caffe,
            )
            tensor.copyFromHostTensor(host)
        if self._decoder.runSession(self._decoder_session) != 0:
            raise RuntimeError("RVC MNN synthesis failed")
        output = self._decoder.getSessionOutput(self._decoder_session, "audio")
        host = MNN.Tensor(
            output.getShape(), MNN.Halide_Type_Float,
            np.zeros(output.getShape(), dtype=np.float32), MNN.Tensor_DimensionType_Caffe,
        )
        output.copyToHostTensor(host)
        audio = np.array(host.getData(), dtype=np.float32).reshape(-1)
        if audio.shape != (8000,) or not np.isfinite(audio).all():
            raise RuntimeError("RVC synthesis returned invalid audio")
        return audio

    def convert(self, audio: np.ndarray, *, semitones: int = 0) -> tuple[np.ndarray, dict]:
        if (audio.shape != (3200,) or audio.dtype != np.float32
                or not np.isfinite(audio).all() or np.any(np.abs(audio) > 1)):
            raise ValueError("RVC experiment requires normalized 3200-sample mono float32")
        if type(semitones) is not int or not -24 <= semitones <= 24:
            raise ValueError("Experimental pitch shift must be an integer within [-24, 24]")
        start = time.perf_counter()
        features = upsample_content_features(
            self._content.extract(audio), 20, align_to_input_frames=True,
        )
        content_end = time.perf_counter()
        pitch_feed = {"waveform": audio[None], "threshold": np.array([0.3], dtype=np.float32)}
        mel = self._pre.run([self._pitch_manifest["input_name"]], {"waveform": audio[None]})[0]
        convolution = self._pitch_runtime.inference(inputs=[mel], data_format=["nchw"])
        if convolution is None or len(convolution) != 1:
            raise RuntimeError("RVC F0 NPU returned invalid tensors")
        f0 = self._post.run(
            ["pitchf"], {self._pitch_manifest["output_name"]: convolution[0], **pitch_feed},
        )[0][0, :20].copy()
        f0 *= np.float32(2 ** (semitones / 12))
        if f0.shape != (20,) or not np.isfinite(f0).all() or np.any(f0 < 0):
            raise RuntimeError("RVC F0 output is invalid")
        mel_f0 = 1127 * np.log1p(f0.astype(np.float64) / 700)
        low, high = 1127 * np.log1p(np.array([50., 1100.]) / 700)
        coarse = np.rint(np.clip((mel_f0 - low) * 254 / (high - low) + 1, 1, 255)).astype(np.int64)
        pitch_end = time.perf_counter()
        result = self.synthesize_feed({
            "feats": features[None], "pitch": coarse[None], "pitchf": f0[None],
            "latent_noise": self._rng.normal(size=(1, 192, 20)).astype(np.float32),
            "source_noise": self._rng.normal(size=(1, 8000, 1)).astype(np.float32),
        })
        end = time.perf_counter()
        return np.clip(result, -1, 1), {
            "content_ms": (content_end - start) * 1000,
            "f0_ms": (pitch_end - content_end) * 1000,
            "synthesis_ms": (end - pitch_end) * 1000,
            "inference_ms": (end - start) * 1000,
        }

    def close(self) -> None:
        if self._pitch_runtime is not None:
            self._pitch_runtime.release()
            self._pitch_runtime = None
