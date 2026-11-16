"""CPU-only RVC content and F0 frontends with explicit ONNX contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .rvc_onnx import (
    SynthesisSession,
    TensorMetadata,
    RvcOnnxSynthesizer,
    load_cpu_session,
)


class FrontendSession(Protocol):
    def get_inputs(self) -> list[TensorMetadata]: ...

    def get_outputs(self) -> list[TensorMetadata]: ...

    def get_providers(self) -> list[str]: ...

    def run(self, output_names: list[str],
            input_feed: dict[str, np.ndarray]) -> list[np.ndarray]: ...


class RvcSynthesizer(Protocol):
    def synthesize(
        self,
        features: np.ndarray,
        pitch: np.ndarray,
        pitch_hz: np.ndarray,
        *,
        speaker_id: int = 0,
    ) -> np.ndarray: ...


class RvcOnnxContentEncoder:
    """Extract batch-one HuBERT/ContentVec frame features from 16 kHz audio."""

    def __init__(self, session: FrontendSession, *,
                 input_name: str = "source", output_name: str | None = None) -> None:
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("RVC content extraction requires CPUExecutionProvider only")
        inputs = list(session.get_inputs())
        self._inputs = {item.name: item for item in inputs}
        if input_name not in ("source", "audio"):
            raise ValueError("Content input must be explicitly selected as source or audio")
        required = {input_name}
        if set(self._inputs) not in (required, required | {"padding_mask"}):
            raise ValueError(f"Unsupported content model inputs; expected {input_name} and optional padding_mask")
        source = self._inputs[input_name]
        if source.type != "tensor(float)" or len(source.shape) != 2:
            raise ValueError("Content model source must be float32 [1, samples]")
        if isinstance(source.shape[0], int) and source.shape[0] != 1:
            raise ValueError("Content model source batch size must be one")
        padding = self._inputs.get("padding_mask")
        if padding is not None and (
            padding.type != "tensor(bool)" or len(padding.shape) != 2
            or isinstance(padding.shape[0], int) and padding.shape[0] != 1
        ):
            raise ValueError("Content model padding_mask must be bool [1, samples]")
        outputs = list(session.get_outputs())
        if output_name is None:
            if len(outputs) != 1:
                raise ValueError("Content model must expose one output or explicitly select a layer")
            output = outputs[0]
        else:
            selected = [item for item in outputs if item.name == output_name]
            if len(selected) != 1:
                raise ValueError(f"Content model has no unique output named {output_name}")
            output = selected[0]
        if output.type != "tensor(float)" or len(output.shape) != 3:
            raise ValueError("Content model output must be float32 [1, frames, channels]")
        if isinstance(output.shape[0], int) and output.shape[0] != 1:
            raise ValueError("Content model output batch size must be one")
        channels = output.shape[-1]
        if isinstance(channels, int) and channels not in (256, 768):
            raise ValueError("RVC content model must output 256 or 768 channels")
        self._output = output
        self._input_name = input_name
        self._session = session

    @classmethod
    def load(cls, model_file: Path, *, threads: int = 2, input_name: str = "source",
             output_name: str | None = None) -> RvcOnnxContentEncoder:
        return cls(load_cpu_session(model_file, threads), input_name=input_name,
                   output_name=output_name)

    def extract(self, audio_16k: np.ndarray) -> np.ndarray:
        if (not isinstance(audio_16k, np.ndarray) or audio_16k.ndim != 1
                or audio_16k.dtype != np.float32 or audio_16k.size == 0
                or not np.isfinite(audio_16k).all()):
            raise ValueError("Content input must be nonempty finite mono float32 at 16 kHz")
        if np.any(np.abs(audio_16k) > 1):
            raise ValueError("Content input must be normalized to [-1, 1]")
        waveform = np.ascontiguousarray(audio_16k[None])
        feed = {self._input_name: waveform}
        if "padding_mask" in self._inputs:
            feed["padding_mask"] = np.zeros(waveform.shape, dtype=np.bool_)
        for name, value in feed.items():
            for expected, actual in zip(self._inputs[name].shape, value.shape):
                if isinstance(expected, int) and expected != actual:
                    raise ValueError(f"Content model static dimension mismatch for {name}")
        result = self._session.run([self._output.name], feed)
        if len(result) != 1 or not isinstance(result[0], np.ndarray):
            raise RuntimeError("Content model returned no single feature array")
        features = result[0]
        if (features.dtype != np.float32 or features.ndim != 3 or features.shape[0] != 1
                or features.shape[1] == 0 or features.shape[2] not in (256, 768)
                or not np.isfinite(features).all()):
            raise RuntimeError("Content model returned incompatible or non-finite features")
        if isinstance(self._output.shape[-1], int) and features.shape[-1] != self._output.shape[-1]:
            raise RuntimeError("Content model output channel count changed")
        return np.ascontiguousarray(features[0])


class RvcOnnxPitchExtractor:
    """RMVPE ONNX adapter returning 100 Hz coarse bins and continuous F0."""

    _F0_MIN = 50.0
    _F0_MAX = 1100.0
    _THRESHOLD = 0.3
    _HOP = 160

    def __init__(self, session: FrontendSession) -> None:
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("RVC pitch extraction requires CPUExecutionProvider only")
        inputs = list(session.get_inputs())
        self._inputs = {item.name: item for item in inputs}
        if len(inputs) != 2 or set(self._inputs) != {"waveform", "threshold"}:
            raise ValueError("Unsupported RMVPE ONNX inputs; expected waveform and threshold")
        waveform = self._inputs["waveform"]
        threshold = self._inputs["threshold"]
        if waveform.type != "tensor(float)" or len(waveform.shape) != 2:
            raise ValueError("RMVPE waveform must be float32 [1, samples]")
        if isinstance(waveform.shape[0], int) and waveform.shape[0] != 1:
            raise ValueError("RMVPE waveform batch size must be one")
        if threshold.type != "tensor(float)" or len(threshold.shape) != 1:
            raise ValueError("RMVPE threshold must be float32 [1]")
        if isinstance(threshold.shape[0], int) and threshold.shape[0] != 1:
            raise ValueError("RMVPE threshold length must be one")
        outputs = list(session.get_outputs())
        if len(outputs) != 1 or outputs[0].name != "pitchf":
            raise ValueError("Unsupported RMVPE output; expected pitchf")
        output = outputs[0]
        if output.type != "tensor(float)" or len(output.shape) not in (1, 2):
            raise ValueError("RMVPE pitchf output must be float32 [frames] or [1, frames]")
        if len(output.shape) == 2 and isinstance(output.shape[0], int) and output.shape[0] != 1:
            raise ValueError("RMVPE pitchf output batch size must be one")
        self._output = output
        self._session = session

    @classmethod
    def load(cls, model_file: Path, *, threads: int = 2) -> RvcOnnxPitchExtractor:
        return cls(load_cpu_session(model_file, threads))

    def extract(self, audio_16k: np.ndarray, *, semitones: int = 0) -> tuple[np.ndarray, np.ndarray]:
        if (not isinstance(audio_16k, np.ndarray) or audio_16k.ndim != 1
                or audio_16k.dtype != np.float32 or audio_16k.size < self._HOP
                or audio_16k.size % self._HOP != 0
                or not np.isfinite(audio_16k).all()
                or np.any(np.abs(audio_16k) > 1)):
            raise ValueError("RMVPE input must be finite normalized mono float32, aligned to 160 samples")
        if isinstance(semitones, bool) or not isinstance(semitones, int) or not -48 <= semitones <= 48:
            raise ValueError("Pitch shift must be an integer from -48 to 48 semitones")
        frames = audio_16k.size // self._HOP
        waveform = np.ascontiguousarray(audio_16k[None])
        feed = {
            "waveform": waveform,
            "threshold": np.array([self._THRESHOLD], dtype=np.float32),
        }
        for name, value in feed.items():
            for expected, actual in zip(self._inputs[name].shape, value.shape):
                if isinstance(expected, int) and expected != actual:
                    raise ValueError(f"RMVPE static dimension mismatch for {name}")
        result = self._session.run(["pitchf"], feed)
        if len(result) != 1 or not isinstance(result[0], np.ndarray):
            raise RuntimeError("RMVPE returned no pitchf array")
        raw = result[0]
        if raw.ndim == 2 and raw.shape[0] == 1:
            raw = raw[0]
        if (raw.ndim != 1 or raw.size == 0 or raw.dtype != np.float32
                or not np.isfinite(raw).all() or np.any(raw < 0)):
            raise RuntimeError("RMVPE returned empty, non-finite or invalid F0")
        continuous = np.zeros(frames, dtype=np.float32)
        count = min(frames, raw.size)
        shifted = raw[:count].astype(np.float64) * (2.0 ** (semitones / 12.0))
        if (not np.isfinite(shifted).all()
                or np.any(shifted > np.finfo(np.float32).max)):
            raise RuntimeError("Pitch shift produced non-finite F0")
        continuous[-count:] = shifted.astype(np.float32)
        mel = 1127.0 * np.log1p(continuous.astype(np.float64) / 700.0)
        mel_min = 1127.0 * np.log1p(self._F0_MIN / 700.0)
        mel_max = 1127.0 * np.log1p(self._F0_MAX / 700.0)
        coarse = np.rint(np.clip((mel - mel_min) * 254.0 / (mel_max - mel_min) + 1.0, 1, 255))
        return coarse.astype(np.int64), continuous


def upsample_content_features(features: np.ndarray, target_frames: int, *,
                              align_to_input_frames: bool = False) -> np.ndarray:
    """Match RVC's 50 Hz hidden states to its 100 Hz F0/inference frame grid."""
    if (not isinstance(features, np.ndarray) or features.ndim != 2
            or features.dtype != np.float32 or features.shape[0] == 0
            or features.shape[1] not in (256, 768) or not np.isfinite(features).all()):
        raise ValueError("RVC content features must be finite float32 [frames, 256 or 768]")
    if type(align_to_input_frames) is not bool:
        raise ValueError("align_to_input_frames must be a boolean")
    if (type(target_frames) is not int or target_frames < 1
            or features.shape[0] * 2 + (2 if align_to_input_frames else 0) < target_frames):
        raise ValueError("Content model produced too few frames for the requested 100 Hz grid")
    output_frames = target_frames if align_to_input_frames else features.shape[0] * 2
    scale = features.shape[0] / output_frames
    source_positions = np.clip(
        (np.arange(output_frames, dtype=np.float64) + 0.5) * scale - 0.5,
        0, features.shape[0] - 1,
    )
    left = np.floor(source_positions).astype(np.intp)
    right = np.minimum(left + 1, features.shape[0] - 1)
    weight = (source_positions - left).astype(np.float32)[:, None]
    interpolated = features[left] * (1.0 - weight) + features[right] * weight
    return np.ascontiguousarray(interpolated[-target_frames:])


class RvcOnnxFrontEnd:
    """Produce aligned content, coarse F0 and continuous F0 without synthesis."""

    def __init__(self, content: RvcOnnxContentEncoder, pitch: RvcOnnxPitchExtractor, *,
                 align_to_input_frames: bool = False) -> None:
        if type(align_to_input_frames) is not bool:
            raise ValueError("align_to_input_frames must be a boolean")
        self.content = content
        self.pitch = pitch
        self.align_to_input_frames = align_to_input_frames

    def extract(self, audio_16k: np.ndarray, *, semitones: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (not isinstance(audio_16k, np.ndarray) or audio_16k.ndim != 1
                or audio_16k.dtype != np.float32 or audio_16k.size == 0
                or audio_16k.size % RvcOnnxPitchExtractor._HOP != 0):
            raise ValueError("RVC frontend audio must be mono float32 and aligned to 160 samples")
        if (isinstance(semitones, bool) or not isinstance(semitones, int)
                or not -48 <= semitones <= 48):
            raise ValueError("Pitch shift must be an integer from -48 to 48 semitones")
        features = self.content.extract(audio_16k)
        frames = audio_16k.size // RvcOnnxPitchExtractor._HOP
        aligned_features = upsample_content_features(
            features, frames, align_to_input_frames=self.align_to_input_frames,
        )
        coarse, continuous = self.pitch.extract(audio_16k, semitones=semitones)
        if len(aligned_features) != len(coarse) or len(coarse) != len(continuous):
            raise RuntimeError("RVC content and F0 frame grids do not match")
        return aligned_features, coarse, continuous


class RvcOnnxPipeline:
    """Convert one aligned 16 kHz mono chunk to model-rate mono audio."""

    def __init__(
        self,
        frontend: RvcOnnxFrontEnd,
        synthesizer: RvcSynthesizer,
        *,
        model_sample_rate: int,
    ) -> None:
        if type(model_sample_rate) is not int or model_sample_rate not in (32000, 40000, 48000):
            raise ValueError("RVC v1/v2 output rate must be 32000, 40000 or 48000 Hz")
        self.frontend = frontend
        self.synthesizer = synthesizer
        self.model_sample_rate = model_sample_rate

    def convert(
        self, audio_16k: np.ndarray, *, semitones: int = 0, speaker_id: int = 0,
    ) -> np.ndarray:
        features, coarse, continuous = self.frontend.extract(
            audio_16k, semitones=semitones,
        )
        audio = self.synthesizer.synthesize(
            features, coarse, continuous, speaker_id=speaker_id,
        )
        expected_samples = len(coarse) * (self.model_sample_rate // 100)
        if len(audio) != expected_samples:
            raise RuntimeError(
                f"RVC model returned {len(audio)} samples; expected {expected_samples} "
                f"for {len(coarse)} frames at {self.model_sample_rate} Hz"
            )
        return audio
