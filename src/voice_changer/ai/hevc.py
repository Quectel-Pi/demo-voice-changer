"""Public 16 kHz causal ONNX voice models (HEVoicechanger family): identify only.

The reference engine streams these graphs by prepending a rolling history window
of already-played samples to each 128 ms chunk. On the Pi M2 CPU the measured
cost is about 415 ms per 128 ms chunk (RTF ~3.25), so these models cannot run in
the live conversion chain here. This module validates the graph interface and
reports the fixed measurement; it never runs inference and never starts audio.
"""

from __future__ import annotations

INPUT_NAME = "audio_in"
OUTPUT_NAME = "audio_out"
SAMPLE_RATE = 16000
HISTORY_SAMPLES = 16000
CHUNK_SAMPLES = 2048
CHUNK_MS = 128
MEASURED_CHUNK_MS = 415.5
MEASURED_RTF = 3.25
LIVE_REASON = (
    "实验模型：板端 CPU 实测每 128 ms 分块约需 415 ms（RTF≈3.25），无法实时；"
    "页面只做识别、命名与离线试听，不会加载到实时链，开始按钮保持禁用。 / "
    "Experimental: about 415 ms per 128 ms chunk on the Pi M2 CPU (RTF≈3.25); "
    "identified, named and offline-previewed only, never opened in the live chain; "
    "Start stays disabled."
)


def _is_dynamic(size) -> bool:
    return isinstance(size, str) or size is None or (type(size) is int and size < 0)


def _schema(item) -> tuple:
    return tuple(None if _is_dynamic(size) else size for size in item.shape)


def validate_hevc_session(session) -> None:
    """Accept one float32 audio_in/audio_out waveform pair, or raise ValueError."""
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("HEVoicechanger models must expose exactly one audio input and output")
    source, sink = inputs[0], outputs[0]
    if source.name != INPUT_NAME or sink.name != OUTPUT_NAME:
        raise ValueError("HEVoicechanger model audio tensor names mismatch")
    if source.type != "tensor(float)" or sink.type != "tensor(float)":
        raise ValueError("HEVoicechanger model requires float32 audio tensors")

    source_shape = _schema(source)
    if len(source_shape) != 3 or source_shape[0] not in (1, None) or source_shape[1] != 1:
        raise ValueError("HEVoicechanger model input must be [1, 1, samples] mono audio")
    if not _is_dynamic(source.shape[-1]) and source.shape[-1] < CHUNK_SAMPLES:
        raise ValueError("HEVoicechanger model input is shorter than one 128 ms chunk")

    sink_shape = _schema(sink)
    if len(sink_shape) != 3 or sink_shape[0] not in (1, None) or sink_shape[1] != 1:
        raise ValueError("HEVoicechanger model output must be [samples, 1] mono audio")
    if not _is_dynamic(sink.shape[-1]) and sink.shape[-1] < CHUNK_SAMPLES:
        raise ValueError("HEVoicechanger model output is shorter than one 128 ms chunk")
