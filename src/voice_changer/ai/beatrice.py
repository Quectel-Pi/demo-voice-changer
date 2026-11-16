"""Experimental Project Beatrice v2 CPU voice conversion, non-commercial only.

The engine is not bundled with this demo. Point ``VOICE_CHANGER_BEATRICE_API_DIR``
at the ``beatrice`` Python package extracted from the VCClient
``only_beatrice`` distribution, and ``VOICE_CHANGER_BEATRICE_MODEL_DIR`` at its
JVS paraphernalia directory (the five ``*.bin`` files plus the TOML). The model
package and API are licensed for non-commercial use only.

Measured on the Quectel Pi M2 board over 48 kHz device blocks of 1024 frames:
about 0.92x real-time with no UI load. The live path is therefore labelled
experimental in the page, and this module never substitutes passthrough, DSP or
offline conversion when the model cannot run.

Importing this module loads nothing: the API is imported only when a converter
is created.

Set ``VOICE_CHANGER_BEATRICE_SUBPROCESS=1`` to run the same stateful engine in
``scripts/beatrice_worker.py`` instead of the GUI process. That is still an
experiment, because the hardware only measured the helper path offline; the
factory still raises instead of substituting passthrough when it fails.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import importlib
import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from ..audio.device_types import CaptureConfig, StreamConfig
    from .rvc_worker import RvcWorkerProcessor

API_ENV = "VOICE_CHANGER_BEATRICE_API_DIR"
MODEL_ENV = "VOICE_CHANGER_BEATRICE_MODEL_DIR"
SUBPROCESS_ENV = "VOICE_CHANGER_BEATRICE_SUBPROCESS"
VERSION = "2.0.0-rc.0"
SAMPLE_RATE_IN = 16000
SAMPLE_RATE_OUT = 24000
HOP_IN = 160  # 10 ms at 16 kHz, the API's fixed input hop
HOP_OUT = 240  # 10 ms at 24 kHz, the API's fixed output hop
SPEAKER_COUNT = 100
PARAMETER_FILES = (
    "phone_extractor.bin",
    "pitch_estimator.bin",
    "embedding_setter.bin",
    "waveform_generator.bin",
    "speaker_embeddings.bin",
)
# 48 kHz device blocks of 1024 frames, Pi M2 CPU, single worker thread.
MEASURED_RTF = 0.92
MEASURED_BLOCK_MS = 19.7
#: The engine follows the source pitch, so a speaker index alone only changes
#: timbre. Female candidates add a pitch lift and a small formant shift; both
#: values were measured on the board (a 124 Hz male source reaches about
#: 232-259 Hz with +12 st, inside the common female range).
MAX_PITCH_SHIFT = 24.0
MIN_PITCH_SHIFT = -24.0
MAX_FORMANT_SHIFT = 2.0
MIN_FORMANT_SHIFT = -2.0
#: The native API lifts formants by at most 2 semitones, which is not enough for
#: a clearly female timbre on a male source. Female candidates therefore add a
#: Rubber Band R3 formant-only stage after the model output. The ratio is a net
#: envelope shift (the DSP "male to female" preset uses 1.18 too); on the board
#: the stage costs about 3.5 ms per 1024-frame block and stays below real time.
MIN_EXTERNAL_FORMANT = 1.0
MAX_EXTERNAL_FORMANT = 1.3
LICENSE_NOTICE = (
    "Beatrice v2 模型与 API 仅限非商业用途（JVS 语料条款）；再分发需 Project Beatrice 许可。 / "
    "The Beatrice v2 model package and API are non-commercial only (JVS corpus terms); "
    "redistribution needs permission from Project Beatrice."
)


@dataclass(frozen=True)
class BeatriceConfig:
    """Where the experimental engine lives and which JVS voice it uses.

    ``pitch_shift_semitone`` / ``formant_shift`` are optional per-voice defaults:
    the engine keeps the source pitch, so the female candidates raise it by 12
    semitones (native API ``set_*`` setters accept exactly these units).

    ``external_formant`` is a net formant ratio applied by a Rubber Band stage
    after the model output (1.0 disables it). The native ``formant_shift`` can
    only reach 2 semitones, so the female voices use this larger envelope lift
    instead of pushing the model parameter further.
    """

    api_dir: Path
    model_dir: Path
    speaker: int = 0
    pitch_shift_semitone: float = 0.0
    formant_shift: float = 0.0
    external_formant: float = 1.0

    def with_speaker(self, speaker: int) -> "BeatriceConfig":
        return replace(self, speaker=clamp_speaker(speaker))

    def with_voice(
        self,
        speaker: int,
        pitch_shift_semitone: float = 0.0,
        formant_shift: float = 0.0,
        external_formant: float = 1.0,
    ) -> "BeatriceConfig":
        return replace(
            self,
            speaker=clamp_speaker(speaker),
            pitch_shift_semitone=clamp_pitch_shift(pitch_shift_semitone),
            formant_shift=clamp_formant_shift(formant_shift),
            external_formant=clamp_external_formant(external_formant),
        )


def subprocess_enabled() -> bool:
    """Whether the helper-process experiment is requested for the live path."""
    return os.environ.get(SUBPROCESS_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def clamp_speaker(value: int) -> int:
    if type(value) is not int:
        raise ValueError("Beatrice speaker index must be an integer")
    return min(max(value, 0), SPEAKER_COUNT - 1)


def clamp_pitch_shift(value: float) -> float:
    if type(value) not in (int, float):
        raise ValueError("Beatrice pitch shift must be a number")
    return float(min(max(float(value), MIN_PITCH_SHIFT), MAX_PITCH_SHIFT))


def clamp_formant_shift(value: float) -> float:
    if type(value) not in (int, float):
        raise ValueError("Beatrice formant shift must be a number")
    return float(min(max(float(value), MIN_FORMANT_SHIFT), MAX_FORMANT_SHIFT))


def clamp_external_formant(value: float) -> float:
    if type(value) not in (int, float):
        raise ValueError("Beatrice external formant must be a number")
    return float(min(max(float(value), MIN_EXTERNAL_FORMANT), MAX_EXTERNAL_FORMANT))


def configured() -> BeatriceConfig | None:
    """Read the engine location from the environment; never touches the disk."""
    api_dir = os.environ.get(API_ENV, "").strip()
    model_dir = os.environ.get(MODEL_ENV, "").strip()
    if not api_dir or not model_dir:
        return None
    return BeatriceConfig(Path(api_dir).expanduser(), Path(model_dir).expanduser())


def unavailable_reason(config: BeatriceConfig | None = None) -> str | None:
    """Return a user-facing reason, or None when the engine looks installable."""
    config = config or configured()
    if config is None:
        return (
            "Beatrice 实验引擎未配置：设置 VOICE_CHANGER_BEATRICE_API_DIR 与 "
            "VOICE_CHANGER_BEATRICE_MODEL_DIR 后可用。 / "
            "The Beatrice experimental engine is not configured: set "
            "VOICE_CHANGER_BEATRICE_API_DIR and VOICE_CHANGER_BEATRICE_MODEL_DIR."
        )
    package = config.api_dir / "beatrice" / "__init__.py"
    if not package.is_file():
        return (
            f"Beatrice API 目录无效：{config.api_dir} / invalid Beatrice API directory: "
            f"{config.api_dir}"
        )
    missing = [name for name in PARAMETER_FILES if not (config.model_dir / name).is_file()]
    if missing:
        return (
            "Beatrice 模型文件缺失 / missing Beatrice model files: "
            + ", ".join(missing)
        )
    if importlib.util.find_spec("numpy") is None:
        return "缺少 numpy，无法运行 Beatrice 实验引擎 / numpy is required for the Beatrice engine"
    if not 0 <= config.speaker < SPEAKER_COUNT:
        return f"Beatrice 说话人序号无效：{config.speaker} / invalid speaker: {config.speaker}"
    return None


def _load_runtime(api_dir: Path):
    """Import the extracted ``beatrice`` package and return its rc0 module."""
    import sys

    text = str(api_dir)
    if text not in sys.path:
        sys.path.insert(0, text)
    try:
        package = importlib.import_module("beatrice")
    except Exception as error:  # the .so can fail for many platform reasons
        raise RuntimeError(
            f"Beatrice API 导入失败 / cannot import the Beatrice API: {error}"
        ) from error
    loader = getattr(package, "load_beatrice", None)
    if not callable(loader):
        raise RuntimeError("Beatrice API 缺少 load_beatrice / the API has no load_beatrice")
    try:
        return loader(VERSION)
    except Exception as error:
        raise RuntimeError(
            f"Beatrice {VERSION} 加载失败 / cannot load Beatrice {VERSION}: {error}"
        ) from error


#: The preflight self-check requires the measured block to leave at least 5% of
#: the block period unused; anything less has been observed to underrun as soon
#: as the GUI, analysis threads and the audio callback compete for the CPU.
PREFLIGHT_RATIO = 0.95


def block_budget_ms(config: "StreamConfig | CaptureConfig") -> float:
    """Wall-clock budget of one device block at 48 kHz."""
    return config.blocksize * 1000.0 / 48000.0


def measure_block_ms(
    config: "StreamConfig | CaptureConfig",
    engine_config: BeatriceConfig,
    *,
    blocks: int = 8,
    runtime=None,
) -> float:
    """Measure one converter block in this process (best-effort UI preflight)."""
    import time

    import numpy as np

    converter = BeatriceConverter(config, engine_config, runtime=runtime)
    silence = np.zeros((config.blocksize, config.channels), dtype=np.float32)
    converter(silence)
    converter(silence)
    started = time.perf_counter()
    for _ in range(blocks):
        converter(silence)
    return (time.perf_counter() - started) * 1000.0 / blocks


class ExternalFormant:
    """Formant-only Rubber Band R3 stage for converted audio blocks.

    The native model lifts formants by at most two semitones, which measured as a
    small, inconsistent envelope change on the board. This stage streams the
    model output through Rubber Band with a pitch ratio of 1.0 and a larger net
    formant ratio, so a converted male voice can actually read as female. It
    primes with silence at start and drops the algorithmic startup delay, so the
    caller receives exactly one block per pushed block (plus the fixed delay).

    Instantiate only on a processing thread; ``close`` must be called when the
    session ends.
    """

    def __init__(
        self, samplerate: int, channels: int, blocksize: int, formant: float,
    ) -> None:
        import numpy as np

        from ..dsp.rubberband import RubberBand

        self._np = np
        self._blocksize = blocksize
        self._channels = channels
        self._dsp = RubberBand(samplerate, channels, blocksize)
        self._dsp.set_scales(1.0, formant)
        self._ring: deque[np.ndarray] = deque()
        self._ring_size = 0
        silence = np.zeros((blocksize, channels), dtype=np.float32)
        target = self._dsp.start_delay + blocksize * 2
        while self._ring_size < target:
            self._dsp.process(silence)
            self._drain()
        self._discard(self._dsp.start_delay)

    def _drain(self) -> None:
        np = self._np
        while self._dsp.available > 0:
            buffer = np.empty((self._blocksize, self._channels), dtype=np.float32)
            received = self._dsp.retrieve(buffer, min(self._dsp.available, self._blocksize))
            self._ring.append(buffer[:received].copy())
            self._ring_size += received
        if self._ring_size > self._blocksize * 16:
            raise RuntimeError("Beatrice formant stage buffers exceeded their limit")

    def _take(self, size: int) -> "np.ndarray":
        np = self._np
        output = np.empty((size, self._channels), dtype=np.float32)
        copied = 0
        while copied < size and self._ring:
            head = self._ring[0]
            needed = size - copied
            if head.shape[0] <= needed:
                output[copied:copied + head.shape[0]] = head
                copied += head.shape[0]
                self._ring.popleft()
            else:
                output[copied:copied + needed] = head[:needed]
                self._ring[0] = head[needed:]
                copied += needed
        self._ring_size -= copied
        return output

    def _discard(self, size: int) -> None:
        while size > 0 and self._ring:
            head = self._ring[0]
            if head.shape[0] <= size:
                size -= head.shape[0]
                self._ring.popleft()
                self._ring_size -= head.shape[0]
            else:
                self._ring[0] = head[size:]
                self._ring_size -= size
                size = 0

    def process(self, block: "np.ndarray") -> "np.ndarray":
        """Shift the formants of one block and return the corresponding block."""
        self._dsp.process(block)
        self._drain()
        if self._ring_size < self._blocksize:
            raise RuntimeError("Beatrice formant stage underrun; no passthrough is used")
        return self._take(self._blocksize)

    def flush(self) -> "np.ndarray":
        """Return audio still buffered after the last processed block."""
        np = self._np
        silence = np.zeros((self._blocksize, self._channels), dtype=np.float32)
        zero_blocks = -(-(self._dsp.start_pad + self._dsp.start_delay) // self._blocksize) + 2
        for _ in range(zero_blocks):
            self._dsp.process(silence)
            self._drain()
        if self._ring_size <= 0:
            return np.zeros((0, self._channels), dtype=np.float32)
        return self._take(self._ring_size)

    def close(self) -> None:
        self._dsp.close()


class BeatriceConverter:
    """Stream 48 kHz device blocks through the 16 kHz -> 24 kHz Beatrice graph."""

    def __init__(
        self, config: "StreamConfig | CaptureConfig", engine_config: BeatriceConfig,
        *, runtime=None,
    ) -> None:
        import numpy as np

        from .resampling import StreamingResampler

        if (config.samplerate != 48000 or config.blocksize not in (1024, 2496)
                or config.channels not in (1, 2)):
            raise ValueError("Beatrice requires 48000 Hz, 1024/2496 frames and 1 or 2 channels")
        reason = unavailable_reason(engine_config)
        if reason is not None:
            raise RuntimeError(reason)
        runtime = runtime if runtime is not None else _load_runtime(engine_config.api_dir)
        model_dir = engine_config.model_dir
        try:
            self.engine = runtime.SimpleBeatrice(
                str(model_dir / "phone_extractor.bin"),
                str(model_dir / "pitch_estimator.bin"),
                str(model_dir / "embedding_setter.bin"),
                str(model_dir / "waveform_generator.bin"),
                str(model_dir / "speaker_embeddings.bin"),
            )
        except Exception as error:
            raise RuntimeError(
                f"Beatrice 模型初始化失败 / cannot initialize the Beatrice model: {error}"
            ) from error
        if not self.engine.is_ready():
            raise RuntimeError("Beatrice 模型未就绪 / the Beatrice model is not ready")
        self.speaker = clamp_speaker(engine_config.speaker)
        self.pitch_shift_semitone = clamp_pitch_shift(engine_config.pitch_shift_semitone)
        self.formant_shift = clamp_formant_shift(engine_config.formant_shift)
        self.external_formant = clamp_external_formant(engine_config.external_formant)
        self.engine.set_target_speaker(self.speaker)
        if self.pitch_shift_semitone:
            self.engine.set_pitch_shift_semitone(self.pitch_shift_semitone)
        if self.formant_shift:
            self.engine.set_formant_shift(self.formant_shift)
        self.config = replace(config)
        self.down = StreamingResampler(48000, SAMPLE_RATE_IN)
        self.up = StreamingResampler(SAMPLE_RATE_OUT, 48000)
        # Deques make the per-hop path allocation free. Concatenating growing
        # buffers on every hop cost about 6% of real time on the board, which
        # was enough for the audio callback queue to underrun.
        self._pending: deque[np.ndarray] = deque()
        self._pending_size = 0
        self._ready: deque[np.ndarray] = deque()
        self._ready_size = 0
        # Prime two blocks so both resamplers publish without an initial underrun.
        self._ready.append(np.zeros(config.blocksize * 2, dtype=np.float32))
        self._ready_size = config.blocksize * 2
        # A larger formant lift than the native 2-semitone API needs its own
        # high-quality stage; when it is 1.0 (the male candidate) nothing is
        # allocated and the converter is byte-for-byte the old path.
        self._formant_stage = (
            ExternalFormant(config.samplerate, config.channels, config.blocksize,
                            self.external_formant)
            if self.external_formant != 1.0 else None
        )

    @staticmethod
    def _take(parts: "deque[np.ndarray]", size: int, target: "np.ndarray") -> int:
        """Copy up to ``size`` frames from ``parts`` into ``target`` in place."""
        copied = 0
        while copied < size and parts:
            head = parts[0]
            needed = size - copied
            if head.size <= needed:
                target[copied:copied + head.size] = head
                copied += head.size
                parts.popleft()
            else:
                target[copied:copied + needed] = head[:needed]
                parts[0] = head[needed:]
                copied += needed
        return copied

    def __call__(self, samples: "np.ndarray") -> "np.ndarray":
        import numpy as np

        self._validate(samples)
        if self.config.channels == 1:
            mono = samples[:, 0].copy()
        else:
            mono = samples.mean(axis=1, dtype=np.float32)
        np.clip(mono, -1, 1, out=mono)
        down = self.down.process(mono)
        self._pending.append(down)
        self._pending_size += down.size
        while self._pending_size >= HOP_IN:
            hop = np.empty(HOP_IN, dtype=np.float32)
            self._pending_size -= self._take(self._pending, HOP_IN, hop)
            converted = self._convert_hop(hop)
            upsampled = self.up.process(converted)
            self._ready.append(upsampled)
            self._ready_size += upsampled.size
        blocksize = self.config.blocksize
        if self._ready_size < blocksize:
            raise RuntimeError("Beatrice resampling output underrun; no passthrough is used")
        mono_out = np.empty(blocksize, dtype=np.float32)
        self._ready_size -= self._take(self._ready, blocksize, mono_out)
        if self.config.channels == 1:
            output = mono_out[:, None]
        else:
            output = np.repeat(mono_out[:, None], self.config.channels, axis=1)
        if self._ready_size > blocksize * 4:
            raise RuntimeError("Beatrice streaming buffers exceeded their limit")
        if self._formant_stage is not None:
            output = self._formant_stage.process(output)
        np.clip(output, -1, 1, out=output)
        return output

    def flush(self) -> "np.ndarray":
        """Return audio still buffered after the last input block (preview tail)."""
        import numpy as np

        if self._formant_stage is None:
            return np.zeros((0, self.config.channels), dtype=np.float32)
        return self._formant_stage.flush()

    def _convert_hop(self, block: "np.ndarray") -> "np.ndarray":
        import numpy as np

        result = self.engine.convert(block)
        wave = result[0] if isinstance(result, tuple) else result
        converted = np.asarray(wave, dtype=np.float32).reshape(-1)
        if converted.size != HOP_OUT or not np.isfinite(converted).all():
            raise RuntimeError("Beatrice returned invalid audio")
        error = getattr(self.engine, "last_error", None)
        if callable(error) and error() != 0:
            raise RuntimeError(f"Beatrice engine error {error()}")
        return np.clip(converted, -1, 1)

    def _validate(self, samples: "np.ndarray") -> None:
        import numpy as np

        if (not isinstance(samples, np.ndarray)
                or samples.shape != (self.config.blocksize, self.config.channels)
                or samples.dtype != np.float32 or not np.isfinite(samples).all()):
            raise ValueError("Invalid Beatrice device audio")

    def close(self) -> None:
        if self._formant_stage is not None:
            self._formant_stage.close()
            self._formant_stage = None


@dataclass(frozen=True)
class BeatriceProcessorFactory:
    """Create one Beatrice worker per listening session, like the LLVC factory."""

    engine_config: BeatriceConfig

    def __call__(self, config: "StreamConfig | CaptureConfig", _control) -> "RvcWorkerProcessor":
        from .rvc_worker import RvcWorkerProcessor

        if config.samplerate != 48000 or config.blocksize not in (1024, 2496):
            raise ValueError("Beatrice audio device requires 48000 Hz and 1024 frames")
        if subprocess_enabled():
            from .beatrice_subprocess import subprocess_processor
            return subprocess_processor(config, self.engine_config)
        converter = BeatriceConverter(config, self.engine_config)
        # The Pi M2 runs this engine at roughly 0.9x real time, so the default
        # two-block startup reserve leaves too little room for scheduler jitter
        # and periodic resampler cost spikes. A deeper queue and reserve keep
        # the experimental engine from declaring an underrun on short stalls.
        return RvcWorkerProcessor(
            config, converter, conversion_frames=converter.config.blocksize,
            queue_blocks=16, startup_blocks=4,
        )
