"""Single-owner selection state; no queries, audio activity or Qt dependencies."""

from typing import Literal

from ..audio.device_discovery import DeviceInfo
from ..audio.device_types import CaptureConfig, DeviceRef, OutputConfig, StreamConfig


DeviceKind = Literal["input", "output"]


class DeviceSelectionError(RuntimeError):
    pass


class SelectionInvalidatedError(DeviceSelectionError):
    """A valid refresh was committed, but affected selections must be reselected."""

    def __init__(self, invalidated: tuple[DeviceKind, ...]) -> None:
        self.invalidated = invalidated
        super().__init__(
            f"Selected {' and '.join(invalidated)} device changed, disappeared or "
            "lost channel capability; reselect explicitly. No automatic fallback."
        )


class DeviceSelection:
    """Call serially from one owner; supply actual busy/resource facts explicitly."""

    def __init__(self) -> None:
        self._devices: tuple[DeviceInfo, ...] = ()
        self._input: DeviceRef | None = None
        self._output: DeviceRef | None = None
        self._busy = False
        self._owns_devices = False

    @property
    def devices(self) -> tuple[DeviceInfo, ...]:
        return self._devices

    @property
    def input_device(self) -> DeviceRef | None:
        return self._input

    @property
    def output_device(self) -> DeviceRef | None:
        return self._output

    @property
    def locked(self) -> bool:
        return self._busy or self._owns_devices

    def set_activity(self, *, busy: bool, owns_devices: bool) -> None:
        if type(busy) is not bool or type(owns_devices) is not bool:
            raise ValueError("busy and owns_devices must be explicit booleans")
        self._busy = busy
        self._owns_devices = owns_devices

    def _require_idle(self) -> None:
        if self.locked:
            raise DeviceSelectionError(
                "Device selection and manual refresh are locked until operations "
                "finish and devices are released"
            )

    @staticmethod
    def _supports(
        devices: tuple[DeviceInfo, ...], kind: DeviceKind, ref: DeviceRef,
        *, channels: int = 1,
    ) -> bool:
        for info in devices:
            if info.ref == ref:
                capacity = (
                    info.max_input_channels if kind == "input" else info.max_output_channels
                )
                return capacity >= channels
        return False

    def build_stream_config(
        self, *, samplerate: int, blocksize: int, channels: int, output_ceiling: float,
    ) -> StreamConfig:
        """Validate against the current snapshot; never probe formats or open audio."""
        self._require_idle()
        if self._input is None or self._output is None:
            raise DeviceSelectionError(
                "Both input and output devices must be explicitly selected; "
                "refresh and select before configuring a duplex stream"
            )
        config = StreamConfig(
            input_device=self._input, output_device=self._output,
            samplerate=samplerate, blocksize=blocksize, channels=channels,
            output_ceiling=output_ceiling,
        )
        selections: tuple[tuple[DeviceKind, DeviceRef], ...] = (
            ("input", config.input_device), ("output", config.output_device),
        )
        for kind, ref in selections:
            if not self._supports(self._devices, kind, ref, channels=config.channels):
                raise DeviceSelectionError(
                    f"Selected {kind} device is missing, changed or has too few "
                    "channels for this configuration; refresh and reselect"
                )
        return config

    def select(self, kind: DeviceKind, ref: DeviceRef | None) -> None:
        self._require_idle()
        if kind not in ("input", "output"):
            raise ValueError("Device kind must be input or output")
        if ref is not None:
            if not isinstance(ref, DeviceRef):
                raise ValueError("Selection must be an explicit DeviceRef or None")
            if not self._supports(self._devices, kind, ref):
                raise DeviceSelectionError(
                    f"Selected {kind} device is missing, changed or has no {kind} "
                    "channels; refresh and reselect"
                )
        if kind == "input":
            self._input = ref
        else:
            self._output = ref

    def build_capture_config(
        self, *, samplerate: int, blocksize: int, channels: int, output_ceiling: float,
    ) -> CaptureConfig:
        """Read a frozen snapshot; selection changes remain locked during activity."""
        if self._input is None:
            raise DeviceSelectionError("Select an input device before recording")
        config = CaptureConfig(self._input, samplerate, blocksize, channels, output_ceiling)
        if not self._supports(self._devices, "input", self._input, channels=channels):
            raise DeviceSelectionError("Selected input changed or has too few channels")
        return config

    def build_output_config(
        self, *, samplerate: int, blocksize: int, channels: int, output_ceiling: float,
    ) -> OutputConfig:
        """Validate only the selected output; playback never requires an input."""
        if self._output is None:
            raise DeviceSelectionError("Select an output device before playback or monitoring")
        config = OutputConfig(self._output, samplerate, blocksize, channels, output_ceiling)
        if not self._supports(self._devices, "output", self._output, channels=channels):
            raise DeviceSelectionError("Selected output changed or has too few channels")
        return config

    def refresh(self, devices: tuple[DeviceInfo, ...]) -> None:
        """Commit a valid snapshot, then report invalidation; never query devices."""
        self._require_idle()
        if not isinstance(devices, tuple) or any(
            not isinstance(info, DeviceInfo) for info in devices
        ):
            raise ValueError("Refresh requires a tuple of validated DeviceInfo records")
        indices = [info.ref.index for info in devices]
        if len(set(indices)) != len(indices):
            raise ValueError("Device snapshot contains duplicate global indices")
        invalidated: list[DeviceKind] = []
        selections: tuple[tuple[DeviceKind, DeviceRef | None], ...] = (
            ("input", self._input), ("output", self._output),
        )
        for kind, ref in selections:
            if ref is not None and not self._supports(devices, kind, ref):
                invalidated.append(kind)
        self._devices = devices
        if "input" in invalidated:
            self._input = None
        if "output" in invalidated:
            self._output = None
        if invalidated:
            raise SelectionInvalidatedError(tuple(invalidated))
