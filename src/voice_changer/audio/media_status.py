"""Plain status snapshots; importing these types does not access audio or files."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .analysis_status import AnalysisSnapshot
    from .recording import RecordingGroup


Track = Literal["original", "effect"]
PlaybackTrack = Literal["original", "effect", "generated"]


class RecordingState(Enum):
    NONE = "none"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    UNSAVED = "unsaved"
    SAVING = "saving"
    SAVED = "saved"
    SAVE_FAILED = "save_failed"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class MediaStatus:
    monitoring: bool = False
    recording: bool = False
    playback: PlaybackTrack | None = None
    recording_state: RecordingState = RecordingState.NONE
    group: "RecordingGroup | None" = None
    duration: float = 0.0
    free_bytes: int | None = None
    output_peak: float | None = None
    output_peak_sequence: int = 0
    meter_source_id: int = 0
    analysis: "AnalysisSnapshot | None" = None
    recoverable: tuple[str, ...] = ()
