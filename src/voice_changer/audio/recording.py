"""Paired PCM16 recording storage; no audio backend or NumPy import at import time.

``submit`` has one DSP-thread producer and a preallocated 64-block SPSC queue.
Control operations require the producer to have stopped before ``finish``. This
CPython/GIL handoff is bounded, not a hard-realtime or lock-free guarantee.
Parameter changes are streamed to parameters.jsonl, never accumulated in a list.
Alignment describes caller-provided pairing, not an acoustic latency measurement.
"""

from __future__ import annotations

from dataclasses import asdict
from contextlib import ExitStack, contextmanager
import fcntl
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import threading
import time
from typing import TYPE_CHECKING
import uuid
import wave

if TYPE_CHECKING:
    from voice_changer.dsp.parameters import EffectParameters


RESERVE_BYTES = 64 * 1024 * 1024
DISK_CHECK_SECONDS = 1.0
RIFF_DATA_LIMIT = 0xFFFFFFFF - 36
JOIN_TIMEOUT_SECONDS = 5.0
_PREFIX = ".voice-changer-recording-"
_OWNER = "voice-changer paired recording v1\n"
_FILES = (".owner", "original.wav", "effect.wav", "parameters.jsonl", "metadata.json",
          "metadata.json.pending")
_NAMES = re.compile(r"\.voice-changer-recording-[0-9a-f]{32}\Z")


class RecordingError(RuntimeError):
    """Explicit storage, lifecycle, or recovery failure."""


class RecordingRecoveryError(RecordingError):
    """Files were retained; this directory is not a validated paired recording."""


class RecordingSaveError(RecordingError):
    """A failed save can leave published files and/or a verified saved copy."""

    def __init__(self, destination, published, cause, saved_group=None, failed_item=None):
        self.destination = destination
        self.published = tuple(published)
        self.cause = cause
        self.saved_group = saved_group
        self.failed_item = failed_item
        super().__init__(
            f"Save failed at {destination} ({failed_item}); published={self.published}: {cause}"
        )


def _root(data_root):
    if data_root is not None:
        return Path(data_root)
    home = os.environ.get("XDG_DATA_HOME")
    base = Path(home) if home and Path(home).is_absolute() else Path.home() / ".local/share"
    return base / "voice-changer" / "recordings"


def _regular(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise RecordingRecoveryError(f"Not a regular, non-symlink file: {path}")


def _directory_identity(directory):
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise RecordingRecoveryError(f"Not a non-symlink directory: {directory}")
    return info.st_dev, info.st_ino


def _owned(directory):
    if not _NAMES.fullmatch(directory.name):
        raise RecordingRecoveryError("Not an owned temporary recording directory")
    identity = _directory_identity(directory)
    _regular(directory / ".owner")
    if (directory / ".owner").read_text(encoding="utf-8") != _OWNER:
        raise RecordingRecoveryError("Invalid temporary recording ownership marker")
    return identity


@contextmanager
def _owner_lock(directory):
    """Nonblocking Linux advisory lock, with owner-file/path identity validation."""
    identity = _owned(directory)
    fd = os.open(directory / ".owner", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RecordingRecoveryError("Owner marker is not a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RecordingRecoveryError("Recording group is actively owned/locked") from exc
        marker = (directory / ".owner").lstat()
        if (
            (marker.st_dev, marker.st_ino) != (info.st_dev, info.st_ino)
            or _owned(directory) != identity
            or os.read(fd, len(_OWNER.encode()) + 1) != _OWNER.encode()
        ):
            raise RecordingRecoveryError("Recording owner/path changed during locking")
        yield identity
    finally:
        os.close(fd)


def recoverable_groups(data_root: Path | None = None) -> tuple[Path, ...]:
    """List owned leftovers only; no creation, playback, or implicit recovery."""
    root = _root(data_root)
    if not root.exists():
        return ()
    result = []
    for path in root.iterdir():
        if not _NAMES.fullmatch(path.name):
            continue
        try:
            with _owner_lock(path):
                result.append(path)
        except (OSError, UnicodeError, RecordingRecoveryError):
            continue
    return tuple(sorted(result))


def discard_recoverable(directory: Path) -> None:
    """Explicit deletion of an owned leftover, even if recovery validation fails.

    The caller must obtain user approval; active owners are rejected.
    No audio format is inferred and no RecordingGroup is fabricated.
    """
    directory = Path(directory)
    with _owner_lock(directory) as identity:
        metadata_path = directory / "metadata.json"
        if metadata_path.exists() and not metadata_path.is_symlink():
            try:
                metadata = _metadata(directory)
            except (ValueError, UnicodeError):
                metadata = None
            if isinstance(metadata, dict) and metadata.get("saved") is True:
                raise RecordingError("Cannot discard a user's saved recording group")
        _delete_owned_directory(directory, identity)


def _discard_owned_directory(directory, expected_identity):
    with _owner_lock(directory) as identity:
        if identity != expected_identity:
            raise RecordingError("Temporary directory identity changed; refusing cleanup")
        _delete_owned_directory(directory, identity)


def _delete_owned_directory(directory, expected_identity):
    identity = _owned(directory)
    if identity != expected_identity:
        raise RecordingError("Temporary directory identity changed; refusing cleanup")
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) != identity:
            raise RecordingError("Temporary directory changed during cleanup")
        entries = os.listdir(fd)
        if set(entries) - set(_FILES):
            raise RecordingError("Unknown files present; refusing temporary cleanup")
        for filename in entries:
            info = os.stat(filename, dir_fd=fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise RecordingError("Symlink or non-file present; refusing cleanup")
        # Keep ownership marker until last so interrupted cleanup stays discoverable.
        for filename in (*_FILES[1:], ".owner"):
            if filename in entries:
                os.unlink(filename, dir_fd=fd)
        if _directory_identity(directory) != identity:
            raise RecordingError("Temporary path changed; refusing directory removal")
        directory.rmdir()
    finally:
        os.close(fd)


def _metadata(directory):
    _regular(directory / "metadata.json")
    with (directory / "metadata.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.digest()


def _validate(directory, metadata):
    """Strict validation is intentional: stale/misaligned data is never guessed."""
    from voice_changer.dsp.parameters import EffectParameters

    if not isinstance(metadata, dict) or metadata.get("schema") != 1:
        raise RecordingRecoveryError("Unsupported recording metadata")
    rate, channels, frames = (
        metadata.get("samplerate"), metadata.get("channels"), metadata.get("frames")
    )
    if (
        type(rate) is not int or rate != 48000 or type(channels) is not int
        or channels not in (1, 2) or type(frames) is not int or frames < 0
    ):
        raise RecordingRecoveryError("Invalid recording format or frame count")
    if metadata.get("format") != "PCM16_LE" or not isinstance(metadata.get("alignment"), dict):
        raise RecordingRecoveryError("Missing PCM16 format/alignment description")
    for field in ("complete", "finalized", "saved"):
        if type(metadata.get(field)) is not bool:
            raise RecordingRecoveryError(f"Invalid {field} state")
    if metadata["complete"] and not metadata["finalized"]:
        raise RecordingRecoveryError("Active recording cannot be complete")
    if not metadata["complete"] and (
        not isinstance(metadata.get("reason"), str) or not metadata["reason"]
    ):
        raise RecordingRecoveryError("Incomplete recording has no reason")
    interval = metadata.get("source_interval")
    if (
        not isinstance(interval, dict)
        or (frames and type(interval.get("start")) is not int)
        or (frames and interval["start"] < 0)
        or (frames and interval.get("end_exclusive") != interval["start"] + frames)
        or (not frames and interval != {"start": None, "end_exclusive": None})
    ):
        raise RecordingRecoveryError("Invalid source interval")
    tracks = metadata.get("tracks")
    if not isinstance(tracks, dict):
        raise RecordingRecoveryError("Missing paired tracks")
    for name in ("original", "effect"):
        path = directory / f"{name}.wav"
        _regular(path)
        track = tracks.get(name)
        if (
            not isinstance(track, dict) or track.get("file") != path.name
            or track.get("frames") != frames or track.get("source_interval") != interval
        ):
            raise RecordingRecoveryError(f"{name}: mismatched track/source alignment")
        with wave.open(str(path), "rb") as wav:
            if (
                wav.getnchannels() != channels or wav.getframerate() != rate
                or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE"
                or wav.getnframes() != frames
            ):
                raise RecordingRecoveryError(f"{name}: WAV format/length mismatch")
            remaining = frames * channels * 2
            while remaining:
                data = wav.readframes(min(65536, remaining // (channels * 2)))
                if not data:
                    raise RecordingRecoveryError(f"{name}: truncated PCM payload")
                remaining -= len(data)
    changes = metadata.get("parameter_changes")
    if not isinstance(changes, dict) or changes.get("file") != "parameters.jsonl":
        raise RecordingRecoveryError("Invalid parameter journal reference")
    _regular(directory / "parameters.jsonl")
    previous_source = None
    count = 0
    with (directory / "parameters.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            source = record["source_frame"]
            revision = record["revision"]
            if (
                type(source) is not int or type(revision) is not int or revision < 0
                or not frames or not interval["start"] <= source < interval["end_exclusive"]
                or (previous_source is not None and source <= previous_source)
            ):
                raise RecordingRecoveryError("Parameter journal source/revision mismatch")
            EffectParameters(**record["parameters"])
            if count == 0 and source != interval["start"]:
                raise RecordingRecoveryError("Missing initial parameters")
            previous_source = source
            count += 1
    if count != changes.get("count") or (frames and not count):
        raise RecordingRecoveryError("Parameter journal count mismatch")
    return rate, channels, frames


class RecordingGroup:
    """Control-thread handle. Saved directories are never eligible for discard."""

    def __init__(self, directory, samplerate, channels, *, frames=0, complete=False,
                 saved=False, finalized=False):
        self.directory = Path(directory)
        self.original_path = self.directory / "original.wav"
        self.effect_path = self.directory / "effect.wav"
        self.metadata_path = self.directory / "metadata.json"
        self.samplerate = samplerate
        self.channels = channels
        self.frames = frames
        self.complete = complete
        self.saved = saved
        self._finalized = finalized
        self._identity = None
        self._discarded = False
        self._operation = threading.Lock()

    @classmethod
    def recover(cls, directory: Path) -> "RecordingGroup":
        directory = Path(directory)
        try:
            with _owner_lock(directory) as identity:
                metadata = _metadata(directory)
                rate, channels, frames = _validate(directory, metadata)
                if metadata["saved"]:
                    raise RecordingRecoveryError("Saved directories are not temporary recovery groups")
                group = cls(directory, rate, channels, frames=frames,
                            complete=metadata["complete"], finalized=True)
                group._identity = identity
                return group
        except (OSError, ValueError, KeyError, TypeError, EOFError, wave.Error) as exc:
            raise RecordingRecoveryError(f"Cannot recover {directory}: {exc}") from exc

    def _available(self):
        if not self._finalized or self._discarded:
            raise RecordingError("Recording must be finalized and not discarded")

    def save(self, parent: Path, name: str) -> "RecordingGroup":
        destination = Path(parent) / name
        if not self._operation.acquire(blocking=False):
            raise RecordingError("Recording group has another control operation in progress")
        published = []
        saved = None
        failed_item = "source validation"
        ownership = ExitStack()
        try:
            self._available()
            if self.saved:
                raise RecordingError("Recording group is already saved")
            if (
                not name or name in (".", "..") or Path(name).name != name
                or "/" in name or "\\" in name or "\x00" in name or name.startswith(_PREFIX)
            ):
                raise ValueError("A new single directory name is required")
            identity = ownership.enter_context(_owner_lock(self.directory))
            if identity != self._identity:
                raise RecordingError("Temporary directory identity changed")
            metadata = _metadata(self.directory)
            _validate(self.directory, metadata)
            failed_item = "destination directory"
            destination.mkdir(mode=0o700)
            for filename in ("original.wav", "effect.wav", "parameters.jsonl"):
                failed_item = filename
                with (self.directory / filename).open("rb") as source:
                    with (destination / filename).open("xb") as target:
                        shutil.copyfileobj(source, target, 1024 * 1024)
                        target.flush()
                        os.fsync(target.fileno())
                published.append(filename)
                if _digest(self.directory / filename) != _digest(destination / filename):
                    raise RecordingError(f"Copied {filename} differs from temporary data")
            metadata["saved"] = True
            metadata["finalized"] = True
            failed_item = "metadata.json"
            with (destination / "metadata.json").open("x", encoding="utf-8") as target:
                json.dump(metadata, target, allow_nan=False)
                target.flush()
                os.fsync(target.fileno())
            published.append("metadata.json")
            _validate(destination, _metadata(destination))
            saved = RecordingGroup(destination, self.samplerate, self.channels,
                                   frames=self.frames, complete=self.complete,
                                   saved=True, finalized=True)
            failed_item = "temporary cleanup (saved copy verified)"
            self._discard_owned(locked_identity=identity)
            return saved
        except (OSError, ValueError, TypeError, RecordingError, EOFError, wave.Error) as exc:
            raise RecordingSaveError(destination, published, exc, saved, failed_item) from exc
        finally:
            try:
                ownership.close()
            finally:
                self._operation.release()

    def _discard_owned(self, *, locked_identity=None):
        if self._identity is None:
            raise RecordingError("Missing temporary directory identity; refusing cleanup")
        if locked_identity is None:
            _discard_owned_directory(self.directory, self._identity)
        else:
            if locked_identity != self._identity:
                raise RecordingError("Temporary directory identity changed; refusing cleanup")
            _delete_owned_directory(self.directory, locked_identity)
        self._discarded = True

    def discard(self) -> None:
        if not self._operation.acquire(blocking=False):
            raise RecordingError("Recording group has another control operation in progress")
        try:
            self._available()
            if self.saved:
                raise RecordingError("Cannot discard a user's saved recording group")
            self._discard_owned()
        finally:
            self._operation.release()


class RecordingWriter:
    """One DSP producer; one disk consumer. All lifecycle calls are control-thread."""

    def __init__(self, samplerate: int, channels: int, blocksize: int, *,
                 data_root: Path | None = None, alignment: dict | None = None):
        if (
            type(samplerate) is not int or samplerate != 48000
            or type(channels) is not int or channels not in (1, 2)
            or type(blocksize) is not int or blocksize < 1
        ):
            raise ValueError("Recording requires 48000 Hz, mono/stereo, positive blocksize")
        if alignment is not None and not isinstance(alignment, dict):
            raise ValueError("alignment must be a JSON dictionary")
        import numpy as np
        from voice_changer.dsp.parameters import EffectParameters

        self._np = np
        self._parameter_type = EffectParameters
        self.samplerate, self.channels, self.blocksize = samplerate, channels, blocksize
        self.alignment = json.loads(json.dumps(
            alignment if alignment is not None else {
                "status": "caller-paired; acoustic alignment not verified",
                "gate": "caller owns start/end and DSP tail handling",
            }, allow_nan=False,
        ))
        self.error: Exception | None = None
        self.secondary_errors: tuple[Exception, ...] = ()
        self.frames_written = 0
        self.free_bytes = 0
        self.group = RecordingGroup(_root(data_root) / (_PREFIX + uuid.uuid4().hex),
                                    samplerate, channels)
        self._raw = np.empty((64, blocksize, channels), dtype=np.float32)
        self._effect = np.empty_like(self._raw)
        self._finite = np.empty((blocksize, channels), dtype=np.bool_)
        self._lengths = [0] * 64
        self._sources = [0] * 64
        self._parameters = [None] * 64
        self._revisions = [0] * 64
        self._read = self._write = 0
        self._source_start = self._next_source = None
        self._accepting = self._stop = self._started = False
        self._requested_complete = False
        self._reason = None
        self._thread = None
        self._owner_fd = None
        self._waves = []
        self._journal = None
        self._track_frames = [0, 0]
        self._clipped = [0, 0]
        self._changes = 0
        self._last_parameters = None
        self._last_check = 0.0
        self._budget = 0
        self._invalid = RecordingError("Invalid recording block, parameters, or source continuity")
        self._overflow = RecordingError("Recording's bounded 64-block queue overflowed")
        self._inactive = RecordingError("Recording writer is not accepting blocks")
        try:
            self.group.directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.group.directory.mkdir(mode=0o700)
            self.group._identity = _directory_identity(self.group.directory)
            self._owner_fd = os.open(
                self.group.directory / ".owner",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600,
            )
            fcntl.flock(self._owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if os.write(self._owner_fd, _OWNER.encode()) != len(_OWNER.encode()):
                raise OSError("Short recording owner marker write")
            self._check_disk()
            for name in ("original.wav", "effect.wav"):
                wav = wave.open(str(self.group.directory / name), "wb")
                self._waves.append(wav)
                wav.setnchannels(channels)
                wav.setsampwidth(2)
                wav.setframerate(samplerate)
                wav.writeframes(b"")
            self._journal = (self.group.directory / "parameters.jsonl").open("x", encoding="utf-8")
            self._write_metadata(finalized=False)
        except (OSError, ValueError, wave.Error) as exc:
            self._set_error(exc)
            self._finalize()

    def _set_error(self, error):
        if self.error is None:
            self.error = error
        elif self.error is not error:
            self.secondary_errors = (*self.secondary_errors, error)
        self._accepting = False
        self._stop = True
        self._requested_complete = False
        self.group.complete = False

    def _check_disk(self):
        self.free_bytes = shutil.disk_usage(self.group.directory).free
        self._budget = self.free_bytes
        self._last_check = time.monotonic()
        # Allow one second of paired PCM between periodic checks, above the reserve.
        if self.free_bytes < RESERVE_BYTES + self.samplerate * self.channels * 4:
            raise OSError("Recording disk space reached the 64 MiB safety reserve")

    def start(self) -> None:
        self.check_health()
        if self._started or self.group._finalized:
            raise RecordingError("Recording writer cannot be started twice or after finish")
        self._started = self._accepting = True
        self._thread = threading.Thread(target=self._run, name="recording-disk-writer")
        try:
            self._thread.start()
        except RuntimeError as exc:
            self._set_error(exc)
            self._finalize()
            raise

    def submit(self, raw, effect, *, source_frame: int,
               parameters: EffectParameters, revision: int) -> bool:
        """DSP thread only: validate/copy/publish, no I/O, locks, waits or formatting."""
        if not self._accepting:
            if self.error is None:
                self.error = self._inactive
            return False
        np = self._np
        if (
            not isinstance(raw, np.ndarray) or not isinstance(effect, np.ndarray)
            or raw.dtype != np.float32 or effect.dtype != np.float32
            or raw.ndim != 2 or effect.shape != raw.shape
            or raw.shape[1] != self.channels or not 0 < raw.shape[0] <= self.blocksize
            or type(source_frame) is not int or source_frame < 0
            or (self._next_source is not None and source_frame != self._next_source)
            or not isinstance(parameters, self._parameter_type)
            or type(revision) is not int or revision < 0
        ):
            self.error = self._invalid
            self._accepting = False
            self._stop = True
            return False
        n = raw.shape[0]
        finite = self._finite[:n]
        np.isfinite(raw, out=finite)
        valid = finite.all()
        np.isfinite(effect, out=finite)
        if not valid or not finite.all():
            self.error = self._invalid
            self._accepting = False
            self._stop = True
            return False
        if self._write - self._read >= 64:
            self.error = self._overflow
            self._accepting = False
            self._stop = True
            return False
        slot = self._write % 64
        np.copyto(self._raw[slot, :n], raw)
        np.copyto(self._effect[slot, :n], effect)
        if not self._accepting:
            return False
        self._lengths[slot] = n
        self._sources[slot] = source_frame
        self._parameters[slot] = parameters
        self._revisions[slot] = revision
        self._next_source = source_frame + n
        self._write += 1
        return True

    def check_health(self) -> None:
        """Raise the retained real failure, including after finish() returned a group."""
        if self.error is not None:
            raise self.error

    def _write_block(self, slot):
        np = self._np
        n = self._lengths[slot]
        size = n * self.channels * 2
        if any(frames * self.channels * 2 + size > RIFF_DATA_LIMIT
               for frames in self._track_frames):
            raise RecordingError("PCM WAV RIFF size limit reached")
        if time.monotonic() - self._last_check >= DISK_CHECK_SECONDS or self._budget - 2 * size < RESERVE_BYTES:
            self._check_disk()
        if self._source_start is None:
            self._source_start = self._sources[slot]
        for index, data in enumerate((self._raw[slot, :n], self._effect[slot, :n])):
            clipped = int(np.count_nonzero((data < -1) | (data > 1)))
            pcm = np.rint(np.clip(data, -1, 1) * 32768).clip(-32768, 32767).astype("<i2")
            self._waves[index].writeframes(pcm.tobytes())
            self._track_frames[index] += n
            self._clipped[index] += clipped
        self.frames_written = min(self._track_frames)
        self.group.frames = self.frames_written
        self._budget -= 2 * size
        parameters = self._parameters[slot]
        revision = self._revisions[slot]
        snapshot = (revision, parameters)
        if snapshot != self._last_parameters:
            json.dump({"source_frame": self._sources[slot], "revision": revision,
                       "parameters": asdict(parameters)}, self._journal, allow_nan=False)
            self._journal.write("\n")
            self._journal.flush()
            self._changes += 1
            self._last_parameters = snapshot
        self._parameters[slot] = None

    def _run(self):
        checkpoint = time.monotonic()
        try:
            while not self._stop or self._read < self._write:
                if time.monotonic() - self._last_check >= DISK_CHECK_SECONDS:
                    self._check_disk()
                if self._read < self._write:
                    self._write_block(self._read % 64)
                    self._read += 1
                    if time.monotonic() - checkpoint >= DISK_CHECK_SECONDS:
                        self._write_metadata(finalized=False)
                        checkpoint = time.monotonic()
                else:
                    time.sleep(0.002)
        except (OSError, ValueError, TypeError, RuntimeError, OverflowError, wave.Error) as exc:
            self._set_error(exc)
        finally:
            # Preserve unexpected thread failures too, without suppressing their traceback.
            unexpected = sys.exc_info()[1]
            if isinstance(unexpected, Exception):
                self._set_error(unexpected)
            self._accepting = False
            self._finalize()

    def _write_metadata(self, *, finalized):
        complete = bool(finalized and self._requested_complete and self.error is None)
        reason = None
        if not complete:
            reason = f"{type(self.error).__name__}: {self.error}" if self.error is not None else (
                self._reason or ("Recording marked incomplete" if finalized
                                 else "Recording not finalized")
            )
        frames = self.frames_written
        interval = {
            "start": self._source_start if frames else None,
            "end_exclusive": self._source_start + frames if frames else None,
        }
        tracks = {}
        for index, name in enumerate(("original", "effect")):
            tracks[name] = {
                "file": f"{name}.wav", "frames": self._track_frames[index],
                "source_interval": {
                    "start": self._source_start if self._track_frames[index] else None,
                    "end_exclusive": self._source_start + self._track_frames[index]
                    if self._track_frames[index] else None,
                },
                "meaning": "pre-input-gain original" if index == 0 else "caller-limited effect",
                "samples_clipped_outside_unit_range": self._clipped[index],
            }
        metadata = {
            "schema": 1, "samplerate": self.samplerate, "channels": self.channels,
            "format": "PCM16_LE", "sample_width_bytes": 2, "frames": frames,
            "duration_seconds": frames / self.samplerate, "source_interval": interval,
            "tracks": tracks, "alignment": self.alignment,
            "conversion": "clip [-1,1]; round-to-even(sample*32768); saturate int16",
            "parameter_changes": {"file": "parameters.jsonl", "encoding": "jsonl",
                                  "count": self._changes},
            "complete": complete, "finalized": finalized, "saved": False, "reason": reason,
            "stop_reason": self._reason,
            "additional_failures": [str(error) for error in self.secondary_errors],
            "disk_safety": {"reserve_bytes": RESERVE_BYTES,
                            "check_interval_seconds": DISK_CHECK_SECONDS},
        }
        # One bounded metadata snapshot; crash during replacement retains previous snapshot.
        staging = self.group.directory / "metadata.json.pending"
        with staging.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        staging.replace(self.group.metadata_path)
        self.group.frames, self.group.complete = frames, complete

    def _finalize(self):
        try:
            self._finalize_files()
        finally:
            if self._owner_fd is not None:
                os.close(self._owner_fd)
                self._owner_fd = None

    def _finalize_files(self):
        for wav in self._waves:
            try:
                wav.close()
            except (OSError, ValueError, wave.Error) as exc:
                self._set_error(exc)
        if self._journal is not None:
            try:
                self._journal.close()
            except OSError as exc:
                self._set_error(exc)
        try:
            self._write_metadata(finalized=True)
            if self.error is not None and self.group.complete:
                # A join timeout may have been reported while metadata fsync released the GIL.
                self._write_metadata(finalized=True)
        except (OSError, ValueError, TypeError) as exc:
            self._set_error(exc)
            self.group.complete = False
        self.group._finalized = True
        self.group.frames = self.frames_written

    def finish(self, *, complete: bool, reason: str | None = None) -> RecordingGroup:
        """Stop after producer quiescence. First finalization is terminal/idempotent.

        Safe before start and after failed start. Repeated calls preserve the
        original completeness/reason and do not touch saved/discarded temp data.
        Errors remain available through check_health; timeout raises immediately
        and retains the thread handle for a later finish/join.
        """
        if type(complete) is not bool or (reason is not None and not isinstance(reason, str)):
            raise ValueError("finish requires boolean completeness and an optional text reason")
        if self.group._finalized and (
            self._thread is None or not self._thread.is_alive()
        ):
            return self.group
        self._accepting = False
        self._requested_complete = complete
        self._reason = reason
        self._stop = True
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(JOIN_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                error = TimeoutError("Recording disk writer did not finish; handle/data retained")
                self._set_error(error)
                raise error
            if self.error is not None and self.group.complete:
                self.group.complete = False
                try:
                    self._write_metadata(finalized=True)
                except (OSError, ValueError, TypeError) as exc:
                    self._set_error(exc)
        elif not self.group._finalized:
            self._requested_complete = False
            self._reason = reason or "Recording was never started"
            self._finalize()
        return self.group
