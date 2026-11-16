"""Local voice profiles: authorised reference recordings plus capability metadata.

A profile only becomes usable for read-aloud or voice conversion when a
registered backend actually prepares it. Until then it is "reference only".
Deletion is limited to the files this module created inside one owned directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import uuid
import wave

SCHEMA = "voice-changer voice profile v1"
_OWNER = "voice-changer voice profile v1\n"
_NAMES = re.compile(r"voice-[0-9a-f]{32}\Z")
_OWNED_FILES = (".owner", "reference.wav", "profile.json")
NAME_LIMIT = 64
RELATIONS = ("self", "authorized")
CONSENT_STATEMENT = (
    "说话人本人或已明确同意者，同意录制此段声音并仅在本机用于音色生成。\n"
    "The speaker is me or has explicitly agreed to this recording being used "
    "for local voice generation only."
)


class ProfileError(RuntimeError):
    """A profile operation failed; the source recording is never modified."""


@dataclass(frozen=True)
class VoiceProfile:
    profile_id: str
    name: str
    directory: Path
    created_at: str = ""
    relation: str = ""
    prompt_text: str = ""
    duration: float = 0.0
    capabilities: dict[str, bool] = field(default_factory=dict)
    problem: str | None = None

    @property
    def damaged(self) -> bool:
        return self.problem is not None

    def supports(self, feature: str) -> bool:
        return not self.damaged and self.capabilities.get(feature) is True

    @property
    def reference_path(self) -> Path:
        return self.directory / "reference.wav"


@dataclass(frozen=True)
class DeleteResult:
    removed: tuple[str, ...]
    residual: tuple[str, ...]


def default_root() -> Path:
    home = os.environ.get("XDG_DATA_HOME")
    base = Path(home) if home and Path(home).is_absolute() else Path.home() / ".local/share"
    return base / "voice-changer" / "voices"


def capability_text(profile: VoiceProfile) -> str:
    if profile.damaged:
        return f"档案损坏，不可使用 / Damaged profile: {profile.problem}"
    labels = []
    if profile.supports("tts"):
        labels.append("可用于 AI 朗读 / AI read-aloud")
    if profile.supports("vc"):
        labels.append("可用于 AI 变声 / AI voice conversion")
    if not labels:
        return (
            "参考音色：配置 CosyVoice 3 并提供准确原文后可用于朗读；未接入模型的能力不启用\n"
            "Reference voice: read-aloud requires CosyVoice 3 and an exact transcript; other capabilities need a backend"
        )
    return "；".join(labels)


def validate_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ProfileError("请输入档案名称 / Enter a profile name")
    if len(cleaned) > NAME_LIMIT:
        raise ProfileError(f"名称最多 {NAME_LIMIT} 个字符 / Name is limited to {NAME_LIMIT} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in cleaned):
        raise ProfileError("名称不能包含控制字符 / Name contains control characters")
    return cleaned


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wave_info(path: Path) -> tuple[int, int, int]:
    try:
        with wave.open(str(path), "rb") as reader:
            return reader.getframerate(), reader.getnchannels(), reader.getnframes()
    except (OSError, EOFError, wave.Error) as error:
        raise ProfileError(f"参考录音不可读 / Unreadable reference recording: {error}") from error


def _regular(path: Path) -> None:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ProfileError(f"不是普通文件 / Not a regular file: {path.name}")


class ProfileStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()

    def create(
        self, name: str, *, relation: str, consent: bool, prompt_text: str, source: Path,
    ) -> VoiceProfile:
        """Copy ``source`` into a new owned profile; ``source`` itself is left untouched."""
        name = validate_name(name)
        if consent is not True:
            raise ProfileError("需要说话人授权确认 / Speaker consent is required")
        if relation not in RELATIONS:
            raise ProfileError("请选择说话人关系 / Choose the speaker relationship")
        source = Path(source)
        _regular(source)
        rate, channels, frames = _wave_info(source)
        if frames <= 0 or rate <= 0:
            raise ProfileError("参考录音为空 / The reference recording is empty")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        profile_id = uuid.uuid4().hex
        partial = self.root / f".voice-{profile_id}.partial"
        final = self.root / f"voice-{profile_id}"
        partial.mkdir(mode=0o700)
        try:
            (partial / ".owner").write_text(_OWNER, encoding="utf-8")
            target = partial / "reference.wav"
            with source.open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, 1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            digest = _sha256(target)
            if digest != _sha256(source):
                raise ProfileError("复制后的录音校验不一致 / Copied recording failed verification")
            created = datetime.now(timezone.utc).isoformat(timespec="seconds")
            metadata = {
                "schema": SCHEMA,
                "id": profile_id,
                "name": name,
                "created_at": created,
                "consent": {
                    "confirmed": True, "relation": relation,
                    "statement": CONSENT_STATEMENT, "confirmed_at": created,
                },
                "prompt_text": prompt_text,
                "reference": {
                    "file": "reference.wav", "sha256": digest,
                    "samplerate": rate, "channels": channels, "frames": frames,
                },
                "capabilities": {"tts": False, "vc": False},
                "status": "reference_only",
            }
            with (partial / "profile.json").open("x", encoding="utf-8") as writer:
                json.dump(metadata, writer, ensure_ascii=False, indent=2)
                writer.flush()
                os.fsync(writer.fileno())
            os.rename(partial, final)
        except BaseException as error:
            self._remove_partial(partial)
            if isinstance(error, ProfileError):
                raise
            if isinstance(error, OSError):
                raise ProfileError(f"建立档案失败 / Could not create profile: {error}") from error
            raise
        return self._load(final)

    def _remove_partial(self, partial: Path) -> None:
        for filename in _OWNED_FILES:
            try:
                (partial / filename).unlink()
            except FileNotFoundError:
                pass
            except OSError:
                return
        try:
            partial.rmdir()
        except OSError:
            pass

    def _load(self, directory: Path) -> VoiceProfile:
        profile_id = directory.name.removeprefix("voice-")
        try:
            if not stat.S_ISDIR(directory.lstat().st_mode):
                raise ProfileError("不是目录 / Not a directory")
            for filename in _OWNED_FILES:
                _regular(directory / filename)
            if (directory / ".owner").read_text(encoding="utf-8") != _OWNER:
                raise ProfileError("所有权标记无效 / Invalid ownership marker")
            metadata = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
            if metadata.get("schema") != SCHEMA or metadata.get("id") != profile_id:
                raise ProfileError("档案元数据不匹配 / Metadata mismatch")
            reference = metadata["reference"]
            capabilities = metadata["capabilities"]
            if not all(isinstance(capabilities.get(key), bool) for key in ("tts", "vc")):
                raise ProfileError("能力标注无效 / Invalid capability metadata")
            rate = int(reference["samplerate"])
            frames = int(reference["frames"])
            return VoiceProfile(
                profile_id=profile_id,
                name=validate_name(str(metadata["name"])),
                directory=directory,
                created_at=str(metadata["created_at"]),
                relation=str(metadata["consent"]["relation"]),
                prompt_text=str(metadata.get("prompt_text", "")),
                duration=frames / rate if rate > 0 else 0.0,
                capabilities={"tts": capabilities["tts"], "vc": capabilities["vc"]},
            )
        except (OSError, ValueError, KeyError, TypeError, ProfileError) as error:
            return VoiceProfile(profile_id, profile_id[:8], directory, problem=str(error))

    def list(self) -> tuple[VoiceProfile, ...]:
        if not self.root.is_dir():
            return ()
        profiles = [
            self._load(path) for path in sorted(self.root.iterdir())
            if _NAMES.fullmatch(path.name)
        ]
        return tuple(sorted(profiles, key=lambda profile: (profile.created_at, profile.name)))

    def delete(self, profile: VoiceProfile) -> DeleteResult:
        """Remove only this module's files from one owned directory; report leftovers."""
        directory = profile.directory
        if directory.parent != self.root or not _NAMES.fullmatch(directory.name):
            raise ProfileError("不是本应用的音色档案目录 / Not an owned voice profile directory")
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise ProfileError("档案目录无效 / Invalid profile directory")
        owner = directory / ".owner"
        try:
            _regular(owner)
            owned = owner.read_text(encoding="utf-8") == _OWNER
        except (OSError, ProfileError):
            owned = False
        if not owned:
            raise ProfileError("缺少所有权标记，未删除任何文件 / Missing ownership marker; nothing deleted")
        removed = []
        for filename in ("reference.wav", "profile.json", ".owner"):
            path = directory / filename
            try:
                if path.is_symlink() or path.is_file():
                    path.unlink()
                    removed.append(filename)
            except OSError:
                pass
        residual = tuple(sorted(item.name for item in directory.iterdir()))
        if not residual:
            directory.rmdir()
        return DeleteResult(tuple(removed), residual)
