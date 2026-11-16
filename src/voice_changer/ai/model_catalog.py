"""Display names and capability notes for known voice models.

Display names are product labels, not proof of speaker identity: the same public
project can target one voice at several quality levels. Users may rename any
identified model in the page or by editing a small JSON file; names never grant
live-conversion capability, and this module never downloads or runs a model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

from . import hevc
from .llvc import MODEL_SHA256, STABLE_MODEL_SHA256

SCHEMA = "voice-changer model names v1"
NAME_LIMIT = 48


@dataclass(frozen=True)
class CatalogEntry:
    entry_id: str
    label: str
    model_type: str
    sha256: tuple[str, ...]
    filenames: tuple[str, ...]
    description: str
    live_reason: str | None


ENTRIES = (
    CatalogEntry(
        "llvc_koe",
        "晓晓 / Xiaoxiao",
        "llvc_onnx",
        (MODEL_SHA256, STABLE_MODEL_SHA256),
        ("llvc-stateful-8.onnx",),
        "LLVC fixed target voice with 52/104 ms model blocks; accepted for live conversion.",
        None,
    ),
    CatalogEntry(
        "hevc_hfg_new",
        "自然女生 / Natural Female",
        "hevc_onnx",
        ("e0084cf7eaa0956f284037dad6ac8344563c936132c47d8ed3cb74202d0633b6",),
        ("female_voice_hfg_new.onnx",),
        "HEVoicechanger public female voice (200-epoch teacher-student distillation, "
        "HiFi-GAN topology). Interface checked without inference; quality is unheard here.",
        hevc.LIVE_REASON,
    ),
    CatalogEntry(
        "hevc_old_hq",
        "清亮女声 / Bright Female",
        "hevc_onnx",
        ("069819e0589fe5bc7253576daa30ab37abf3c2d7bc82f89e08c2d089f3310f55",),
        ("female_voice_old_hq.onnx",),
        "HEVoicechanger public high-quality baseline female voice. Interface checked "
        "without inference; quality is unheard here.",
        hevc.LIVE_REASON,
    ),
    CatalogEntry(
        "hevc_old",
        "柔和女声 / Gentle Female",
        "hevc_onnx",
        ("60d900e8b13523a816bb1b235c9c8614e81f3d0ff96554a9c6ee24d2d2946c84",),
        ("female_voice_old.onnx",),
        "HEVoicechanger public standard baseline female voice. Interface checked without "
        "inference; quality is unheard here.",
        hevc.LIVE_REASON,
    ),
)

ENTRY_BY_ID = {entry.entry_id: entry for entry in ENTRIES}


def names_path() -> Path:
    home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(home) if home and Path(home).is_absolute() else Path.home() / ".config"
    return base / "voice-changer" / "model-names.json"


def _valid_label(value) -> str:
    if not isinstance(value, str):
        raise ValueError("模型名称必须是文本 / A model name must be text")
    text = value.strip()
    if not text:
        raise ValueError("模型名称不能为空 / A model name cannot be empty")
    if len(text) > NAME_LIMIT:
        raise ValueError(f"模型名称最多 {NAME_LIMIT} 个字符 / At most {NAME_LIMIT} characters")
    if any(ord(character) < 32 for character in text):
        raise ValueError("模型名称不能包含换行或控制字符 / No control characters")
    return text


def load_name_overrides() -> tuple[dict[str, str], str | None]:
    """Read user display names; a damaged file is reported, not fatal."""
    path = names_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, None
    except (OSError, UnicodeDecodeError):
        return {}, f"无法读取模型名称文件 / Cannot read model names: {path}"
    except json.JSONDecodeError:
        return {}, f"模型名称文件不是有效 JSON / Model name file is not valid JSON: {path}"
    if not isinstance(payload, dict) or payload.get("format") != SCHEMA:
        return {}, f"模型名称文件格式不匹配 / Model name file format mismatch: {path}"
    names = payload.get("names")
    if not isinstance(names, dict):
        return {}, f"模型名称列表缺失 / Model name list is missing: {path}"
    overrides = {}
    for key, value in names.items():
        try:
            overrides[key] = _valid_label(value)
        except ValueError:
            continue
    return overrides, None


def save_name_overrides(overrides: dict[str, str]) -> None:
    path = names_path()
    payload = {"format": SCHEMA, "names": dict(overrides)}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError as error:
        raise ValueError(f"无法保存模型名称 / Cannot save model names: {error}") from error


_RKNN_SUFFIXES = ("_rknn", "-rknn", "_rknn_fp16", "_rknn_fp16_model")


def _compatible(model_type: str, candidate_type: str) -> bool:
    if model_type == candidate_type:
        return True
    # A converted HEVC package keeps the display name of its source ONNX graph.
    return model_type == "hevc_rknn" and candidate_type == "hevc_onnx"


def _names(path: Path) -> set[str]:
    name = Path(path).name.casefold()
    stem = Path(path).stem.casefold()
    names = {name, stem}
    for suffix in _RKNN_SUFFIXES:
        if stem.endswith(suffix):
            names.add(stem[: -len(suffix)])
    return names


def find_entry(path: Path, sha256: str | None = None,
               model_type: str | None = None) -> CatalogEntry | None:
    """Match a known model by digest first, then by its published filename."""
    names = _names(Path(path))
    for candidate in ENTRIES:
        if sha256 and sha256 in candidate.sha256 and (
                model_type is None or _compatible(model_type, candidate.model_type)):
            return candidate
    for candidate in ENTRIES:
        published = {item.casefold() for item in candidate.filenames}
        published |= {Path(item).stem.casefold() for item in candidate.filenames}
        if names & published and (model_type is None or _compatible(model_type, candidate.model_type)):
            return candidate
    return None


def display_label(path: Path, sha256: str | None = None, model_type: str | None = None,
                  overrides: dict[str, str] | None = None) -> str:
    overrides = overrides or {}
    entry = find_entry(path, sha256, model_type)
    keys = [key for key in (
        entry.entry_id if entry is not None else None,
        f"sha:{sha256}" if sha256 else None,
    ) if key]
    for key in keys:
        if key in overrides:
            return overrides[key]
    if entry is not None:
        return entry.label
    return Path(path).stem


def configured_label(overrides: dict[str, str] | None = None) -> str:
    """Label for the model passed on the command line or in the environment."""
    overrides = overrides or {}
    entry = ENTRY_BY_ID["llvc_koe"]
    return overrides.get(entry.entry_id, entry.label)


def rename(path: Path, sha256: str | None, model_type: str | None, label: str,
           overrides: dict[str, str]) -> str:
    """Persist a display name (empty restores the default) and return the new label."""
    entry = find_entry(path, sha256, model_type)
    keys = [key for key in (
        entry.entry_id if entry is not None else None,
        f"sha:{sha256}" if sha256 else None,
    ) if key]
    if not keys:
        raise ValueError("此模型尚未识别，暂时无法重命名 / This model is not identified yet")
    updated = dict(overrides)
    text = (label or "").strip()
    if text:
        updated[keys[0]] = _valid_label(text)
        for key in keys[1:]:
            updated.pop(key, None)
    else:
        for key in keys:
            updated.pop(key, None)
    save_name_overrides(updated)
    overrides.clear()
    overrides.update(updated)
    return display_label(path, sha256, model_type, overrides)
