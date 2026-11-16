"""Persist the interface language in the existing XDG voice-changer setting."""

import json
import os
from pathlib import Path


LANGUAGES = ("zh", "en")
DEFAULT_LANGUAGE = "zh"


def settings_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME", "")
    root = Path(base) if os.path.isabs(base) else Path.home() / ".config"
    return root / "voice-changer" / "prototype-ui.json"


def load_language() -> tuple[str, str, str]:
    """Return (language, problem key or "", error detail); missing file is not a problem."""
    path = settings_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DEFAULT_LANGUAGE, "", ""
    except (OSError, UnicodeDecodeError) as error:
        return DEFAULT_LANGUAGE, "settings_read_failed", str(error)
    try:
        language = json.loads(raw).get("language")
    except (ValueError, AttributeError):
        language = None
    if language not in LANGUAGES:
        return DEFAULT_LANGUAGE, "settings_invalid", ""
    return language, "", ""


def save_language(language: str) -> None:
    """Atomically write the language choice; raises OSError for the caller to show."""
    if language not in LANGUAGES:
        raise ValueError(language)
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps({"language": language}) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise

