#!/usr/bin/env python3
"""Download the optional public HEEVoicechanger ONNX models with SHA-256 checks.

This demo does not distribute these GPL-3.0 weights. Run this script explicitly
to fetch them for local experiments; the AI 变声 page then identifies and names
them but never opens them in the live chain (one 128 ms chunk measured about
415 ms on the Pi M2 CPU, RTF ~3.25).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

sys.dont_write_bytecode = True

REVISION = "31fe76c9cd3c38408b4ab6cb24fe9ca2e98ad690"
BASE_URL = f"https://raw.githubusercontent.com/ultrapg/hevoicechanger/{REVISION}/models"
MODELS: dict[str, tuple[str, int]] = {
    "female_voice_hfg_new.onnx": (
        "e0084cf7eaa0956f284037dad6ac8344563c936132c47d8ed3cb74202d0633b6", 13_756_183,
    ),
    "female_voice_old_hq.onnx": (
        "069819e0589fe5bc7253576daa30ab37abf3c2d7bc82f89e08c2d089f3310f55", 13_718_163,
    ),
    "female_voice_old.onnx": (
        "60d900e8b13523a816bb1b235c9c8614e81f3d0ff96554a9c6ee24d2d2946c84", 13_756_183,
    ),
}
CHUNK_SIZE = 1 << 20
TIMEOUT_SECONDS = 60


def default_destination() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME")
    base = (
        Path(data_home)
        if data_home and Path(data_home).is_absolute()
        else Path.home() / ".local" / "share"
    )
    return base / "voice-changer" / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(name: str, expected: str, size: int, destination: Path) -> Path:
    target = destination / name
    if target.exists():
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"refusing to replace a non-regular file: {target}")
        actual = _sha256(target)
        if actual == expected:
            print(f"present  {target}")
            return target
        raise ValueError(
            f"existing file has unexpected SHA-256 {actual}; move it away first: {target}"
        )

    temporary = destination / f".{name}.part"
    url = f"{BASE_URL}/{name}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            with temporary.open("wb") as stream:
                while chunk := response.read(CHUNK_SIZE):
                    stream.write(chunk)
    except (urllib.error.URLError, OSError) as error:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"download failed for {url}: {error}") from error

    try:
        actual_size = temporary.stat().st_size
        actual = _sha256(temporary)
        if actual_size != size or actual != expected:
            raise ValueError(
                f"download checksum mismatch for {name}: "
                f"{actual_size} bytes, SHA-256 {actual}"
            )
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"fetched  {target}")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models", nargs="*", metavar="MODEL",
        help="model filenames to fetch; default: all three",
    )
    parser.add_argument(
        "--destination", type=Path, default=None,
        help=f"download directory; default: {default_destination()}",
    )
    parser.add_argument(
        "--list", action="store_true", help="list known models without downloading",
    )
    arguments = parser.parse_args(argv)

    names = list(MODELS) if not arguments.models else list(arguments.models)
    unknown = [name for name in names if name not in MODELS]
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}")
    if arguments.list:
        for name, (expected, size) in MODELS.items():
            print(f"{name}\n  sha256 {expected}\n  bytes  {size}")
        return 0

    destination = (arguments.destination or default_destination()).expanduser()
    try:
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            expected, size = MODELS[name]
            _fetch(name, expected, size, destination)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        "These GPL-3.0 weights are not part of this repository. Add a file from the "
        "AI 变声 page: it is identified and named only, and Start stays disabled "
        "because this model is not realtime on the Pi M2 CPU."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
