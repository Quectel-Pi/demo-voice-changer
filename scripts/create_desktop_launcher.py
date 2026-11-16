#!/usr/bin/env python3
"""Generate one explicit desktop entry; never install packages or start audio."""

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import sys


def exec_argument(value: str) -> str:
    if any(character in value for character in ("\n", "\r", "\0")):
        raise ValueError("Desktop paths cannot contain newline or NUL")
    value = value.replace("%", "%%")
    value = value.replace("\\", "\\\\\\\\")
    for character in ('"', "`", "$"):
        value = value.replace(character, "\\\\" + character)
    return '"' + value + '"'


def create_launcher(python: Path, output: Path, *, app_root: Path | None = None) -> None:
    if output.suffix != ".desktop":
        raise ValueError("Output must be a new .desktop file")
    python = python.absolute()
    if "%" in str(python):
        raise ValueError("Desktop executable paths containing % are unsupported by GLib; choose another interpreter path")
    root = (app_root or Path(__file__).resolve().parents[1]).absolute()
    entry = root / "scripts" / "launch_voice_changer.py"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("Choose an existing executable Python interpreter")
    if not entry.is_file() or not (root / "src/voice_changer/app/main.py").is_file():
        raise ValueError("Application source tree or entry script is missing")
    text = (
        "[Desktop Entry]\nType=Application\nVersion=1.0\n"
        "Name=Voice Changer Demo\nName[zh_CN]=变声器演示\n"
        "Comment=Local DSP voice changer; audio off on startup\n"
        f"Exec={exec_argument(str(python))} -B {exec_argument(str(entry))}\n"
        "Icon=audio-input-microphone\nTerminal=true\nCategories=AudioVideo;Audio;\n"
    )
    with output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    output.chmod(0o755)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .desktop file; parent must exist")
    args = parser.parse_args(argv)
    try:
        create_launcher(args.python, args.output)
    except (OSError, ValueError) as error:
        print(f"启动器生成失败 / Launcher generation failed: {error}", file=sys.stderr)
        return 1
    print(f"启动器已生成（未安装/未启动） / Generated only: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
