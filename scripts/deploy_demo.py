#!/usr/bin/env python3
"""Deploy this demo to a fresh directory and optionally install a desktop entry.

The script never starts audio, never changes PipeWire or system volume, and
never touches an existing service. It only copies the source tree, creates a
virtual environment, and writes a desktop entry on request.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
COPY_ITEMS = ("src", "scripts", "tests", "docs", "install.sh", "README.md", "README_zh.md",
              "requirements-prototype.txt",
              "requirements-m0-a.txt", "requirements-backend-tests.txt", "requirements-ai.txt")
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git", "history", "saved", "data"}


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for item in COPY_ITEMS:
        source = root / item
        if source.is_file():
            files.append(source)
            continue
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            if SKIP_PARTS & set(path.relative_to(root).parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            files.append(path)
    return files


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_tree(root: Path, destination: Path) -> dict[str, str]:
    """Copy the source tree and return relative path digests of what was written."""
    destination.mkdir(parents=True, exist_ok=False)
    written: dict[str, str] = {}
    for source in _iter_files(root):
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        written[str(relative)] = file_digest(target)
    return written


def build_archive(root: Path, archive: Path) -> Path:
    """Write a reproducible-ish tar.gz of the copyable tree."""
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")
    with tarfile.open(archive, "w:gz") as bundle:
        for source in _iter_files(root):
            bundle.add(source, arcname=str(Path(root.name) / source.relative_to(root)))
    return archive


def create_venv(python: Path, destination: Path) -> Path:
    """Create a venv that reuses system Qt and NumPy; do not install Qt again."""
    subprocess.run(
        [str(python), "-m", "venv", "--system-site-packages", str(destination)],
        check=True,
    )
    return destination / "bin" / "python"


def register_source(venv: Path, app_root: Path) -> None:
    """Make ``python -m voice_changer`` work without callers setting PYTHONPATH."""
    lib = next((venv / "lib").glob("python3*"), None)
    if lib is None:
        raise RuntimeError(f"venv has no python library directory: {venv}")
    site_packages = lib / "site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)
    (site_packages / "voice-changer-demo.pth").write_text(
        str((app_root / "src").resolve()) + "\n", encoding="utf-8",
    )


def write_launcher(python: Path, app_root: Path, output: Path) -> None:
    sys.path.insert(0, str(app_root / "scripts"))
    try:
        import create_desktop_launcher as launcher
    finally:
        sys.path.pop(0)
    launcher.create_launcher(python, output, app_root=app_root)


def default_destination() -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path.home() / f"voice-changer-{stamp}"


class Steps:
    def __init__(self, *, dry_run: bool, log=print) -> None:
        self.dry_run = dry_run
        self.log = log

    def run(self, command: Sequence[str]) -> None:
        shown = " ".join(shlex.quote(part) for part in command)
        if self.dry_run:
            self.log(f"[dry-run] {shown}")
            return
        self.log(f"[run] {shown}")
        subprocess.run(list(command), check=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy the demo to a new directory, create its venv and optionally "
                    "write a desktop launcher. No audio is started.",
    )
    parser.add_argument("--destination", type=Path, help="new directory (must not exist)")
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="interpreter used to create the venv")
    parser.add_argument("--desktop-entry", type=Path,
                        help="write a .desktop file here (not installed)")
    parser.add_argument("--archive", type=Path, help="also write a tar.gz here")
    parser.add_argument("--no-venv", action="store_true", help="only copy files")
    parser.add_argument(
        "--install-audio", action="store_true",
        help="pip install the hashed audio requirements into the new venv (needs network)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print steps, change nothing")
    arguments = parser.parse_args(argv)

    destination = arguments.destination or default_destination()
    if destination.exists():
        print(f"Refusing to use an existing directory: {destination}", file=sys.stderr)
        return 2
    if not arguments.python.is_file() or not os.access(arguments.python, os.X_OK):
        print(f"Interpreter is missing or not executable: {arguments.python}", file=sys.stderr)
        return 2

    steps = Steps(dry_run=arguments.dry_run)
    if arguments.dry_run:
        print(f"Would copy the demo to {destination}")
    else:
        written = copy_tree(ROOT, destination)
        print(f"Copied {len(written)} files to {destination}")

    python = arguments.python
    if not arguments.no_venv:
        if arguments.dry_run:
            steps.run([str(python), "-m", "venv", "--system-site-packages",
                       str(destination / "venv")])
        else:
            python = create_venv(python, destination / "venv")
            register_source(destination / "venv", destination)
            print(f"Created venv: {python}")

    if arguments.install_audio:
        if arguments.no_venv:
            print("--install-audio needs a venv; drop --no-venv", file=sys.stderr)
            return 2
        steps.run([
            str(python), "-m", "pip", "install", "--require-hashes",
            "-r", str(destination / "requirements-m0-a.txt"),
        ])
        print("Audio packages installed into the demo venv (system Qt/NumPy are reused)")

    if arguments.desktop_entry is not None:
        if destination.exists():
            write_launcher(python, destination, arguments.desktop_entry)
            print(f"Wrote desktop entry: {arguments.desktop_entry}")
        else:
            steps.run(["write-desktop-entry", str(arguments.desktop_entry)])

    if arguments.archive is not None:
        if arguments.dry_run:
            steps.run(["tar", "czf", str(arguments.archive)])
        else:
            build_archive(ROOT, arguments.archive)
            print(f"Wrote archive: {arguments.archive}")

    if not arguments.dry_run:
        print(
            "\nNext steps:\n"
            f"  1. cd {destination} && ./venv/bin/python -B -m voice_changer --version\n"
            "  2. Start the demo from the desktop entry, then confirm devices in the window.\n"
            "  3. Choose the 'Voice Changer Virtual Mic' output to feed other applications.\n"
            "This script did not start audio or change any system audio setting."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
