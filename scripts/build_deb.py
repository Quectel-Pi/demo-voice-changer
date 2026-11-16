#!/usr/bin/env python3
"""Build a .deb for the demo using dpkg-deb; never installs or runs audio."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "voice-changer-demo"
VERSION = "1.0.0~demo.1"
ARCHITECTURE = "arm64"
DEPENDS = "python3 (>= 3.13), python3-venv, python3-pyside6.qtwidgets, python3-numpy, " \
          "libasound2t64 | libasound2, librubberband2, pipewire, pipewire-bin"
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git", "history", "saved", "data"}

CONTROL = """Package: {package}
Version: {version}
Section: sound
Priority: optional
Architecture: {arch}
Depends: {depends}
Recommends: gnome-shell-extension-desktop-icons-ng
Maintainer: Voice Changer Demo <noreply@example.invalid>
Description: Local DSP voice changer demo for the Pi M2 desktop
 PySide6 desktop application that applies streaming pitch/formant and
 effect processing to a microphone and can expose the result as a
 PipeWire virtual microphone. Audio is off on startup and requires
 explicit device and volume confirmation.
"""


def _files(root: Path) -> list[Path]:
    result: list[Path] = []
    for item in ("src", "scripts", "tests", "docs", "README.md",
                 "requirements-m0-a.txt", "requirements-prototype.txt", "requirements-ai.txt"):
        source = root / item
        if source.is_file():
            result.append(source)
        elif source.is_dir():
            result.extend(
                path for path in sorted(source.rglob("*"))
                if path.is_file() and not (SKIP_PARTS & set(path.relative_to(root).parts))
                and path.suffix not in {".pyc", ".pyo"}
            )
    return result


def stage(root: Path, stage_dir: Path, *, version: str = VERSION) -> Path:
    """Populate a Debian package tree; returns the package root."""
    app = stage_dir / "opt" / PACKAGE
    for source in _files(root):
        target = app / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    launcher = f"/opt/{PACKAGE}/venv/bin/python"
    app_root = f"/opt/{PACKAGE}"

    desktop_dir = stage_dir / "usr" / "share" / "applications"
    desktop_dir.mkdir(parents=True, exist_ok=True)
    (desktop_dir / f"{PACKAGE}.desktop").write_text(
        "[Desktop Entry]\nType=Application\nVersion=1.0\n"
        "Name=Voice Changer Demo\nName[zh_CN]=变声器演示\n"
        "Comment=Local DSP voice changer; audio off on startup\n"
        f'Exec="{launcher}" -B "{app_root}/scripts/launch_voice_changer.py"\n'
        "Icon=audio-input-microphone\nTerminal=true\nCategories=AudioVideo;Audio;\n",
        encoding="utf-8",
    )

    postinst_dir = stage_dir / "DEBIAN"
    postinst_dir.mkdir(parents=True, exist_ok=True)
    postinst_dir.joinpath("postinst").write_text(
        "#!/bin/sh\nset -e\n"
        f"if [ ! -x {launcher} ]; then\n"
        f"  python3 -m venv --system-site-packages {app_root}/venv\n"
        "fi\n"
        f"{launcher} -B - <<'PY'\n"
        "from pathlib import Path\n"
        "import runpy\n"
        f"deploy = runpy.run_path('{app_root}/scripts/deploy_demo.py')\n"
        f"deploy['register_source'](Path('{app_root}/venv'), Path('{app_root}'))\n"
        "PY\n"
        "# Refresh the desktop database only; no audio or service is touched.\n"
        "if command -v update-desktop-database >/dev/null; then\n"
        "  update-desktop-database -q /usr/share/applications\n"
        "fi\n"
        "exit 0\n", encoding="utf-8")
    postinst_dir.joinpath("postinst").chmod(0o755)
    installed = sum(path.stat().st_size for path in app.rglob("*") if path.is_file())
    control = CONTROL.format(
        package=PACKAGE, version=version, arch=ARCHITECTURE, depends=DEPENDS,
    )
    control += f"Installed-Size: {max(1, installed // 1024)}\n"
    postinst_dir.joinpath("control").write_text(control, encoding="utf-8")
    return stage_dir


def build(root: Path, output: Path, *, version: str = VERSION) -> Path:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with tempfile.TemporaryDirectory() as temporary:
        stage_dir = Path(temporary) / "pkg"
        stage_dir.mkdir()
        stage(root, stage_dir, version=version)
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["dpkg-deb", "--root-owner-group", "--build", str(stage_dir), str(output)],
            check=True, stdout=subprocess.DEVNULL,
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the demo .deb (arm64).")
    parser.add_argument("--output", type=Path,
                        default=Path(f"{PACKAGE}_{VERSION}_{ARCHITECTURE}.deb"))
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--dry-run", action="store_true", help="stage only; do not build")
    arguments = parser.parse_args(argv)
    if shutil.which("dpkg-deb") is None:
        print("dpkg-deb is required to build the package", file=sys.stderr)
        return 1
    if arguments.dry_run:
        with tempfile.TemporaryDirectory() as temporary:
            stage_dir = Path(temporary) / "pkg"
            stage_dir.mkdir()
            stage(ROOT, stage_dir, version=arguments.version)
            count = sum(1 for _ in stage_dir.rglob("*"))
            print(f"Staged {count} entries; nothing was built or installed")
        return 0
    path = build(ROOT, arguments.output, version=arguments.version)
    print(f"Built {path} ({path.stat().st_size} bytes)")
    print("Install with: sudo apt install ./" + path.name)
    print("This script did not install anything, start audio or change settings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
