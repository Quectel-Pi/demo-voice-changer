#!/usr/bin/env python3
"""Launch the Pi M2 LLVC demo with the system Qt/Python runtime."""

import argparse
import os
from pathlib import Path
import platform
import sys


def main() -> int:
    if sys.version_info[:2] != (3, 13) or platform.machine().lower() not in ("aarch64", "arm64"):
        print(
            "Pi M2 LLVC requires the Debian 13 aarch64 Python 3.13 runtime. "
            "Do not mix system PySide6 with the Python 3.10 AI environment.",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", type=Path,
        help="validated 52/104 ms LLVC ONNX graph or an experimental RK3576 .rknn artifact",
    )
    arguments = parser.parse_args()
    model = arguments.model.expanduser()
    if not model.is_file():
        print(f"LLVC model not found: {model}", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    os.environ["VOICE_CHANGER_LLVC_MODEL"] = str(model.resolve())
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    from voice_changer.app.main import main as run_application

    return run_application(["--llvc-model", str(model.resolve())])


if __name__ == "__main__":
    raise SystemExit(main())
