#!/usr/bin/env python3
"""Render the desktop window offscreen and save a PNG for the README.

The window is built exactly as in the app, but the ``sounddevice`` module is
replaced with the test double from ``tests/integration``. No device is opened,
no audio is captured or played, and nothing is written outside ``--output``.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=src:tests/integration \
        python -B scripts/capture_screenshot.py --output docs/images/main-window.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _fake_device_library() -> ModuleType:
    """Minimal read-only stand-in; mirrors tests/integration/device_fakes.py."""
    library = ModuleType("sounddevice")

    class PortAudioError(Exception):
        pass

    library.PortAudioError = PortAudioError
    library.devices = [
        dict(name="USB Audio Device: Input", hostapi=0, max_input_channels=2,
             max_output_channels=0, default_samplerate=48000.0),
        dict(name="Built-in Audio: Speakers", hostapi=0, max_input_channels=0,
             max_output_channels=2, default_samplerate=48000.0),
        dict(name="Voice Changer Virtual Mic", hostapi=1, max_input_channels=0,
             max_output_channels=2, default_samplerate=48000.0),
    ]
    library.hostapis = [
        dict(name="ALSA", devices=[0, 1]),
        dict(name="PulseAudio", devices=[2]),
    ]
    library.query_devices = lambda: library.devices
    library.query_hostapis = lambda: library.hostapis

    def _refuse(**options):
        raise AssertionError("Screenshot capture must not open a stream")

    library.Stream = _refuse
    return library


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="PNG path to write")
    parser.add_argument("--size", default="640x440", help="window size, e.g. 640x440")
    arguments = parser.parse_args(argv)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.modules["sounddevice"] = _fake_device_library()

    from PySide6.QtCore import QCoreApplication, QEvent, QTimer
    from PySide6.QtWidgets import QApplication

    from voice_changer.ui.main_window import MainWindow

    width, height = (int(part) for part in arguments.size.lower().split("x"))
    app = QApplication.instance() or QApplication([parser.prog])
    window = MainWindow()
    window.resize(width, height)
    window.show()

    app.processEvents()
    for _ in range(30):
        QTimer.singleShot(10, lambda: None)
        app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    image = window.grab().toImage()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(arguments.output), "PNG"):
        print(f"Could not write {arguments.output}", file=sys.stderr)
        return 1

    # Let the window finish its own asynchronous shutdown so no QThread is
    # destroyed while still running.
    window.close()
    deadline = time.monotonic() + 5.0
    while not window._allow_close and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()

    print(f"Wrote {arguments.output} ({image.width()}x{image.height()})")
    print("Offscreen render with a device stand-in; not a hardware screenshot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
