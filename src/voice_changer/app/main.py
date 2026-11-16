"""Start the local desktop window without initializing any audio backend."""

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from .. import __version__

if TYPE_CHECKING:
    from ..ui.main_window import MainWindow


def main(
    argv: Sequence[str] | None = None,
    *,
    window_ready: Callable[["MainWindow"], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="voice-changer",
        description=(
            "变声器DSP演示版；启动不采集/输出，在窗口确认设备和安全音量后开启监听。 / "
            "Desktop DSP demo. No capture or output on startup; confirm devices "
            "and safe volume in the window before listening."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--llvc-model", metavar="PATH",
        help="enable optional fixed-target AI conversion with a validated 52/104 ms ONNX model",
    )
    parser.add_argument(
        "--tts-model", metavar="DIRECTORY",
        help="local trusted CosyVoice 3 model directory; loaded only on Generate",
    )
    arguments = parser.parse_args(argv)
    if arguments.llvc_model:
        os.environ["VOICE_CHANGER_LLVC_MODEL"] = arguments.llvc_model
    if arguments.tts_model:
        os.environ["VOICE_CHANGER_COSYVOICE_MODEL"] = arguments.tts_model

    # PortAudio caches ALSA device names on first use, so the virtual PCM must
    # exist before anything imports sounddevice.
    from ..audio import virtual_mic
    virtual_mic.prepare()

    try:
        from PySide6.QtCore import QCoreApplication, QEvent, QTimer
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError as error:
        if error.name != "PySide6":
            raise
        print(
            "缺少 PySide6；请按 README 配置 Qt 开发环境。 / "
            "PySide6 is missing; follow the README Qt environment instructions.",
            file=sys.stderr,
        )
        return 1

    from ..ui.main_window import MainWindow

    app = QApplication([parser.prog])
    app.setApplicationName("voice-changer")
    window = MainWindow()
    window.show()
    # XWayland needs the window mapped before accepting maximization.
    QTimer.singleShot(0, window.showMaximized)
    if window_ready is not None:
        window_ready(window)
    result = app.exec()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return result
