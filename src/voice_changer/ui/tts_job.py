"""Cancellable, isolated TTS synthesis; never opens an audio device."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import wave

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from ..ai.profiles import VoiceProfile


def audio_info(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as reader:
        rate, frames = reader.getframerate(), reader.getnframes()
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2 or frames <= 0:
            raise ValueError("Expected non-empty mono PCM16 WAV")
        if not 8000 <= rate <= 192000:
            raise ValueError("Unsupported generated sample rate")
        remaining = frames * 2
        while remaining:
            chunk = reader.readframes(min(remaining // 2, 65536))
            if not chunk:
                raise ValueError("Generated WAV is truncated")
            remaining -= len(chunk)
    return rate, frames


def _check_kokoro_memory() -> None:
    try:
        lines = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
        available = next(int(line.split()[1]) for line in lines if line.startswith("MemAvailable:"))
    except (OSError, ValueError, IndexError, StopIteration) as error:
        raise RuntimeError(f"Cannot check available memory for Kokoro: {error}") from error
    if available < 1200 * 1024:
        raise RuntimeError("可用内存不足 1200 MiB，未启动 Kokoro / Less than 1200 MiB available; Kokoro not started")


class TtsJob(QObject):
    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process: QProcess | None = None
        self.result: Path | None = None
        self.sample_rate = 0
        self.frames = 0
        self.message = ""
        self._pending: tempfile.TemporaryDirectory | None = None
        self._result_directory: tempfile.TemporaryDirectory | None = None
        self._cancelled = False
        self._log = ""
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._kill)
        self._timeout_timer = QTimer(self)
        self._timeout_timer.setSingleShot(True)
        self._timeout_timer.timeout.connect(self._timed_out)
        self._timeout = False

    @property
    def busy(self) -> bool:
        return self.process is not None

    def start(self, profile: VoiceProfile, text: str, model_dir: Path,
              *, python: str = sys.executable) -> None:
        self._start(text, python, "voice_changer.ai.cosyvoice_tts", [
            "--model-dir", str(model_dir), "--profile-root", str(profile.directory.parent),
            "--profile-id", profile.profile_id,
        ])

    def start_kokoro(self, voice: str, text: str, model_dir: Path, *, python: str) -> None:
        _check_kokoro_memory()
        self._start(text, python, "voice_changer.ai.kokoro_tts", [
            "--model-dir", str(model_dir), "--voice", voice,
        ], limited=True)

    def _start(self, text: str, python: str, module: str, arguments: list[str],
               *, limited: bool = False) -> None:
        if self.busy:
            raise RuntimeError("Speech generation is already running")
        self._pending = tempfile.TemporaryDirectory(prefix="voice-changer-tts-")
        root = Path(self._pending.name)
        try:
            (root / "text.txt").write_text(text, encoding="utf-8")
        except OSError:
            self._pending.cleanup()
            self._pending = None
            raise
        self._cancelled = False
        self._timeout = False
        self._log = ""
        process = QProcess(self)
        self.process = process
        environment = QProcessEnvironment.systemEnvironment()
        source_root = str(Path(__file__).resolve().parents[2])
        environment.insert("PYTHONPATH", os.pathsep.join(filter(None, (
            source_root, environment.value("PYTHONPATH"),
        ))))
        environment.insert("HF_HUB_OFFLINE", "1")
        environment.insert("TRANSFORMERS_OFFLINE", "1")
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read)
        process.errorOccurred.connect(self._error)
        process.finished.connect(self._finished)
        self.message = "正在加载模型并生成，可点停止取消… / Loading model and generating; Stop cancels…"
        command = [
            python, "-B", "-m", module, *arguments,
            "--text-file", str(root / "text.txt"), "--output", str(root / "generated.wav"),
        ]
        if limited:
            command = [
                "systemd-run", "--user", "--scope", "--quiet",
                "--property=MemoryMax=700M", "--property=MemorySwapMax=0", *command,
            ]
            self._timeout_timer.start(300000)
        process.start(command[0], command[1:])
        self.changed.emit()

    def _read(self) -> None:
        if self.process is not None:
            self._log = (self._log + bytes(self.process.readAllStandardOutput()).decode(
                "utf-8", errors="replace",
            ))[-8192:]

    def _error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._log = self.process.errorString()
            self._finished(1, QProcess.ExitStatus.CrashExit)

    def _finished(self, code: int, exit_status: QProcess.ExitStatus) -> None:
        self._kill_timer.stop()
        self._timeout_timer.stop()
        self._read()
        if self._timeout:
            self.message = "生成超时，已停止；保留上一次结果 / Generation timed out; previous result kept"
        elif self._cancelled:
            self.message = "已取消生成，保留上一次结果 / Generation cancelled; previous result kept"
        elif code != 0 or exit_status != QProcess.ExitStatus.NormalExit:
            self.message = (
                f"朗读生成失败 / Speech generation failed\n"
                f"exit={code}; {exit_status.name}\n{self._log}"
            )
        else:
            path = Path(self._pending.name) / "generated.wav"
            try:
                rate, frames = audio_info(path)
            except (OSError, EOFError, ValueError, wave.Error) as error:
                self.message = f"生成结果无效 / Invalid generated audio: {error}"
            else:
                if self._result_directory is not None:
                    self._result_directory.cleanup()
                self._result_directory, self._pending = self._pending, None
                self.result, self.sample_rate, self.frames = path, rate, frames
                self.message = "生成完成，点击播放试听 / Generated; press Play to listen"
        if self._pending is not None:
            self._pending.cleanup()
            self._pending = None
        self.process.deleteLater()
        self.process = None
        self.changed.emit()

    def cancel(self) -> None:
        if self.process is not None:
            self._cancelled = True
            self.message = "正在停止生成… / Stopping generation…"
            self.process.terminate()
            self._kill_timer.start(2000)
            self.changed.emit()

    def _kill(self) -> None:
        if self.process is not None:
            self.process.kill()

    def _timed_out(self) -> None:
        if self.process is not None:
            self._timeout = True
            self.process.terminate()
            self._kill_timer.start(2000)

    def dispose(self) -> None:
        if self.busy:
            raise RuntimeError("Wait for generation to stop before closing")
        if self._result_directory is not None:
            self._result_directory.cleanup()
            self._result_directory = None
            self.result = None
