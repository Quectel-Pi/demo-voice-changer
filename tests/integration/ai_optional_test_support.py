"""Shared subprocess isolation for P0 tests without optional AI dependencies."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
ISOLATION = r"""
import importlib
import importlib.abc
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
sys.path[:0] = [
    str(root / "src"), str(root / "tests" / "unit"),
    str(root / "tests" / "integration"), *json.loads(sys.argv[2]),
]
blocked = (
    "sherpa_onnx", "onnxruntime", "kokoro", "torch", "torchaudio",
    "tensorflow", "transformers", "cosyvoice", "f5_tts", "huggingface_hub",
    "vocos", "misaki", "pypinyin", "jieba", "phonemizer",
)

class NoAI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == item or fullname.startswith(item + ".") for item in blocked):
            raise ModuleNotFoundError("AI dependency intentionally unavailable", name=fullname)
        return None

sys.meta_path.insert(0, NoAI())
for name in blocked:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
    else:
        raise AssertionError("AI import guard failed for " + name)

def assert_ai_unloaded():
    if any(name == item or name.startswith(item + ".") for name in sys.modules for item in blocked):
        raise AssertionError("an AI dependency was loaded")

def audit(event, args):
    if event.startswith(("socket.", "subprocess.")) or event in (
        "os.system", "os.fork", "os.exec", "os.posix_spawn",
    ):
        raise RuntimeError("operation outside test scope: " + event)
    if event == "ctypes.dlopen" and args and args[0] is not None:
        name = os.fsdecode(args[0]).lower()
        if any(item in name for item in ("portaudio", "libasound", "libpulse", "libjack")):
            raise RuntimeError("audio-device library load outside test scope")

sys.addaudithook(audit)
"""


def run_without_ai(body: str) -> subprocess.CompletedProcess[str]:
    paths = [os.path.abspath(path or os.getcwd()) for path in sys.path]
    with tempfile.TemporaryDirectory(prefix="voice-changer-no-ai-") as runtime:
        env = {
            "HOME": runtime,
            "XDG_CONFIG_HOME": runtime,
            "XDG_DATA_HOME": runtime,
            "XDG_CACHE_HOME": runtime,
            "XDG_RUNTIME_DIR": runtime,
            "TMPDIR": runtime,
            "QT_QPA_PLATFORM": "offscreen",
            "LANG": "C.UTF-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        return subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", ISOLATION + body, str(ROOT), json.dumps(paths)],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=30, check=False,
        )
