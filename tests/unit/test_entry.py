"""CLI and import isolation checks; Qt and audio libraries are not required."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BLOCK_QT = """
import importlib.abc
import sys

class NoQt(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "PySide6" or fullname.startswith("PySide6."):
            raise ModuleNotFoundError("Qt intentionally unavailable", name="PySide6")

sys.meta_path.insert(0, NoQt())
"""


class EntryTests(unittest.TestCase):
    def run_entry(self, *args, script=None):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        command = [sys.executable, "-B"]
        if script is None:
            command.extend(["-m", "voice_changer", *args])
        else:
            command.extend(["-c", script, *args])
        with tempfile.TemporaryDirectory(prefix="voice-changer-entry-runtime-") as runtime:
            env["XDG_RUNTIME_DIR"] = runtime
            return subprocess.run(
                command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=15,
            )

    def test_help_without_qt(self):
        result = self.run_entry(script=BLOCK_QT + """
from voice_changer.app.main import main
raise SystemExit(main(["--help"]))
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No capture or output", " ".join(result.stdout.split()))
        self.assertIn("before listening", " ".join(result.stdout.split()))
        self.assertEqual(result.stderr, "")

    def test_module_help(self):
        result = self.run_entry("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("voice-changer", result.stdout)
        self.assertIn("--tts-model", result.stdout)
        self.assertNotIn("--tts-preset-root", result.stdout)

    def test_help_does_not_invoke_optional_window_hook_or_import_qt(self):
        result = self.run_entry(script=BLOCK_QT + """
from voice_changer.app.main import main
def unexpected(window):
    raise AssertionError("window hook must not run for --help")
raise SystemExit(main(["--help"], window_ready=unexpected))
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("voice-changer", result.stdout)

    def test_invalid_args_without_qt(self):
        for option in ("--prototype", "--monitor", "--unknown"):
            with self.subTest(option=option):
                result = self.run_entry(script=BLOCK_QT + f"""
from voice_changer.app.main import main
raise SystemExit(main([{option!r}]))
""")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("unrecognized arguments", result.stderr)
                self.assertNotIn("PySide6 is missing", result.stderr)

    def test_missing_qt_is_visible_failure(self):
        result = self.run_entry(script=BLOCK_QT + """
import runpy
runpy.run_module("voice_changer", run_name="__main__")
""")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("PySide6 is missing", result.stderr)
        self.assertIn("README", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_unrelated_import_failure_is_not_relabelled(self):
        result = self.run_entry(script=BLOCK_QT.replace(
            'name="PySide6"', 'name="unrelated_dependency"'
        ) + """
from voice_changer.app.main import main
raise SystemExit(main([]))
""")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Traceback", result.stderr)
        self.assertIn("Qt intentionally unavailable", result.stderr)
        self.assertNotIn("PySide6 is missing", result.stderr)

    def test_import_has_no_runtime_side_effects(self):
        result = self.run_entry(script="""
import sys
import voice_changer
import voice_changer.__main__
import voice_changer.app.main
for name in sys.modules:
    assert not name.startswith(("PySide6", "sounddevice", "numpy", "voice_changer.audio",
                               "voice_changer.dsp", "voice_changer.ai",
                               "voice_changer.ui.prototype")), name
""")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_startup_preparation_uses_an_isolated_runtime(self):
        result = self.run_entry(script="""
import os
from pathlib import Path
from voice_changer.audio import virtual_mic
runtime = Path(os.environ["XDG_RUNTIME_DIR"])
assert runtime.is_dir()
assert not (runtime / "pipewire-0").exists()
assert not virtual_mic.prepare().available
assert not (runtime / "voice-changer").exists()
""")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
