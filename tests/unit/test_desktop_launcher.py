import importlib.util
import ctypes
from ctypes.util import find_library
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("create_desktop_launcher", ROOT / "scripts/create_desktop_launcher.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class DesktopLauncherTests(unittest.TestCase):
    def test_generated_entry_uses_explicit_interpreter_no_cwd_or_automatic_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "voice changer.desktop"
            launcher.create_launcher(Path(sys.executable), output)
            text = output.read_text()
            self.assertIn("Terminal=true", text)
            self.assertNotIn("Path=", text)
            self.assertNotIn("--prototype", text)
            command = next(line[5:] for line in text.splitlines() if line.startswith("Exec="))
            args = shlex.split(command) + ["--help"]
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            result = subprocess.run(args, cwd=temporary, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Desktop DSP demo", result.stdout)
            self.assertTrue(output.stat().st_mode & 0o111)
            with self.assertRaises(FileExistsError):
                launcher.create_launcher(Path(sys.executable), output)

    def test_missing_interpreter_does_not_create_success_shaped_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bad.desktop"
            with self.assertRaisesRegex(ValueError, "executable Python"):
                launcher.create_launcher(Path(temporary) / "missing", output)
            self.assertFalse(output.exists())

    def test_percent_in_executable_is_explicitly_rejected_not_generated(self):
        with tempfile.TemporaryDirectory() as temporary:
            python = Path(temporary) / "python%name"
            python.symlink_to(sys.executable)
            output = Path(temporary) / "bad.desktop"
            with self.assertRaisesRegex(ValueError, "unsupported by GLib"):
                launcher.create_launcher(python, output)
            self.assertFalse(output.exists())

    def test_exec_quoting_handles_spaces_reserved_characters_and_field_codes(self):
        self.assertEqual(launcher.exec_argument("/path with %name/$value"), '"/path with %%name/\\\\$value"')
        for value in ("bad\npath", "bad\rpath", "bad\0path"):
            with self.assertRaises(ValueError):
                launcher.exec_argument(value)

    def test_launch_dependency_failure_is_explicit_and_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            shadow = Path(temporary) / "PySide6.py"
            shadow.write_text("raise ModuleNotFoundError('synthetic missing Qt', name='PySide6')\n")
            env = dict(
                os.environ, PYTHONPATH=temporary, PYTHONDONTWRITEBYTECODE="1",
                XDG_RUNTIME_DIR=temporary,
            )
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "scripts/launch_voice_changer.py")],
                cwd=temporary, env=env, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("PySide6 is missing", result.stderr)
            self.assertIn("exit status: 1", result.stderr)

    def test_real_desktop_exec_expansion_preserves_actual_paths(self):
        gio_name, object_name = find_library("gio-2.0"), find_library("gobject-2.0")
        if not gio_name or not object_name:
            self.skipTest("Existing GLib desktop parser unavailable; native Exec not verified")
        with tempfile.TemporaryDirectory(prefix="voice-changer-exec-") as temporary:
            root = Path(temporary)
            app = root / 'source with $ " and % characters'
            (app / "scripts").mkdir(parents=True)
            (app / "src/voice_changer/app").mkdir(parents=True)
            (app / "src/voice_changer/app/main.py").touch()
            facts = root / "facts.json"
            probe = app / "scripts/launch_voice_changer.py"
            probe.write_text(
                "import json,pathlib,sys\ntarget=pathlib.Path(" + repr(str(facts))
                + ")\npending=target.with_suffix('.tmp')\n"
                "pending.write_text(json.dumps([sys.executable,sys.argv[0]]))\npending.replace(target)\n",
            )
            python = root / 'python with $ " characters'
            python.symlink_to(sys.executable)
            output = root / "probe.desktop"
            launcher.create_launcher(python, output, app_root=app)
            output.write_text(output.read_text().replace("Terminal=true", "Terminal=false"))
            gio, gobject = ctypes.CDLL(gio_name), ctypes.CDLL(object_name)
            gio.g_desktop_app_info_new_from_filename.argtypes = [ctypes.c_char_p]
            gio.g_desktop_app_info_new_from_filename.restype = ctypes.c_void_p
            gio.g_app_info_launch.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ]
            gio.g_app_info_launch.restype = ctypes.c_int
            gobject.g_object_unref.argtypes = [ctypes.c_void_p]
            gobject.g_object_unref.restype = None
            info = gio.g_desktop_app_info_new_from_filename(str(output).encode())
            self.assertTrue(info, "Native desktop parser rejected the entry")
            try:
                error = ctypes.c_void_p()
                self.assertTrue(gio.g_app_info_launch(info, None, None, ctypes.byref(error)))
                deadline = time.monotonic() + 5
                while not facts.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(facts.exists(), "Argument-only probe did not complete")
                self.assertEqual(json.loads(facts.read_text()), [str(python), str(probe)])
            finally:
                gobject.g_object_unref(info)
