"""Debian package staging: paths, dependency names and no personal data."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("build_deb", ROOT / "scripts/build_deb.py")
deb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deb)


class StageTests(unittest.TestCase):
    def _stage(self, temporary: str) -> Path:
        stage_dir = Path(temporary) / "pkg"
        stage_dir.mkdir()
        return deb.stage(ROOT, stage_dir)

    def test_stages_application_under_opt_and_entry_in_share(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._stage(temporary)
            self.assertTrue((root / "opt" / deb.PACKAGE / "src" / "voice_changer" / "app" / "main.py").is_file())
            self.assertTrue((root / "opt" / deb.PACKAGE / "src" / "voice_changer" / "ai" / "llvc.py").is_file())
            self.assertTrue((root / "opt" / deb.PACKAGE / "requirements-ai.txt").is_file())
            self.assertTrue((root / "usr" / "share" / "applications" / f"{deb.PACKAGE}.desktop").is_file())
            self.assertTrue((root / "DEBIAN" / "control").is_file())
            self.assertTrue((root / "DEBIAN" / "postinst").is_file())
            mode = (root / "DEBIAN" / "postinst").stat().st_mode
            self.assertTrue(mode & 0o111)

    def test_never_includes_recordings_caches_or_virtualenvs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._stage(temporary)
            names = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()]
            joined = " ".join(names)
            self.assertNotIn("__pycache__", joined)
            self.assertNotIn(".venv/", joined)
            self.assertNotIn("/saved/", joined)
            self.assertFalse(any(name.endswith(".pyc") for name in names))

    def test_control_uses_only_verified_board_package_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._stage(temporary)
            control = (root / "DEBIAN" / "control").read_text(encoding="utf-8")
        self.assertIn(f"Package: {deb.PACKAGE}", control)
        self.assertIn(f"Architecture: {deb.ARCHITECTURE}", control)
        for name in ("python3-pyside6.qtwidgets", "python3-numpy", "libasound2t64",
                     "librubberband2", "pipewire", "python3-venv"):
            self.assertIn(name, control)
        self.assertIn("libasound2t64 | libasound2", control)
        self.assertIn("Installed-Size:", control)

    def test_postinst_creates_venv_but_never_touches_audio_or_services(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._stage(temporary)
            script = (root / "DEBIAN" / "postinst").read_text(encoding="utf-8")
        self.assertIn("python3 -m venv --system-site-packages", script)
        self.assertIn("register_source", script)
        for forbidden in ("systemctl", "pactl", "pipewire", "apt install", "rm -rf"):
            self.assertNotIn(forbidden, script)

    def test_postinst_registers_source_for_direct_module_entry_and_repairs_old_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._stage(temporary)
            app = root / "opt" / deb.PACKAGE
            environment = app / "venv"
            venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
            python = environment / "bin" / "python"
            library = next((environment / "lib").glob("python3*")) / "site-packages"
            pth = library / "voice-changer-demo.pth"
            pth.write_text(str(library) + "\n", encoding="utf-8")
            script = (root / "DEBIAN" / "postinst").read_text(encoding="utf-8")
            script = script.replace(f"/opt/{deb.PACKAGE}", str(app))
            safe_path = Path(temporary) / "empty-bin"
            safe_path.mkdir()
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            env["PATH"] = str(safe_path)
            subprocess.run(
                ["/bin/sh", "-e"], input=script, env=env, text=True,
                capture_output=True, check=True, cwd=temporary,
            )
            self.assertEqual(pth.read_text(encoding="utf-8"), str(app / "src") + "\n")
            for argument, expected in (("--version", "1.0.0-demo.1"), ("--help", "usage:")):
                with self.subTest(argument=argument):
                    result = subprocess.run(
                        [str(python), "-I", "-B", "-m", "voice_changer", argument],
                        env=env, text=True, capture_output=True, check=True, cwd=temporary,
                    )
                    self.assertIn(expected, result.stdout)

    def test_postinst_registration_failure_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._stage(temporary)
            app = root / "opt" / deb.PACKAGE
            venv.EnvBuilder(with_pip=False).create(app / "venv")
            (app / "scripts" / "deploy_demo.py").write_text(
                "raise RuntimeError('synthetic source registration failure')\n",
                encoding="utf-8",
            )
            script = (root / "DEBIAN" / "postinst").read_text(encoding="utf-8")
            script = script.replace(f"/opt/{deb.PACKAGE}", str(app))
            result = subprocess.run(
                ["/bin/sh", "-e"], input=script, text=True, capture_output=True,
                cwd=temporary,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("synthetic source registration failure", result.stderr)

    def test_postinst_desktop_database_failure_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._stage(temporary)
            app = root / "opt" / deb.PACKAGE
            venv.EnvBuilder(with_pip=False).create(app / "venv")
            tools = Path(temporary) / "bin"
            tools.mkdir()
            refresh = tools / "update-desktop-database"
            refresh.write_text(
                "#!/bin/sh\n"
                "echo 'synthetic desktop database failure' >&2\nexit 7\n",
                encoding="utf-8",
            )
            refresh.chmod(0o755)
            script = (root / "DEBIAN" / "postinst").read_text(encoding="utf-8")
            script = script.replace(f"/opt/{deb.PACKAGE}", str(app))
            result = subprocess.run(
                ["/bin/sh", "-e"], input=script, text=True, capture_output=True,
                env={**os.environ, "PATH": str(tools)}, cwd=temporary,
            )
            self.assertEqual(result.returncode, 7)
            self.assertIn("synthetic desktop database failure", result.stderr)

    def test_desktop_entry_points_at_the_installed_launcher(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._stage(temporary)
            entry = (root / "usr" / "share" / "applications" / f"{deb.PACKAGE}.desktop").read_text(encoding="utf-8")
        self.assertIn(f'Exec="/opt/{deb.PACKAGE}/venv/bin/python"', entry)
        self.assertIn(f'/opt/{deb.PACKAGE}/scripts/launch_voice_changer.py', entry)
        self.assertIn("Terminal=true", entry)


@unittest.skipUnless(shutil.which("dpkg-deb"), "dpkg-deb is required")
class BuildTests(unittest.TestCase):
    def test_builds_a_readable_archive_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "demo.deb"
            deb.build(ROOT, output)
            self.assertTrue(output.is_file())
            listing = subprocess.run(
                ["dpkg-deb", "-c", str(output)], capture_output=True, text=True, check=True,
            ).stdout
            self.assertIn("opt/voice-changer-demo/src/voice_changer", listing)
            with self.assertRaises(FileExistsError):
                deb.build(ROOT, output)

    def test_dry_run_builds_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "demo.deb"
            result = deb.main(["--output", str(output), "--dry-run"])
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
