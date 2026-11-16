"""Deployment helper: copy, venv, archive and launcher generation; no audio."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("deploy_demo", ROOT / "scripts/deploy_demo.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class CopyTests(unittest.TestCase):
    def test_copy_skips_caches_venv_and_personal_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            (source / "src" / "pkg").mkdir(parents=True)
            (source / "src" / "pkg" / "app.py").write_text("print('hi')\n")
            (source / "src" / "pkg" / "__pycache__").mkdir()
            (source / "src" / "pkg" / "__pycache__" / "x.pyc").write_bytes(b"\0")
            (source / "saved").mkdir()
            (source / "saved" / "take.wav").write_bytes(b"audio")
            (source / "README.md").write_text("readme\n")
            (source / "requirements-ai.txt").write_text("onnxruntime==1.16.3\n")

            written = deploy.copy_tree(source, source.parent / "out")
            self.assertEqual(sorted(written), ["README.md", "requirements-ai.txt", "src/pkg/app.py"])
            self.assertTrue((source.parent / "out" / "src" / "pkg" / "app.py").is_file())
            self.assertFalse((source.parent / "out" / "saved").exists())

    def test_copy_refuses_to_overwrite_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            destination = Path(temporary) / "out"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                deploy.copy_tree(source, destination)

    def test_archive_contains_no_personal_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            (source / "src").mkdir(parents=True)
            (source / "src" / "app.py").write_text("x\n")
            (source / "saved").mkdir()
            (source / "saved" / "voice.wav").write_bytes(b"audio")
            archive = Path(temporary) / "bundle.tar.gz"
            deploy.build_archive(source, archive)
            with tarfile.open(archive) as bundle:
                names = bundle.getnames()
            joined = " ".join(names)
            self.assertIn("app.py", joined)
            self.assertNotIn("voice.wav", joined)
            with self.assertRaises(FileExistsError):
                deploy.build_archive(source, archive)


class SourceRegistrationTests(unittest.TestCase):
    def test_pth_file_points_at_the_deployed_source_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            venv = Path(temporary) / "venv"
            (venv / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
            app_root = Path(temporary) / "app"
            app_root.mkdir()
            deploy.register_source(venv, app_root)
            pth = venv / "lib" / "python3.13" / "site-packages" / "voice-changer-demo.pth"
            self.assertEqual(pth.read_text(encoding="utf-8").strip(), str(app_root / "src"))

    def test_missing_library_directory_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                deploy.register_source(Path(temporary) / "venv", Path(temporary))


class CommandLineTests(unittest.TestCase):
    def test_dry_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "deploy"
            result = deploy.main([
                "--destination", str(destination), "--python", sys.executable, "--dry-run",
            ])
            self.assertEqual(result, 0)
            self.assertFalse(destination.exists())

    def test_existing_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "deploy"
            destination.mkdir()
            result = deploy.main(["--destination", str(destination), "--python", sys.executable])
            self.assertEqual(result, 2)

    def test_missing_interpreter_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = deploy.main([
                "--destination", str(Path(temporary) / "d"),
                "--python", str(Path(temporary) / "nope"),
            ])
            self.assertEqual(result, 2)

    def test_copy_only_creates_the_tree_without_a_venv(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "deploy"
            result = deploy.main([
                "--destination", str(destination), "--python", sys.executable, "--no-venv",
            ])
            self.assertEqual(result, 0)
            self.assertTrue((destination / "src" / "voice_changer" / "app" / "main.py").is_file())
            self.assertFalse((destination / "venv").exists())

    @unittest.skipUnless(hasattr(os, "geteuid") and os.geteuid() != 0, "needs a normal user")
    def test_desktop_entry_is_written_but_not_installed(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "deploy"
            entry = Path(temporary) / "voice-changer.desktop"
            result = deploy.main([
                "--destination", str(destination), "--python", sys.executable,
                "--no-venv", "--desktop-entry", str(entry),
            ])
            self.assertEqual(result, 0)
            text = entry.read_text()
            self.assertIn("Exec=", text)
            self.assertIn(str(destination / "scripts" / "launch_voice_changer.py"), text)
            self.assertIn("Terminal=true", text)


if __name__ == "__main__":
    unittest.main()
