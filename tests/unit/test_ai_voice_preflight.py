"""Local synthetic tests for the standalone, offline AI preflight."""

import contextlib
from email.errors import MessageParseError
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import mock_open, patch
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ai_voice_preflight.py"
spec = importlib.util.spec_from_file_location("ai_voice_preflight", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
MEMINFO = "MemTotal: 1000 kB\nMemAvailable: 600 kB\nSwapTotal: 200 kB\nSwapFree: 50 kB\n"


class SyntheticFiles(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ai-preflight-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()

    def path(self, name):
        return self.directory / name

    def file(self, name, content=b"synthetic model"):
        path = self.path(name)
        with path.open("xb") as stream:
            stream.write(content)
        return path


class MemoryAndMetadataTests(unittest.TestCase):
    def test_synthetic_meminfo_is_parsed_as_bytes(self):
        with patch("builtins.open", mock_open(read_data=MEMINFO)):
            self.assertEqual(preflight.read_memory(), {
                "total_bytes": 1024000, "available_bytes": 614400,
                "swap_total_bytes": 204800, "swap_free_bytes": 51200,
            })

    def test_bad_missing_duplicate_and_oversized_meminfo_fail(self):
        invalid = (
            "", MEMINFO + "MemTotal: 1 kB\n",
            MEMINFO.replace("1000 kB", "-1 kB"),
            MEMINFO.replace("1000 kB", "1000 MB"),
            MEMINFO.replace("600 kB", "1001 kB"),
            MEMINFO.replace("50 kB", "201 kB"),
            "x" * (preflight.MEMINFO_LIMIT + 1),
        )
        for text in invalid:
            with self.subTest(text=text[:40]):
                with patch("builtins.open", mock_open(read_data=text)):
                    with self.assertRaises(preflight.PreflightError):
                        preflight.read_memory()

    def test_inaccessible_and_non_ascii_meminfo_fail(self):
        for error in (PermissionError(errno.EACCES, "denied"),
                      UnicodeDecodeError("ascii", b"\xff", 0, 1, "bad")):
            with self.subTest(error=type(error)):
                with patch("builtins.open", side_effect=error):
                    with self.assertRaises(preflight.PreflightError):
                        preflight.read_memory()

    def test_present_and_missing_distribution_metadata_are_distinct(self):
        def fake_version(name):
            if name == "torch":
                return "2.3.1"
            raise preflight.importlib.metadata.PackageNotFoundError(name)

        with patch.object(preflight.importlib.metadata, "version", side_effect=fake_version):
            evidence = preflight.dependency_evidence()
        self.assertEqual([item["name"] for item in evidence], list(preflight.CANDIDATE_DISTRIBUTIONS))
        torch = next(item for item in evidence if item["name"] == "torch")
        self.assertEqual(torch, {"name": "torch", "status": "metadata_present", "version": "2.3.1"})
        self.assertTrue(all(item["version"] is None for item in evidence if item["status"] == "missing"))

    def test_pyside_essentials_is_distinct_from_the_pyside_metapackage(self):
        versions = {"PySide6-Essentials": "6.8.3", "shiboken6": "6.8.3"}

        def fake_version(name):
            if name in versions:
                return versions[name]
            raise preflight.importlib.metadata.PackageNotFoundError(name)

        with patch.object(preflight.importlib.metadata, "version", side_effect=fake_version):
            evidence = {item["name"]: item for item in preflight.dependency_evidence()}

        self.assertEqual(evidence["PySide6"], {
            "name": "PySide6", "status": "missing", "version": None,
        })
        for name, version in versions.items():
            with self.subTest(name=name):
                self.assertEqual(evidence[name], {
                    "name": name, "status": "metadata_present", "version": version,
                })

    def test_corrupt_or_unreadable_metadata_is_not_treated_as_missing(self):
        for result in ("invalid version\n", None, PermissionError("synthetic")):
            with self.subTest(result=result):
                with patch.object(preflight.importlib.metadata, "version",
                                  side_effect=result if isinstance(result, Exception) else None,
                                  return_value=result):
                    with self.assertRaises(preflight.PreflightError):
                        preflight.dependency_evidence()

    def test_metadata_access_errors_are_clear_nonzero_failures(self):
        errors = (
            PermissionError(errno.EACCES, "denied", "/synthetic/private"),
            FileNotFoundError(errno.ENOENT, "disappeared"),
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"),
            zipfile.BadZipFile("invalid archive"),
            MessageParseError("invalid metadata"),
        )
        for error in errors:
            stdout, stderr = io.StringIO(), io.StringIO()
            with self.subTest(error=type(error).__name__), \
                 patch.object(preflight, "read_memory", return_value={}), \
                 patch.object(preflight.importlib.metadata, "version", side_effect=error), \
                 contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = preflight.main([])
            self.assertEqual(result, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("dependency metadata: cannot read", stderr.getvalue())
            self.assertIn(type(error).__name__, stderr.getvalue())
            self.assertNotIn("/synthetic/private", stderr.getvalue())

    def test_unexpected_programming_errors_are_not_relabelled(self):
        for error in (RuntimeError("bug"), AssertionError("bug"),
                      ValueError("bug"), TypeError("bug")):
            with self.subTest(error=type(error).__name__), \
                 patch.object(preflight, "read_memory", return_value={}), \
                 patch.object(preflight.importlib.metadata, "version", side_effect=error):
                with self.assertRaises(type(error)) as raised:
                    preflight.main([])
            self.assertIs(raised.exception, error)


class ModelAndPathTests(SyntheticFiles):
    def test_synthetic_files_use_a_private_temporary_directory(self):
        target = self.file("synthetic")
        self.assertEqual(target.parent, self.directory)
        self.assertNotEqual(target.parent, Path.cwd())
        self.assertEqual(target.parent.stat().st_mode & 0o777, 0o700)

    def test_streamed_hash_declarations_and_no_path_disclosure(self):
        content = b"synthetic bytes" * 500
        path = self.file("model", content)
        expected = hashlib.sha256(content).hexdigest()
        updates = []
        real_digest = hashlib.sha256()

        class ObservedDigest:
            def update(self, block):
                updates.append(len(block))
                real_digest.update(block)

            def hexdigest(self):
                return real_digest.hexdigest()

        with patch.object(preflight, "HASH_CHUNK_SIZE", 37), \
             patch.object(preflight.hashlib, "sha256", return_value=ObservedDigest()):
            result = preflight.model_evidence(str(path), "v1", "publisher/model", "Apache-2.0")
        self.assertEqual(result["sha256"], expected)
        self.assertEqual(result["size_bytes"], len(content))
        self.assertGreater(len(updates), 1)
        self.assertLessEqual(max(updates), 37)
        self.assertEqual(result["declared"], {
            "version": "v1", "source": "publisher/model", "license": "Apache-2.0",
        })
        self.assertEqual(result["license_verification"], "not_verified")
        self.assertNotIn(str(path), json.dumps(result))

    def test_missing_directory_symlink_and_unreadable_model_are_rejected(self):
        target = self.file("target")
        file_link = self.path("file-link")
        os.symlink(target, file_link)
        dir_link = self.path("dir-link")
        os.symlink(target.parent, dir_link)
        unreadable = self.file("unreadable")
        unreadable.chmod(0)
        fifo = self.path("fifo")
        os.mkfifo(fifo)
        cases = (
            (self.path("missing"), "access or I/O error"),
            (Path.cwd(), "regular file"),
            (file_link, "symbolic links"),
            (dir_link / target.name, "symbolic links"),
            (unreadable, "read permission"),
            (fifo, "regular file"),
        )
        for path, message in cases:
            with self.subTest(path=path.name):
                with self.assertRaisesRegex(preflight.PreflightError, message):
                    preflight.model_evidence(str(path), None, None, None)
        with self.assertRaisesRegex(preflight.PreflightError, "path components"):
            preflight.model_evidence("one/../two", None, None, None)

    def test_file_changes_during_hash_are_rejected(self):
        target = self.file("changing")
        original_fstat = os.fstat
        count = 0

        def changed(fd):
            nonlocal count
            info = original_fstat(fd)
            count += 1
            if count == 3:
                return types.SimpleNamespace(
                    st_dev=info.st_dev, st_ino=info.st_ino, st_size=info.st_size + 1,
                    st_mtime_ns=info.st_mtime_ns, st_ctime_ns=info.st_ctime_ns,
                )
            return info

        with patch.object(preflight.os, "fstat", side_effect=changed):
            with self.assertRaisesRegex(preflight.PreflightError, "changed during hashing"):
                preflight.model_evidence(str(target), None, None, None)

    def test_file_replacement_during_open_is_rejected(self):
        target = self.file("original", b"first")
        replacement = self.file("replacement", b"second")
        original_open = os.open
        swapped = False

        def swapping_open(path, flags, *args, **kwargs):
            nonlocal swapped
            if isinstance(path, str) and path.startswith("/proc/self/fd/"):
                os.replace(replacement, target)
                swapped = True
            return original_open(path, flags, *args, **kwargs)

        with patch.object(preflight.os, "open", side_effect=swapping_open):
            with self.assertRaisesRegex(preflight.PreflightError, "changed while opening"):
                preflight.model_evidence(str(target), None, None, None)
        self.assertTrue(swapped)

    def test_permission_denied_at_model_open_is_reported_without_path(self):
        target = self.file("denied")
        original_open = os.open

        def denying_open(path, flags, *args, **kwargs):
            if isinstance(path, str) and path.startswith("/proc/self/fd/"):
                raise PermissionError(errno.EACCES, "synthetic", str(target))
            return original_open(path, flags, *args, **kwargs)

        with patch.object(preflight.os, "open", side_effect=denying_open):
            with self.assertRaises(preflight.PreflightError) as raised:
                preflight.model_evidence(str(target), None, None, None)
        self.assertIn("EACCES", str(raised.exception))
        self.assertNotIn(str(target), str(raised.exception))

    def test_disk_requires_explicit_directory_without_symlinks(self):
        self.assertEqual(preflight.disk_evidence(None), {"status": "not_requested"})
        fake_stat = types.SimpleNamespace(f_blocks=100, f_bavail=20, f_frsize=4096)
        with patch.object(preflight.os, "fstatvfs", return_value=fake_stat):
            self.assertEqual(preflight.disk_evidence("."), {
                "status": "collected", "total_bytes": 409600, "available_bytes": 81920,
            })
        link = self.path("disk-link")
        os.symlink(Path.cwd(), link)
        with self.assertRaisesRegex(preflight.PreflightError, "symbolic links"):
            preflight.disk_evidence(str(link))
        with self.assertRaises(preflight.PreflightError):
            preflight.disk_evidence(str(self.file("disk-file")))


class CommandLineTests(SyntheticFiles):
    def run_main(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = preflight.main(args)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_json_shape_defaults_no_scans_and_explicit_missing_packages(self):
        def fake_version(name):
            if name == "numpy":
                return "2.2.4"
            raise preflight.importlib.metadata.PackageNotFoundError(name)

        with patch.object(preflight, "read_memory", return_value={"total_bytes": 1}), \
             patch.object(preflight, "model_evidence") as model, \
             patch.object(preflight.os, "scandir", side_effect=AssertionError("scan")), \
             patch.object(preflight.os, "listdir", side_effect=AssertionError("scan")), \
             patch.object(preflight.importlib.metadata, "version", side_effect=fake_version):
            code, out, err = self.run_main([])
        self.assertEqual((code, err), (0, ""))
        report = json.loads(out)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["environment_evidence"]["disk_space"]["status"], "not_requested")
        self.assertEqual(report["model_file"]["status"], "not_requested")
        self.assertEqual(report["not_established"]["model_license"], "not_verified")
        self.assertEqual(report["not_established"]["model_loading"], "not_tested")
        self.assertEqual(report["not_established"]["board_validation"], "not_performed")
        self.assertEqual(report["dependency_metadata_scope"], {
            "site_automatic_import": "disabled" if sys.flags.no_site else "enabled",
            "search": "visible_paths_and_distribution_finders",
            "missing_means": "metadata_not_found_in_searched_scope",
        })
        self.assertEqual({x["status"] for x in report["candidate_dependencies"]},
                         {"missing", "metadata_present"})
        self.assertTrue({"architecture", "os", "kernel_release", "python_version",
                         "logical_cpu_count", "memory_and_swap"} <= report["environment_evidence"].keys())
        model.assert_not_called()

    def test_explicit_new_output_is_private_json_and_never_overwrites(self):
        output = self.path("report.json")
        report = {"synthetic": True}
        with patch.object(preflight, "collect", return_value=report):
            code, out, err = self.run_main(["--output", str(output)])
            self.assertEqual((code, out, err), (0, "", ""))
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            code, out, err = self.run_main(["--output", str(output)])
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("output file", err)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)

    def test_output_symlinks_and_failed_stdout_are_errors(self):
        target = self.file("existing", b"do not replace")
        link = self.path("output-link")
        os.symlink(target, link)
        with patch.object(preflight, "collect", return_value={"synthetic": True}):
            code, out, err = self.run_main(["--output", str(link)])
        self.assertEqual((code, out), (1, ""))
        self.assertIn("output file", err)
        self.assertEqual(target.read_bytes(), b"do not replace")

        class BrokenOutput(io.StringIO):
            def write(self, text):
                raise OSError(errno.ENOSPC, "synthetic")

        stderr = io.StringIO()
        with patch.object(preflight, "collect", return_value={}), \
             patch.object(sys, "stdout", BrokenOutput()), contextlib.redirect_stderr(stderr):
            self.assertEqual(preflight.main([]), 1)
        self.assertIn("stdout", stderr.getvalue())

        class ShortOutput(io.StringIO):
            def write(self, text):
                return len(text) - 1

        stderr = io.StringIO()
        with patch.object(preflight, "collect", return_value={}), \
             patch.object(sys, "stdout", ShortOutput()), contextlib.redirect_stderr(stderr):
            self.assertEqual(preflight.main([]), 1)
        self.assertIn("incomplete write", stderr.getvalue())

    def test_file_write_error_is_nonzero_and_identifies_possible_partial_output(self):
        output = self.path("partial-report")

        class FailedFile:
            def __init__(self, fd):
                self.fd = fd

            def __enter__(self):
                return self

            def __exit__(self, *args):
                os.close(self.fd)

            def write(self, text):
                raise OSError(errno.ENOSPC, "synthetic")

        with patch.object(preflight, "collect", return_value={}), \
             patch.object(preflight.os, "fdopen", side_effect=lambda fd, *a, **kw: FailedFile(fd)):
            code, out, err = self.run_main(["--output", str(output)])
        self.assertEqual((code, out), (1, ""))
        self.assertIn("may be incomplete", err)
        self.assertTrue(output.exists())

    def test_invalid_inputs_and_access_errors_return_nonzero_without_json(self):
        code, out, err = self.run_main(["--model-file", str(self.path("missing"))])
        self.assertEqual((code, out), (1, ""))
        self.assertIn("model file", err)
        for arguments in (
            ["--model-version", "v1"],
            ["--model-file", "model", "--model-source", "https://host/name?token=secret"],
            ["--output", "-"],
        ):
            stderr = io.StringIO()
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    preflight.main(arguments)
            self.assertEqual(raised.exception.code, 2)
            self.assertNotIn("secret", stderr.getvalue())
        with patch.object(preflight, "read_memory", side_effect=preflight.PreflightError("bad proc")):
            code, out, err = self.run_main([])
        self.assertEqual((code, out), (1, ""))
        self.assertIn("bad proc", err)

    def test_isolated_cli_never_imports_candidates_or_opens_network(self):
        model = self.file("isolated", b"synthetic")
        wrapper = r'''
import builtins, io, runpy, sys
blocked = {"torch", "torchaudio", "onnxruntime", "sherpa_onnx", "PySide6", "shiboken6",
           "numpy", "voice_changer", "sounddevice", "soundfile"}
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise AssertionError("unexpected import: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
def audit(event, args):
    if event.startswith("socket.") or event in {"subprocess.Popen", "os.system"}:
        raise AssertionError("unexpected network or subprocess: " + event)
sys.addaudithook(audit)
module = runpy.run_path(sys.argv[1], run_name="preflight_test")
original_open = builtins.open
def fake_open(path, *args, **kwargs):
    if path == "/proc/meminfo":
        return io.StringIO("MemTotal: 100 kB\nMemAvailable: 50 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n")
    return original_open(path, *args, **kwargs)
builtins.open = fake_open
metadata = module["dependency_evidence"].__globals__["importlib"].metadata
def fake_version(name):
    if name == "torch":
        return "2.3.1"
    raise metadata.PackageNotFoundError(name)
metadata.version = fake_version
result = module["main"](["--disk-path", ".", "--model-file", sys.argv[2]])
assert not (blocked & set(sys.modules))
raise SystemExit(result)
'''
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", wrapper, str(SCRIPT), str(model)],
            text=True, capture_output=True, timeout=10, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["model_file"]["sha256"], hashlib.sha256(b"synthetic").hexdigest())
        self.assertEqual(report["environment_evidence"]["memory_and_swap"]["total_bytes"], 102400)
        self.assertEqual(report["model_file"]["license_verification"], "not_verified")

    def test_documented_safe_example_in_a_separate_process(self):
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "scripts/ai_voice_preflight.py",
             "--disk-path", "."],
            cwd=ROOT, text=True, capture_output=True, timeout=10, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["dependency_metadata_scope"], {
            "site_automatic_import": "disabled",
            "search": "visible_paths_and_distribution_finders",
            "missing_means": "metadata_not_found_in_searched_scope",
        })
        self.assertEqual(report["environment_evidence"]["disk_space"]["status"], "collected")
        self.assertEqual(report["model_file"]["status"], "not_requested")
        self.assertEqual({item["name"] for item in report["candidate_dependencies"]},
                         set(preflight.CANDIDATE_DISTRIBUTIONS))
        self.assertEqual(report["not_established"]["board_validation"], "not_performed")


if __name__ == "__main__":
    unittest.main()
