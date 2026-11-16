#!/usr/bin/env python3
"""Collect limited local environment evidence without importing the application."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from collections.abc import Sequence
from email.errors import MessageError
import errno
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import zipfile


CANDIDATE_DISTRIBUTIONS = (
    "sherpa-onnx", "onnxruntime", "onnxruntime-gpu", "torch", "torchaudio",
    "PySide6", "PySide6-Essentials", "shiboken6", "numpy",
)
MEMINFO_LIMIT = 65536
HASH_CHUNK_SIZE = 1 << 20
MEMINFO_FIELDS = {
    "MemTotal": "total_bytes",
    "MemAvailable": "available_bytes",
    "SwapTotal": "swap_total_bytes",
    "SwapFree": "swap_free_bytes",
}


class PreflightError(Exception):
    """The requested evidence could not be collected safely."""


def _os_error(label: str, error: OSError) -> PreflightError:
    code = errno.errorcode.get(error.errno, "IO_ERROR")
    return PreflightError(f"{label}: access or I/O error ({code})")


def _parts(path: str, label: str, *, leaf_required: bool) -> tuple[str, list[str]]:
    if not path or "\0" in path:
        raise PreflightError(f"{label}: empty or invalid path")
    if leaf_required and path.endswith("/"):
        raise PreflightError(f"{label}: a file path is required")
    parts = [part for part in path.split("/") if part and part != "."]
    if ".." in parts:
        raise PreflightError(f"{label}: '..' path components are not allowed")
    if leaf_required and not parts:
        raise PreflightError(f"{label}: a file path is required")
    return ("/" if path.startswith("/") else "."), parts


def _is_link(parent_fd: int, name: str) -> bool:
    try:
        return stat.S_ISLNK(os.stat(name, dir_fd=parent_fd, follow_symlinks=False).st_mode)
    except OSError:
        return False


def _directory(anchor: str, parts: list[str], label: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(anchor, flags)
    except OSError as error:
        raise _os_error(label, error) from error
    try:
        for part in parts:
            try:
                child = os.open(part, flags, dir_fd=fd)
            except OSError as error:
                if _is_link(fd, part):
                    raise PreflightError(f"{label}: symbolic links are not allowed") from error
                raise _os_error(label, error) from error
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _parent(path: str, label: str) -> tuple[int, str]:
    anchor, parts = _parts(path, label, leaf_required=True)
    return _directory(anchor, parts[:-1], label), parts[-1]


def read_memory() -> dict[str, int]:
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as stream:
            text = stream.read(MEMINFO_LIMIT + 1)
    except (OSError, UnicodeError) as error:
        if isinstance(error, OSError):
            raise _os_error("/proc/meminfo", error) from error
        raise PreflightError("/proc/meminfo: invalid ASCII data") from error
    if len(text) > MEMINFO_LIMIT:
        raise PreflightError("/proc/meminfo: data exceeds the size limit")
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, remainder = line.partition(":")
        if not separator or key not in MEMINFO_FIELDS:
            continue
        pieces = remainder.split()
        if (key in values or len(pieces) != 2 or pieces[1] != "kB"
                or not pieces[0].isascii() or not pieces[0].isdigit()
                or len(pieces[0]) > 20):
            raise PreflightError(f"/proc/meminfo: invalid or duplicate {key}")
        values[key] = int(pieces[0]) * 1024
    missing = MEMINFO_FIELDS.keys() - values.keys()
    if missing:
        raise PreflightError(f"/proc/meminfo: missing required fields ({', '.join(sorted(missing))})")
    if values["MemAvailable"] > values["MemTotal"] or values["SwapFree"] > values["SwapTotal"]:
        raise PreflightError("/proc/meminfo: inconsistent memory or swap values")
    return {output: values[key] for key, output in MEMINFO_FIELDS.items()}


def disk_evidence(path: str | None) -> dict:
    if path is None:
        return {"status": "not_requested"}
    anchor, parts = _parts(path, "disk directory", leaf_required=False)
    fd = _directory(anchor, parts, "disk directory")
    try:
        try:
            info = os.fstatvfs(fd)
        except OSError as error:
            raise _os_error("disk directory", error) from error
        return {
            "status": "collected",
            "total_bytes": info.f_blocks * info.f_frsize,
            "available_bytes": info.f_bavail * info.f_frsize,
        }
    finally:
        os.close(fd)


def dependency_evidence() -> list[dict]:
    result = []
    for name in CANDIDATE_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result.append({"name": name, "status": "missing", "version": None})
            continue
        except (OSError, UnicodeError, zipfile.BadZipFile, MessageError) as error:
            raise PreflightError(
                f"dependency metadata: cannot read {name} ({type(error).__name__})"
            ) from error
        if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+!-]{0,127}", version):
            raise PreflightError(f"dependency metadata: invalid version for {name}")
        result.append({"name": name, "status": "metadata_present", "version": version})
    return result


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def model_evidence(path: str, version: str | None, source: str | None,
                   license_name: str | None) -> dict:
    parent_fd, name = _parent(path, "model file")
    path_fd = None
    read_fd = None
    try:
        try:
            # O_PATH inspects the final inode before opening it for reading, so
            # a device or FIFO cannot be opened even if the pathname is swapped.
            path_fd = os.open(name, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
            before = os.fstat(path_fd)
            if stat.S_ISLNK(before.st_mode):
                raise PreflightError("model file: symbolic links are not allowed")
            if not stat.S_ISREG(before.st_mode):
                raise PreflightError("model file: a regular file is required")
            if not before.st_mode & 0o444:
                raise PreflightError("model file: no read permission bits")
            read_fd = os.open(f"/proc/self/fd/{path_fd}", os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            opened = os.fstat(read_fd)
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if _identity(opened) != _identity(before) or _identity(opened) != _identity(named):
                raise PreflightError("model file: changed while opening; evidence discarded")
            digest = hashlib.sha256()
            size = 0
            with os.fdopen(read_fd, "rb") as stream:
                read_fd = None
                while True:
                    block = stream.read(HASH_CHUNK_SIZE)
                    if not block:
                        break
                    size += len(block)
                    digest.update(block)
                after = os.fstat(stream.fileno())
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise _os_error("model file", error) from error
        if _identity(opened) != _identity(after) or _identity(after) != _identity(named) \
                or size != opened.st_size:
            raise PreflightError("model file: changed during hashing; evidence discarded")
        return {
            "status": "hashed",
            "size_bytes": size,
            "sha256": digest.hexdigest(),
            "declared": {"version": version, "source": source, "license": license_name},
            "license_verification": "not_verified",
        }
    finally:
        if read_fd is not None:
            os.close(read_fd)
        if path_fd is not None:
            os.close(path_fd)
        os.close(parent_fd)


def collect(args: argparse.Namespace) -> dict:
    if sys.platform != "linux" or not all(
        hasattr(os, flag) for flag in ("O_PATH", "O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    ):
        raise PreflightError("Linux with safe file-descriptor APIs is required")
    try:
        system = os.uname()
    except OSError as error:
        raise _os_error("system information", error) from error
    model = {"status": "not_requested"}
    if args.model_file is not None:
        model = model_evidence(args.model_file, args.model_version, args.model_source,
                               args.model_license)
    return {
        "schema_version": 1,
        "environment_evidence": {
            "architecture": system.machine,
            "os": system.sysname,
            "kernel_release": system.release,
            "python_version": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "memory_and_swap": read_memory(),
            "disk_space": disk_evidence(args.disk_path),
        },
        "dependency_metadata_scope": {
            "site_automatic_import": "disabled" if sys.flags.no_site else "enabled",
            "search": "visible_paths_and_distribution_finders",
            "missing_means": "metadata_not_found_in_searched_scope",
        },
        "candidate_dependencies": dependency_evidence(),
        "model_file": model,
        "not_established": {
            "dependency_import_and_compatibility": "not_tested",
            "model_license": "not_verified",
            "model_loading": "not_tested",
            "runtime_memory_peak": "not_measured",
            "inference_speed": "not_measured",
            "chinese_quality": "not_tested",
            "board_validation": "not_performed",
        },
    }


def _label(value: str, *, license_expression: bool = False) -> str:
    pattern = r"[A-Za-z0-9][A-Za-z0-9._+() -]{0,127}" if license_expression else \
        r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}"
    if not re.fullmatch(pattern, value) or value.endswith(" "):
        raise argparse.ArgumentTypeError(
            "use a short public identifier without credentials, URL query or private text"
        )
    return value


def _write_output(path: str | None, text: str) -> None:
    if path is None:
        try:
            if sys.stdout.write(text) != len(text):
                raise PreflightError("stdout: incomplete write")
            sys.stdout.flush()
        except OSError as error:
            raise _os_error("stdout", error) from error
        except (ValueError, UnicodeError) as error:
            raise PreflightError("stdout: cannot write JSON") from error
        return
    parent_fd, name = _parent(path, "output file")
    fd = None
    try:
        try:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW |
                         os.O_CLOEXEC, 0o600, dir_fd=parent_fd)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                fd = None
                if stream.write(text) != len(text):
                    raise PreflightError("output file: incomplete write; the file may be incomplete")
        except OSError as error:
            raise _os_error("output file (a new file is required; it may be incomplete)", error) from error
        except (ValueError, UnicodeError) as error:
            raise PreflightError("output file: cannot write JSON; the file may be incomplete") from error
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect offline environment evidence; do not load models or start audio.",
    )
    parser.add_argument("--disk-path", help="explicit existing local directory for free-space evidence")
    parser.add_argument("--model-file", help="explicit local regular file to stream-hash")
    parser.add_argument("--model-version", type=_label, help="declared public model version")
    parser.add_argument("--model-source", type=_label, help="declared public source identifier")
    parser.add_argument("--model-license", type=lambda value: _label(value, license_expression=True),
                        help="declared license identifier; not verified")
    parser.add_argument("--output", help="explicit new JSON file; default is stdout")
    args = parser.parse_args(argv)
    if args.model_file is None and any(
        item is not None for item in (args.model_version, args.model_source, args.model_license)
    ):
        parser.error("model declarations require --model-file")
    if args.output == "-":
        parser.error("omit --output to write to stdout")
    try:
        report = collect(args)
        _write_output(args.output, json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    except PreflightError as error:
        print(f"preflight: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("preflight: interrupted; no evidence was completed", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
