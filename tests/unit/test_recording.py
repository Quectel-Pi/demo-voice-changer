"""Synthetic-only D2 storage tests; all temporary data stays in the project."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import wave

import numpy as np

from voice_changer.audio import recording
from voice_changer.audio.recording import (
    RecordingError, RecordingGroup, RecordingRecoveryError, RecordingSaveError,
    RecordingWriter, discard_recoverable, recoverable_groups,
)
from voice_changer.dsp.parameters import EffectParameters


ORIGINAL = EffectParameters()
CHANGED = EffectParameters(3, 1.1, False)


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".recording-test-", dir=Path(__file__).resolve().parents[2],
        )
        self.root = Path(self.temporary.name)
        self.writers = []

    def tearDown(self):
        for writer in self.writers:
            if writer._thread is not None and writer._thread.is_alive():
                writer.finish(complete=False, reason="test cleanup")
            elif not writer.group._finalized:
                writer.finish(complete=False, reason="test cleanup")
        self.temporary.cleanup()

    def writer(self, channels=1, blocksize=8, **kwargs):
        writer = RecordingWriter(48000, channels, blocksize, data_root=self.root, **kwargs)
        self.writers.append(writer)
        return writer

    def block(self, n=8, channels=1):
        return np.full((n, channels), 0.25, dtype=np.float32)

    def recorded(self, *, complete=True, reason=None):
        writer = self.writer(alignment={"latency_frames": 512, "tail": "caller-gated"})
        writer.start()
        data = self.block()
        self.assertTrue(writer.submit(data, data, source_frame=100,
                                      parameters=ORIGINAL, revision=0))
        return writer, writer.finish(complete=complete, reason=reason)

    def metadata(self, group):
        return json.loads(group.metadata_path.read_text(encoding="utf-8"))

    def rewrite_metadata(self, group, change):
        metadata = self.metadata(group)
        change(metadata)
        group.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def wait_for(self, predicate):
        deadline = time.monotonic() + 2
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.002)
        self.assertTrue(predicate(), "disk writer did not reach expected state")

    def test_import_has_no_numpy_backend_or_file_io(self):
        program = """
import builtins, sys
original = builtins.open
def deny(*args, **kwargs):
    raise AssertionError('unexpected module-import file I/O')
builtins.open = deny
import voice_changer.audio.recording
assert 'numpy' not in sys.modules
assert 'sounddevice' not in sys.modules
assert 'PySide6' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", program],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pcm16_stereo_source_interval_and_streamed_parameters(self):
        writer = self.writer(channels=2, blocksize=4, alignment={"latency_frames": 7})
        writer.start()
        raw = np.array([[-2, 2], [-1, 1], [-0.5, 0.5], [0, 0.25]], dtype=np.float32)
        effect = raw * 0.5
        self.assertTrue(writer.submit(raw, effect, source_frame=900,
                                      parameters=ORIGINAL, revision=0))
        # Mutating the producer buffer must not change accepted PCM.
        raw.fill(0)
        self.assertTrue(writer.submit(raw[:2], effect[:2], source_frame=904,
                                      parameters=CHANGED, revision=5))
        self.assertTrue(writer.submit(raw[:1], effect[:1], source_frame=906,
                                      parameters=CHANGED, revision=5))
        group = writer.finish(complete=True)
        writer.check_health()
        self.assertEqual(group.frames, 7)
        self.assertTrue(group.complete)
        self.assertFalse(group.saved)
        with wave.open(str(group.original_path), "rb") as wav:
            self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()),
                             (48000, 2, 2))
            samples = np.frombuffer(wav.readframes(7), dtype="<i2").reshape(-1, 2)
        np.testing.assert_array_equal(samples[:4], [
            [-32768, 32767], [-32768, 32767], [-16384, 16384], [0, 8192],
        ])
        np.testing.assert_array_equal(samples[4:], 0)
        with wave.open(str(group.effect_path), "rb") as wav:
            samples = np.frombuffer(wav.readframes(7), dtype="<i2").reshape(-1, 2)
        np.testing.assert_array_equal(samples[0], [-32768, 32767])
        metadata = self.metadata(group)
        self.assertEqual(metadata["format"], "PCM16_LE")
        self.assertEqual(metadata["source_interval"], {"start": 900, "end_exclusive": 907})
        self.assertEqual(metadata["alignment"], {"latency_frames": 7})
        self.assertEqual(metadata["tracks"]["original"]["samples_clipped_outside_unit_range"], 2)
        self.assertEqual(metadata["tracks"]["effect"]["samples_clipped_outside_unit_range"], 0)
        self.assertEqual(metadata["parameter_changes"]["count"], 2)
        journal = [json.loads(line) for line in
                   (group.directory / "parameters.jsonl").read_text().splitlines()]
        self.assertEqual([row["source_frame"] for row in journal], [900, 904])
        self.assertEqual([row["revision"] for row in journal], [0, 5])
        self.assertEqual(journal[1]["parameters"]["pitch_semitones"], 3)
        self.assertIsNone(metadata["reason"])
        self.assertEqual(RecordingGroup.recover(group.directory).frames, 7)

    def test_queue_is_preallocated_bounded_and_drains_on_overflow(self):
        writer = self.writer()
        entered, release = threading.Event(), threading.Event()
        write = writer._write_block

        def delayed(slot):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test release timed out")
            write(slot)

        with patch.object(writer, "_write_block", delayed):
            writer.start()
            data = self.block()
            self.assertTrue(writer.submit(data, data, source_frame=0,
                                          parameters=ORIGINAL, revision=0))
            self.assertTrue(entered.wait(2))
            try:
                buffers = id(writer._raw), id(writer._effect)
                for index in range(1, 64):
                    self.assertTrue(writer.submit(data, data, source_frame=index * 8,
                                                  parameters=ORIGINAL, revision=0))
                self.assertEqual(writer._write - writer._read, 64)
                self.assertFalse(writer.submit(data, data, source_frame=512,
                                               parameters=ORIGINAL, revision=0))
                self.assertEqual((id(writer._raw), id(writer._effect)), buffers)
                with self.assertRaisesRegex(RecordingError, "overflow"):
                    writer.check_health()
            finally:
                release.set()
            group = writer.finish(complete=True)
        self.assertEqual(group.frames, 512)
        self.assertFalse(group.complete)
        self.assertIn("overflow", self.metadata(group)["reason"])
        self.assertEqual(RecordingGroup.recover(group.directory).frames, 512)

    def test_ring_wrap_and_many_changes_are_streamed_with_bounded_storage(self):
        writer = self.writer(blocksize=1)
        writer.start()
        for index in range(193):
            data = np.array([[index / 256]], dtype=np.float32)
            self.assertTrue(writer.submit(data, -data, source_frame=1000 + index,
                                          parameters=CHANGED, revision=index))
            if index % 32 == 31:
                self.wait_for(lambda: writer._read == writer._write)
        group = writer.finish(complete=True)
        self.assertEqual(writer._raw.shape, (64, 1, 1))
        self.assertEqual(len(writer._parameters), 64)
        self.assertTrue(all(value is None for value in writer._parameters))
        self.assertEqual(self.metadata(group)["parameter_changes"]["count"], 193)
        with (group.directory / "parameters.jsonl").open() as stream:
            for index, line in enumerate(stream):
                self.assertEqual(json.loads(line)["revision"], index)
        with wave.open(str(group.effect_path), "rb") as wav:
            actual = np.frombuffer(wav.readframes(193), dtype="<i2")
        np.testing.assert_array_equal(actual, -np.arange(193) * 128)
        self.assertEqual(RecordingGroup.recover(group.directory).frames, 193)

    def test_submit_performs_no_io_lock_or_json_serialization(self):
        writer = self.writer()
        writer.start()
        # The producer itself never calls any of these; the worker is paused.
        entered, release = threading.Event(), threading.Event()
        write = writer._write_block

        def delayed(slot):
            entered.set()
            release.wait(2)
            write(slot)

        data = self.block()
        with patch.object(writer, "_write_block", delayed):
            self.assertTrue(writer.submit(data, data, source_frame=0,
                                          parameters=ORIGINAL, revision=0))
            self.assertTrue(entered.wait(2))
            try:
                with patch.object(recording.json, "dump", side_effect=AssertionError("JSON")), \
                     patch.object(recording.shutil, "disk_usage", side_effect=AssertionError("disk")), \
                     patch("builtins.open", side_effect=AssertionError("open")), \
                     patch.object(recording.time, "sleep", side_effect=AssertionError("wait")):
                    self.assertTrue(writer.submit(data, data, source_frame=8,
                                                  parameters=CHANGED, revision=1))
            finally:
                release.set()
            writer.finish(complete=True)
        writer.check_health()

    def test_invalid_blocks_fail_explicitly_without_accepting(self):
        for invalid in ("shape", "dtype", "nan", "inf", "large", "empty",
                        "parameters", "revision", "source", "mismatch"):
            with self.subTest(invalid=invalid):
                writer = self.writer()
                writer.start()
                raw, effect = self.block(), self.block()
                kwargs = dict(source_frame=0, parameters=ORIGINAL, revision=0)
                if invalid == "shape":
                    raw = raw[:, 0]
                elif invalid == "dtype":
                    raw = raw.astype(np.float64)
                elif invalid in ("nan", "inf"):
                    effect[0] = float(invalid)
                elif invalid == "large":
                    raw = effect = self.block(9)
                elif invalid == "empty":
                    raw = effect = self.block(0)
                elif invalid == "parameters":
                    kwargs["parameters"] = {}
                elif invalid == "revision":
                    kwargs["revision"] = -1
                elif invalid == "source":
                    kwargs["source_frame"] = -1
                elif invalid == "mismatch":
                    effect = self.block(7)
                self.assertFalse(writer.submit(raw, effect, **kwargs))
                group = writer.finish(complete=True)
                self.assertFalse(group.complete)
                self.assertEqual(writer.frames_written, 0)
                self.assertIsNotNone(writer.error)

    def test_source_discontinuity_marks_partial_and_preserves_accepted_blocks(self):
        writer = self.writer()
        writer.start()
        data = self.block()
        self.assertTrue(writer.submit(data, data, source_frame=20,
                                      parameters=ORIGINAL, revision=0))
        self.assertFalse(writer.submit(data, data, source_frame=29,
                                       parameters=ORIGINAL, revision=0))
        group = writer.finish(complete=True)
        self.assertFalse(group.complete)
        self.assertEqual(group.frames, 8)
        self.assertEqual(RecordingGroup.recover(group.directory).frames, 8)

    def test_disk_reserve_checked_before_start(self):
        usage = shutil._ntuple_diskusage(10**9, 0, recording.RESERVE_BYTES)
        with patch.object(recording.shutil, "disk_usage", return_value=usage):
            writer = self.writer()
        self.assertIsInstance(writer.group, RecordingGroup)
        self.assertIsInstance(writer.error, OSError)
        self.assertFalse(writer.group.complete)
        with self.assertRaisesRegex(OSError, "64 MiB"):
            writer.start()
        self.assertTrue(writer.group.metadata_path.exists())

    def test_idle_writer_checks_space_at_least_once_per_second(self):
        writer = self.writer()
        writer._last_check = time.monotonic() - 1.1
        usage = shutil._ntuple_diskusage(10**9, 0, recording.RESERVE_BYTES)
        with patch.object(recording.shutil, "disk_usage", return_value=usage) as disk:
            writer.start()
            self.wait_for(lambda: writer.error is not None)
            writer.finish(complete=True)
        self.assertGreaterEqual(disk.call_count, 1)
        self.assertEqual(writer.free_bytes, recording.RESERVE_BYTES)
        self.assertIn("64 MiB", self.metadata(writer.group)["reason"])

    def test_space_budget_rechecks_even_before_periodic_interval(self):
        writer = self.writer()
        writer._budget = recording.RESERVE_BYTES
        usage = shutil._ntuple_diskusage(10**9, 0, recording.RESERVE_BYTES)
        data = self.block()
        with patch.object(recording.shutil, "disk_usage", return_value=usage) as disk:
            writer.start()
            self.assertTrue(writer.submit(data, data, source_frame=0,
                                          parameters=ORIGINAL, revision=0))
            group = writer.finish(complete=True)
        self.assertGreaterEqual(disk.call_count, 1)
        self.assertFalse(group.complete)
        self.assertEqual(group.frames, 0)

    def test_riff_limit_is_enforced_without_a_duration_cap(self):
        writer = self.writer()
        data = self.block()
        with patch.object(recording, "RIFF_DATA_LIMIT", 15):
            writer.start()
            self.assertTrue(writer.submit(data, data, source_frame=0,
                                          parameters=ORIGINAL, revision=0))
            group = writer.finish(complete=True)
        self.assertFalse(group.complete)
        self.assertIn("RIFF", self.metadata(group)["reason"])
        self.assertEqual(group.frames, 0)

    def test_write_error_retains_misaligned_tracks_and_reports_recovery_failure(self):
        writer = self.writer()
        writer.start()
        data = self.block()
        failure = OSError("synthetic effect write failure")
        with patch.object(writer._waves[1], "writeframes", side_effect=failure):
            self.assertTrue(writer.submit(data, data, source_frame=100,
                                          parameters=ORIGINAL, revision=0))
            group = writer.finish(complete=True)
        self.assertIs(writer.error, failure)
        with self.assertRaisesRegex(OSError, "synthetic effect"):
            writer.check_health()
        self.assertFalse(group.complete)
        self.assertTrue(group.original_path.exists())
        self.assertTrue(group.effect_path.exists())
        self.assertIn("synthetic effect", self.metadata(group)["reason"])
        with self.assertRaisesRegex(RecordingRecoveryError, "alignment"):
            RecordingGroup.recover(group.directory)
        self.assertTrue(group.directory.exists())

    def test_close_error_is_explicit_and_never_marks_group_complete(self):
        writer = self.writer()
        close = writer._waves[1].close

        def failed_close():
            close()
            raise OSError("synthetic close failure")

        writer.start()
        data = self.block()
        with patch.object(writer._waves[1], "close", side_effect=failed_close):
            self.assertTrue(writer.submit(data, data, source_frame=0,
                                          parameters=ORIGINAL, revision=0))
            group = writer.finish(complete=True)
        self.assertFalse(group.complete)
        self.assertIn("close failure", self.metadata(group)["reason"])
        self.assertEqual(RecordingGroup.recover(group.directory).frames, 8)

    def test_metadata_failure_keeps_pcm_and_exposes_real_error(self):
        writer = self.writer()
        writer.start()
        data = self.block()
        failure = PermissionError("synthetic metadata failure")
        with patch.object(writer, "_write_metadata", side_effect=failure):
            self.assertTrue(writer.submit(data, data, source_frame=0,
                                          parameters=ORIGINAL, revision=0))
            group = writer.finish(complete=True)
        self.assertIs(writer.error, failure)
        self.assertFalse(group.complete)
        with self.assertRaises(PermissionError):
            writer.check_health()
        with wave.open(str(group.original_path), "rb") as wav:
            self.assertEqual(wav.getnframes(), 8)
        with self.assertRaises(RecordingRecoveryError):
            RecordingGroup.recover(group.directory)
        self.assertTrue(group.effect_path.exists())

    def test_unexpected_worker_exception_cannot_leave_success_shaped_group(self):
        writer = self.writer()
        writer.start()
        data = self.block()
        failure = IndexError("synthetic unexpected writer failure")
        with patch.object(writer, "_write_block", side_effect=failure), \
             patch.object(threading, "excepthook") as hook:
            self.assertTrue(writer.submit(data, data, source_frame=0,
                                          parameters=ORIGINAL, revision=0))
            group = writer.finish(complete=True)
        self.assertIs(writer.error, failure)
        self.assertFalse(group.complete)
        self.assertEqual(hook.call_count, 1)
        self.assertIn("unexpected writer failure", self.metadata(group)["reason"])

    def test_requested_incomplete_reason_is_persisted_and_playable(self):
        writer, group = self.recorded(complete=False, reason="input unplugged")
        self.assertFalse(group.complete)
        self.assertEqual(self.metadata(group)["reason"], "input unplugged")
        recovered = RecordingGroup.recover(group.directory)
        self.assertEqual(recovered.frames, 8)
        self.assertFalse(recovered.complete)
        writer.check_health()

    def test_finish_timeout_preserves_thread_and_handle_until_retry(self):
        writer = self.writer()
        entered, release = threading.Event(), threading.Event()
        write = writer._write_block

        def delayed(slot):
            entered.set()
            release.wait(2)
            write(slot)

        with patch.object(writer, "_write_block", delayed):
            writer.start()
            data = self.block()
            self.assertTrue(writer.submit(data, data, source_frame=0,
                                          parameters=ORIGINAL, revision=0))
            self.assertTrue(entered.wait(2))
            try:
                thread = writer._thread
                with patch.object(recording, "JOIN_TIMEOUT_SECONDS", 0.01):
                    with self.assertRaisesRegex(TimeoutError, "handle/data retained"):
                        writer.finish(complete=True)
                self.assertIs(writer._thread, thread)
                self.assertTrue(thread.is_alive())
                self.assertIsNotNone(writer._owner_fd)
                self.assertNotIn(writer.group.directory, recoverable_groups(self.root))
                with self.assertRaisesRegex(RecordingRecoveryError, "locked"):
                    discard_recoverable(writer.group.directory)
                self.assertFalse(writer.group._finalized)
                with self.assertRaises(RecordingSaveError):
                    writer.group.save(self.root, "not-finalized")
            finally:
                release.set()
            group = writer.finish(complete=False, reason="timeout")
        self.assertFalse(group.complete)
        self.assertEqual(group.frames, 8)
        self.assertIsNone(writer._owner_fd)
        self.assertIn("did not finish", self.metadata(group)["reason"])

    def test_finish_before_start_is_incomplete_recoverable_and_idempotent(self):
        writer = self.writer()
        group = writer.finish(complete=False, reason="user cancelled")
        before = group.metadata_path.read_bytes()
        self.assertEqual(group.frames, 0)
        self.assertFalse(group.complete)
        self.assertEqual(self.metadata(group)["reason"], "user cancelled")
        self.assertEqual(RecordingGroup.recover(group.directory).frames, 0)
        self.assertIs(writer.finish(complete=False, reason="second close"), group)
        self.assertEqual(group.metadata_path.read_bytes(), before)
        with self.assertRaisesRegex(RecordingError, "after finish"):
            writer.start()
        group.discard()
        self.assertIs(writer.finish(complete=False), group)
        self.assertFalse(group.directory.exists())

    def test_finish_after_thread_start_failure_is_safe_and_preserves_error(self):
        writer = self.writer()
        failure = RuntimeError("synthetic start failure")
        with patch.object(threading.Thread, "start", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "start failure"):
                writer.start()
        before = writer.group.metadata_path.read_bytes()
        self.assertIsNone(writer._thread.ident)
        self.assertIsNone(writer._owner_fd)
        self.assertIn(writer.group.directory, recoverable_groups(self.root))
        for _ in range(2):
            group = writer.finish(complete=False, reason="controller close")
            self.assertFalse(group.complete)
            self.assertIs(writer.error, failure)
        self.assertEqual(group.metadata_path.read_bytes(), before)
        self.assertEqual(RecordingGroup.recover(group.directory).frames, 0)

    def test_repeated_finish_after_dsp_failure_preserves_terminal_reason(self):
        writer = self.writer()
        writer.start()
        data = self.block()
        self.assertTrue(writer.submit(data, data, source_frame=0,
                                      parameters=ORIGINAL, revision=0))
        self.assertFalse(writer.submit(data, data, source_frame=9,
                                       parameters=ORIGINAL, revision=0))
        self.wait_for(lambda: writer.group._finalized and not writer._thread.is_alive())
        before = writer.group.metadata_path.read_bytes()
        for complete in (False, False, True):
            group = writer.finish(complete=complete, reason="controller close")
            self.assertFalse(group.complete)
            self.assertEqual(group.frames, 8)
            self.assertEqual(group.metadata_path.read_bytes(), before)
        self.assertEqual(RecordingGroup.recover(group.directory).frames, 8)

    def test_active_frames_update_per_block_but_complete_only_after_finish(self):
        writer = self.writer()
        writer.start()
        data = self.block()
        for index in range(2):
            self.assertTrue(writer.submit(data, data, source_frame=index * 8,
                                          parameters=ORIGINAL, revision=0))
            self.wait_for(lambda: writer._read == writer._write)
            self.assertEqual(writer.frames_written, (index + 1) * 8)
            self.assertEqual(writer.group.frames, (index + 1) * 8)
            self.assertFalse(writer.group.complete)
            self.assertFalse(writer.group._finalized)
        group = writer.finish(complete=True)
        self.assertTrue(group.complete)
        before = group.metadata_path.read_bytes()
        self.assertIs(writer.finish(complete=False, reason="duplicate cleanup"), group)
        self.assertEqual(group.metadata_path.read_bytes(), before)
        saved = group.save(self.root, "terminal")
        self.assertIs(writer.finish(complete=False), group)
        self.assertTrue(saved.original_path.exists())

    def test_default_scanner_is_lazy_without_numpy_or_directory_creation(self):
        missing = self.root / "xdg-missing"
        program = """
import sys
from voice_changer.audio.recording import recoverable_groups
assert recoverable_groups() == ()
assert 'numpy' not in sys.modules
assert 'sounddevice' not in sys.modules
assert 'PySide6' not in sys.modules
"""
        environment = dict(os.environ, XDG_DATA_HOME=str(missing),
                           PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", program], env=environment,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(missing.exists())

    def test_idle_import_scan_and_explicit_delete_never_import_dsp_or_numpy(self):
        _, group = self.recorded(complete=False, reason="corrupt leftover")
        group.metadata_path.write_bytes(b"{broken")
        program = """
import importlib.abc
import sys
from pathlib import Path

class BlockAudioImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == 'numpy' or fullname.startswith('numpy.')
                or fullname == 'voice_changer.dsp'
                or fullname.startswith('voice_changer.dsp.')):
            raise AssertionError('Forbidden idle import: ' + fullname)

sys.meta_path.insert(0, BlockAudioImports())
from voice_changer.audio.recording import recoverable_groups, discard_recoverable
assert not any(name == 'voice_changer.dsp' or name.startswith('voice_changer.dsp.')
               or name == 'numpy' or name.startswith('numpy.') for name in sys.modules)
root, directory = map(Path, sys.argv[1:])
assert recoverable_groups(root) == (directory,)
discard_recoverable(directory)
assert recoverable_groups(root) == ()
assert not directory.exists()
assert not any(name == 'voice_changer.dsp' or name.startswith('voice_changer.dsp.')
               or name == 'numpy' or name.startswith('numpy.') for name in sys.modules)
"""
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [sys.executable, "-c", program, str(self.root), str(group.directory)],
            env=environment, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(group.directory.exists())

    def test_save_publishes_verified_full_group_and_removes_only_temporary_copy(self):
        writer, group = self.recorded()
        original_dir = group.directory
        original_pcm = group.original_path.read_bytes()
        saved = group.save(self.root, "my recording")
        self.assertTrue(saved.saved)
        self.assertTrue(saved.complete)
        self.assertEqual(saved.frames, 8)
        self.assertEqual(saved.directory, self.root / "my recording")
        self.assertFalse(original_dir.exists())
        self.assertEqual(saved.original_path.read_bytes(), original_pcm)
        self.assertTrue(self.metadata(saved)["saved"])
        self.assertEqual(set(p.name for p in saved.directory.iterdir()),
                         {"original.wav", "effect.wav", "parameters.jsonl", "metadata.json"})
        with self.assertRaisesRegex(RecordingError, "saved"):
            saved.discard()
        self.assertTrue(saved.directory.exists())
        with wave.open(str(saved.effect_path), "rb") as wav:
            self.assertEqual(wav.getnframes(), 8)

    def test_active_group_cannot_be_saved_or_discarded(self):
            writer = self.writer()
            self.assertEqual(writer.group.directory.stat().st_mode & 0o777, 0o700)
            with self.assertRaises(RecordingSaveError):
                writer.group.save(self.root, "active")
            with self.assertRaisesRegex(RecordingError, "finalized"):
                writer.group.discard()
            self.assertTrue(writer.group.directory.exists())

    def test_save_never_overwrites_existing_directory(self):
        _, group = self.recorded()
        target = self.root / "existing"
        target.mkdir()
        (target / "mine").write_text("keep")
        with self.assertRaises(RecordingSaveError) as caught:
            group.save(self.root, "existing")
        self.assertIsInstance(caught.exception.cause, FileExistsError)
        self.assertEqual(caught.exception.published, ())
        self.assertEqual((target / "mine").read_text(), "keep")
        self.assertTrue(group.directory.exists())

    def test_partial_save_preserves_source_and_reports_published_files_for_retry(self):
        _, group = self.recorded()
        copy = shutil.copyfileobj
        calls = 0

        def fail_second(source, target, length):
            nonlocal calls
            calls += 1
            if calls == 2:
                target.write(b"partial")
                raise OSError("synthetic full target disk")
            copy(source, target, length)

        with patch.object(recording.shutil, "copyfileobj", side_effect=fail_second):
            with self.assertRaises(RecordingSaveError) as caught:
                group.save(self.root, "partial")
        self.assertEqual(caught.exception.published, ("original.wav",))
        self.assertEqual(caught.exception.failed_item, "effect.wav")
        self.assertIsNone(caught.exception.saved_group)
        self.assertEqual((self.root / "partial/effect.wav").read_bytes(), b"partial")
        self.assertFalse(group.saved)
        self.assertTrue(group.original_path.exists())
        self.assertTrue(group.effect_path.exists())
        saved = group.save(self.root, "retry")
        self.assertTrue(saved.saved)
        self.assertTrue((self.root / "partial").exists())

    def test_corrupted_copy_cannot_be_reported_as_saved(self):
        _, group = self.recorded()
        copy = shutil.copyfileobj

        def corrupt(source, target, length):
            copy(source, target, length)
            target.seek(-1, os.SEEK_END)
            target.write(b"\xff")

        with patch.object(recording.shutil, "copyfileobj", side_effect=corrupt):
            with self.assertRaises(RecordingSaveError) as caught:
                group.save(self.root, "corrupt")
        self.assertIn("differs", str(caught.exception))
        self.assertTrue(group.directory.exists())

    def test_cleanup_failure_surfaces_verified_saved_copy_and_retains_source(self):
        _, group = self.recorded()
        with patch.object(group, "_discard_owned", side_effect=PermissionError("cleanup denied")):
            with self.assertRaises(RecordingSaveError) as caught:
                group.save(self.root, "saved-but-cleanup-failed")
        saved = caught.exception.saved_group
        self.assertIsNotNone(saved)
        self.assertTrue(saved.saved)
        self.assertTrue(group.directory.exists())
        self.assertIn("temporary cleanup", caught.exception.failed_item)
        group.discard()
        self.assertTrue(saved.original_path.exists())

    def test_safe_discard_only_known_owned_files(self):
        _, group = self.recorded()
        outside = self.root / "unrelated"
        outside.write_text("keep")
        group.discard()
        self.assertFalse(group.directory.exists())
        self.assertEqual(outside.read_text(), "keep")

    def test_discard_refuses_unknown_files_and_directory_replacement(self):
        _, group = self.recorded()
        extra = group.directory / "user-file"
        extra.write_text("keep")
        with self.assertRaisesRegex(RecordingError, "Unknown"):
            group.discard()
        self.assertTrue(group.original_path.exists())
        extra.unlink()
        previous = group.directory
        moved = self.root / "moved"
        previous.rename(moved)
        previous.mkdir()
        (previous / ".owner").write_text(recording._OWNER)
        with self.assertRaisesRegex(RecordingError, "identity"):
            group.discard()
        self.assertTrue((moved / "original.wav").exists())

    def test_discard_refuses_track_and_directory_symlinks(self):
        _, group = self.recorded()
        victim = self.root / "victim"
        victim.write_text("private")
        group.original_path.unlink()
        group.original_path.symlink_to(victim)
        with self.assertRaisesRegex(RecordingError, "Symlink"):
            group.discard()
        self.assertEqual(victim.read_text(), "private")
        with self.assertRaises(RecordingRecoveryError):
            RecordingGroup.recover(group.directory)
        other = self.root / (recording._PREFIX + "a" * 32)
        other.symlink_to(group.directory, target_is_directory=True)
        self.assertNotIn(other, recoverable_groups(self.root))
        with self.assertRaises(RecordingRecoveryError):
            RecordingGroup.recover(other)

    def test_discard_unlink_failure_is_explicit_and_remaining_files_discoverable(self):
        _, group = self.recorded()
        with patch.object(recording.os, "unlink", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(PermissionError, "denied"):
                group.discard()
        self.assertTrue(group.directory.exists())
        self.assertIn(group.directory, recoverable_groups(self.root))

    def test_scanner_does_not_create_root_or_accept_arbitrary_groups(self):
        missing = self.root / "does-not-exist"
        self.assertEqual(recoverable_groups(missing), ())
        self.assertFalse(missing.exists())
        (self.root / "user-recording").mkdir()
        fake = self.root / (recording._PREFIX + "b" * 32)
        fake.mkdir()
        (fake / ".owner").write_bytes(b"\xff")
        _, group = self.recorded()
        self.assertEqual(recoverable_groups(self.root), (group.directory,))

    def test_explicit_delete_handles_listed_corrupt_or_missing_tracks_and_metadata(self):
        for problem in ("metadata", "missing-metadata", "missing-track", "corrupt-track",
                        "missing-journal"):
            with self.subTest(problem=problem):
                _, group = self.recorded(complete=False, reason="interrupted")
                if problem == "metadata":
                    group.metadata_path.write_bytes(b"\xff")
                elif problem == "missing-metadata":
                    group.metadata_path.unlink()
                elif problem == "missing-track":
                    group.effect_path.unlink()
                elif problem == "corrupt-track":
                    group.effect_path.write_bytes(b"bad wave")
                else:
                    (group.directory / "parameters.jsonl").unlink()
                self.assertIn(group.directory, recoverable_groups(self.root))
                with self.assertRaises(RecordingRecoveryError):
                    RecordingGroup.recover(group.directory)
                discard_recoverable(group.directory)
                self.assertFalse(group.directory.exists())

    def test_explicit_delete_rejects_saved_groups_user_directories_and_wrong_owner(self):
        _, group = self.recorded()
        saved = group.save(self.root, "saved-user-group")
        with self.assertRaises(RecordingRecoveryError):
            discard_recoverable(saved.directory)
        self.assertTrue(saved.original_path.exists())
        _, marked_saved = self.recorded()
        self.rewrite_metadata(marked_saved, lambda m: m.update(saved=True))
        with self.assertRaisesRegex(RecordingError, "saved"):
            discard_recoverable(marked_saved.directory)
        self.assertTrue(marked_saved.original_path.exists())
        user = self.root / "user-directory"
        user.mkdir()
        (user / ".owner").write_text(recording._OWNER)
        with self.assertRaises(RecordingRecoveryError):
            discard_recoverable(user)
        _, invalid = self.recorded()
        (invalid.directory / ".owner").write_text("wrong owner")
        with self.assertRaises(RecordingRecoveryError):
            discard_recoverable(invalid.directory)
        self.assertTrue(invalid.original_path.exists())

    def test_explicit_delete_rejects_unknown_files_and_symlinks_without_partial_deletion(self):
        _, group = self.recorded()
        unknown = group.directory / "personal-file"
        unknown.write_text("keep")
        with self.assertRaisesRegex(RecordingError, "Unknown"):
            discard_recoverable(group.directory)
        self.assertTrue(group.original_path.exists())
        unknown.unlink()
        victim = self.root / "victim"
        victim.write_text("keep")
        group.metadata_path.unlink()
        group.metadata_path.symlink_to(victim)
        with self.assertRaisesRegex(RecordingError, "Symlink"):
            discard_recoverable(group.directory)
        self.assertEqual(victim.read_text(), "keep")
        self.assertTrue(group.original_path.exists())
        alias = self.root / (recording._PREFIX + "c" * 32)
        alias.symlink_to(group.directory, target_is_directory=True)
        with self.assertRaises(RecordingRecoveryError):
            discard_recoverable(alias)

    def test_explicit_delete_revalidates_identity_and_surfaces_cleanup_errors(self):
        _, group = self.recorded()
        identity = recording._owned(group.directory)
        with patch.object(recording, "_owned", side_effect=[identity, (0, 0)]):
            with self.assertRaisesRegex(RecordingError, "changed"):
                discard_recoverable(group.directory)
        self.assertTrue(group.original_path.exists())
        with patch.object(recording.os, "unlink", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(PermissionError, "denied"):
                discard_recoverable(group.directory)
        self.assertIn(group.directory, recoverable_groups(self.root))

    def test_prepared_and_started_writer_lock_blocks_scan_recover_and_delete(self):
        writer = self.writer()
        for started in (False, True):
            if started:
                writer.start()
            self.assertIsNotNone(writer._owner_fd)
            self.assertNotIn(writer.group.directory, recoverable_groups(self.root))
            for operation in (RecordingGroup.recover, discard_recoverable):
                with self.assertRaisesRegex(RecordingRecoveryError, "locked"):
                    operation(writer.group.directory)
            self.assertTrue(writer.group.original_path.exists())
        group = writer.finish(complete=False, reason="user stopped")
        self.assertIsNone(writer._owner_fd)
        self.assertIn(group.directory, recoverable_groups(self.root))
        RecordingGroup.recover(group.directory)
        discard_recoverable(group.directory)
        self.assertFalse(group.directory.exists())

    def test_finish_before_start_releases_owner_lock(self):
        writer = self.writer()
        self.assertNotIn(writer.group.directory, recoverable_groups(self.root))
        group = writer.finish(complete=False, reason="cancelled before start")
        self.assertIsNone(writer._owner_fd)
        self.assertIn(group.directory, recoverable_groups(self.root))
        self.assertIs(writer.finish(complete=False), group)
        discard_recoverable(group.directory)

    def test_separate_process_owner_blocks_other_window_until_finalized(self):
        program = """
import sys
from pathlib import Path
from voice_changer.audio.recording import RecordingWriter
writer = RecordingWriter(48000, 1, 8, data_root=Path(sys.argv[1]))
print(writer.group.directory, flush=True)
sys.stdin.readline()
writer.finish(complete=False, reason='child finalized')
print('finalized', flush=True)
"""
        child = subprocess.Popen(
            [sys.executable, "-c", program, str(self.root)],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        try:
            directory = Path(child.stdout.readline().strip())
            self.assertTrue(directory.is_dir())
            self.assertNotIn(directory, recoverable_groups(self.root))
            for operation in (RecordingGroup.recover, discard_recoverable):
                with self.assertRaisesRegex(RecordingRecoveryError, "locked"):
                    operation(directory)
            output, error = child.communicate("\n", timeout=5)
            self.assertEqual(child.returncode, 0, error)
            self.assertEqual(output.strip(), "finalized")
            self.assertIn(directory, recoverable_groups(self.root))
            self.assertFalse(RecordingGroup.recover(directory).complete)
            discard_recoverable(directory)
            self.assertFalse(directory.exists())
        finally:
            if child.poll() is None:
                child.communicate("\n", timeout=5)

    def test_separate_process_cannot_recover_or_delete_parent_active_group(self):
        writer = self.writer()
        writer.start()
        program = """
import sys
from pathlib import Path
from voice_changer.audio.recording import (
    recoverable_groups, discard_recoverable, RecordingGroup, RecordingRecoveryError,
)
root, directory = map(Path, sys.argv[1:])
assert directory not in recoverable_groups(root)
for operation in (RecordingGroup.recover, discard_recoverable):
    try:
        operation(directory)
    except RecordingRecoveryError as exc:
        assert 'locked' in str(exc)
    else:
        raise AssertionError('active group was accessible')
assert (directory / 'original.wav').exists()
"""
        result = subprocess.run(
            [sys.executable, "-c", program, str(self.root), str(writer.group.directory)],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True, text=True, timeout=5, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        group = writer.finish(complete=False)
        self.assertIn(group.directory, recoverable_groups(self.root))

    def contender(self, directory):
        program = """
import sys
from pathlib import Path
from voice_changer.audio.recording import (
    RecordingGroup, RecordingRecoveryError, discard_recoverable, recoverable_groups,
)
directory = Path(sys.argv[1])
group = RecordingGroup.recover(directory)
print('ready', flush=True)
sys.stdin.readline()
assert directory not in recoverable_groups(directory.parent)
for operation in (
    lambda: RecordingGroup.recover(directory),
    lambda: discard_recoverable(directory),
    group.discard,
    lambda: group.save(directory.parent, 'contender-save'),
):
    try:
        operation()
    except Exception as exc:
        assert 'locked' in str(exc), repr(exc)
    else:
        raise AssertionError('concurrent operation succeeded')
assert group.original_path.exists()
print('protected', flush=True)
"""
        child = subprocess.Popen(
            [sys.executable, "-c", program, str(directory)],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(child.stdout.readline().strip(), "ready")
        return child

    def test_save_owns_source_across_copy_validation_and_cleanup_without_relocking(self):
        _, group = self.recorded()
        child = self.contender(group.directory)
        copy = shutil.copyfileobj
        contested = False

        def guarded_copy(source, target, length):
            nonlocal contested
            if not contested:
                contested = True
                output, error = child.communicate("\n", timeout=5)
                self.assertEqual(child.returncode, 0, error)
                self.assertEqual(output.strip(), "protected")
            copy(source, target, length)

        delete = recording._delete_owned_directory

        def guarded_delete(directory, identity):
            with self.assertRaisesRegex(RecordingRecoveryError, "locked"):
                with recording._owner_lock(directory):
                    self.fail("save dropped ownership before deletion")
            delete(directory, identity)

        try:
            with patch.object(recording.shutil, "copyfileobj", side_effect=guarded_copy), \
                 patch.object(recording, "_delete_owned_directory", side_effect=guarded_delete):
                saved = group.save(self.root, "save-protected")
            self.assertTrue(saved.saved)
            self.assertFalse(group.directory.exists())
        finally:
            if child.poll() is None:
                child.communicate("\n", timeout=5)

    def test_direct_discard_owns_source_across_deletion_against_other_process(self):
        _, group = self.recorded()
        child = self.contender(group.directory)
        delete = recording._delete_owned_directory

        def guarded_delete(directory, identity):
            output, error = child.communicate("\n", timeout=5)
            self.assertEqual(child.returncode, 0, error)
            self.assertEqual(output.strip(), "protected")
            delete(directory, identity)

        try:
            with patch.object(recording, "_delete_owned_directory", side_effect=guarded_delete):
                group.discard()
            self.assertFalse(group.directory.exists())
        finally:
            if child.poll() is None:
                child.communicate("\n", timeout=5)

    def test_recovery_rejects_missing_corrupt_truncated_and_misaligned_tracks(self):
        for problem in ("missing", "corrupt", "truncated", "source", "format", "traversal"):
            with self.subTest(problem=problem):
                _, group = self.recorded(complete=False, reason="test interrupted")
                if problem == "missing":
                    group.effect_path.unlink()
                elif problem == "corrupt":
                    group.effect_path.write_bytes(b"not a wav")
                elif problem == "truncated":
                    group.effect_path.write_bytes(group.effect_path.read_bytes()[:-2])
                elif problem == "source":
                    self.rewrite_metadata(
                        group, lambda m: m["tracks"]["effect"]["source_interval"].update(start=101),
                    )
                elif problem == "format":
                    self.rewrite_metadata(group, lambda m: m.update(samplerate=44100))
                elif problem == "traversal":
                    self.rewrite_metadata(
                        group, lambda m: m["tracks"]["effect"].update(file="../effect.wav"),
                    )
                with self.assertRaises(RecordingRecoveryError):
                    RecordingGroup.recover(group.directory)
                self.assertTrue(group.directory.exists())
                self.assertTrue(group.original_path.exists())

    def test_recovery_rejects_corrupt_metadata_and_parameter_journal(self):
        for problem in ("metadata", "journal", "missing-journal", "parameters", "interval"):
            with self.subTest(problem=problem):
                _, group = self.recorded()
                if problem == "metadata":
                    group.metadata_path.write_text("{bad")
                elif problem == "journal":
                    (group.directory / "parameters.jsonl").write_text("{bad")
                elif problem == "missing-journal":
                    (group.directory / "parameters.jsonl").unlink()
                elif problem == "parameters":
                    (group.directory / "parameters.jsonl").write_text(json.dumps({
                        "source_frame": 100, "revision": 0, "parameters": {"bypass": "false"},
                    }) + "\n")
                else:
                    self.rewrite_metadata(group, lambda m: m["source_interval"].update(
                        end_exclusive=999,
                    ))
                with self.assertRaises(RecordingRecoveryError):
                    RecordingGroup.recover(group.directory)
                self.assertTrue(group.original_path.exists())

    def test_recover_active_checkpoint_honestly_incomplete(self):
        _, group = self.recorded(complete=False, reason="process interrupted")
        self.rewrite_metadata(group, lambda m: m.update(finalized=False))
        recovered = RecordingGroup.recover(group.directory)
        self.assertFalse(recovered.complete)
        self.assertEqual(recovered.frames, 8)
        saved = recovered.save(self.root, "recovered")
        self.assertTrue(saved.saved)
        self.assertFalse(saved.complete)
        self.assertTrue(self.metadata(saved)["finalized"])

    def test_group_control_operations_are_exclusive(self):
        _, group = self.recorded()
        group._operation.acquire()
        try:
            with self.assertRaisesRegex(RecordingError, "in progress"):
                group.save(self.root, "busy")
            with self.assertRaisesRegex(RecordingError, "in progress"):
                group.discard()
        finally:
            group._operation.release()

    def test_save_name_traversal_is_rejected_without_writing_outside_group(self):
        _, group = self.recorded()
        for name in ("", ".", "..", "../escape", "/absolute", "a/b", "a\\b"):
            with self.subTest(name=name), self.assertRaises(RecordingSaveError):
                group.save(self.root, name)
        self.assertTrue(group.original_path.exists())
        self.assertFalse((self.root / "escape").exists())

    def test_default_xdg_path_is_lazy_and_environment_aware(self):
        with patch.dict(os.environ, {"XDG_DATA_HOME": str(self.root / "xdg")}):
            self.assertEqual(recording._root(None), self.root / "xdg/voice-changer/recordings")
            self.assertFalse((self.root / "xdg").exists())
        with patch.dict(os.environ, {"XDG_DATA_HOME": "relative"}), \
             patch.object(recording.Path, "home", return_value=self.root):
            self.assertEqual(recording._root(None),
                             self.root / ".local/share/voice-changer/recordings")

    def test_invalid_format_rejected_before_files_created(self):
        for rate, channels, blocksize in ((44100, 1, 8), (48000, 3, 8), (48000, 1, 0)):
            with self.assertRaises(ValueError):
                RecordingWriter(rate, channels, blocksize, data_root=self.root)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
