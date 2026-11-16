"""Synthetic recording remains available without optional AI dependencies."""

import unittest

from ai_optional_test_support import run_without_ai


RECORDING_TEST = r"""
import unittest
from unittest.mock import patch
import test_recording

original_temporary_directory = test_recording.tempfile.TemporaryDirectory
created = []
def isolated_temporary_directory(*args, **kwargs):
    if kwargs.get("prefix") != ".recording-test-" or Path(kwargs.get("dir", "")) != root:
        raise RuntimeError("unexpected temporary-directory request")
    kwargs["dir"] = os.environ["TMPDIR"]
    directory = original_temporary_directory(*args, **kwargs)
    created.append(directory.name)
    return directory

names = (
    "test_pcm16_stereo_source_interval_and_streamed_parameters",
    "test_save_publishes_verified_full_group_and_removes_only_temporary_copy",
    "test_save_never_overwrites_existing_directory",
    "test_partial_save_preserves_source_and_reports_published_files_for_retry",
    "test_safe_discard_only_known_owned_files",
)
suite = unittest.TestSuite(test_recording.RecordingTests(name) for name in names)
with patch.object(test_recording.tempfile, "TemporaryDirectory", side_effect=isolated_temporary_directory):
    result = unittest.TextTestRunner(verbosity=1).run(suite)
if (result.testsRun != len(names) or not result.wasSuccessful() or result.skipped
        or result.expectedFailures or result.unexpectedSuccesses):
    raise RuntimeError("AI-optional recording checks failed")
if len(created) != len(names) or any(os.path.lexists(name) for name in created):
    raise RuntimeError("synthetic recording directories were not cleaned")
if any(name in sys.modules for name in ("sounddevice", "pyaudio", "alsaaudio", "PySide6", "shiboken6")):
    raise RuntimeError("an audio-device or Qt package was loaded")
assert_ai_unloaded()
print("AI-optional synthetic recording passed")
"""


class AIOptionalRecordingTests(unittest.TestCase):
    def test_existing_recording_and_save_without_ai_dependencies(self):
        result = run_without_ai(RECORDING_TEST)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("AI-optional synthetic recording passed", result.stdout)
        self.assertIn("Ran 5 tests", result.stderr)


if __name__ == "__main__":
    unittest.main()
