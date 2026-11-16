"""Negative controls for the synthetic no-AI test isolation."""

import unittest

from ai_optional_test_support import run_without_ai


class AIOptionalIsolationTests(unittest.TestCase):
    def test_external_ai_import_is_rejected(self):
        result = run_without_ai("\nimport torch\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AI dependency intentionally unavailable", result.stderr)

    def test_network_subprocess_and_audio_library_load_are_rejected(self):
        cases = (
            ("\nimport socket\nsocket.socket()\n", "operation outside test scope: socket."),
            ("\nimport subprocess\nsubprocess.run(['/usr/bin/true'])\n",
             "operation outside test scope: subprocess."),
            ("\nimport ctypes\nctypes.CDLL('libasound.so.2')\n",
             "audio-device library load outside test scope"),
        )
        for body, error in cases:
            with self.subTest(error=error):
                result = run_without_ai(body)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)


if __name__ == "__main__":
    unittest.main()
