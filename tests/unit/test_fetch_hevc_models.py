"""Synthetic tests for the optional HEEVoicechanger weight fetcher; no network."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "fetch_hevc_models", ROOT / "scripts" / "fetch_hevc_models.py",
)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self, _size):
        data, self._payload = self._payload, b""
        return data


class FetchModelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="fetch-hevc-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_known_models_are_the_three_public_hevc_graphs(self):
        self.assertEqual(
            set(fetch.MODELS),
            {"female_voice_hfg_new.onnx", "female_voice_old_hq.onnx", "female_voice_old.onnx"},
        )
        for digest, size in fetch.MODELS.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertGreater(size, 13_000_000)

    def test_default_destination_uses_xdg_data_home(self):
        with patch.dict(os.environ, {"XDG_DATA_HOME": str(self.directory)}):
            self.assertEqual(
                fetch.default_destination(), self.directory / "voice-changer" / "models",
            )

    def test_existing_good_file_is_kept_without_downloading(self):
        target = self.directory / "model.onnx"
        target.write_bytes(b"synthetic model")
        with patch.object(fetch.urllib.request, "urlopen") as urlopen:
            result = fetch._fetch(
                target.name, fetch._sha256(target), target.stat().st_size, self.directory,
            )
        self.assertEqual(result, target)
        urlopen.assert_not_called()

    def test_existing_mismatched_file_is_refused_and_kept(self):
        target = self.directory / "model.onnx"
        target.write_bytes(b"user file")
        with self.assertRaisesRegex(ValueError, "unexpected SHA-256"):
            fetch._fetch(target.name, "0" * 64, 9, self.directory)
        self.assertEqual(target.read_bytes(), b"user file")

    def test_checksum_mismatch_deletes_the_partial_download(self):
        with patch.object(
            fetch.urllib.request, "urlopen", return_value=FakeResponse(b"wrong weights"),
        ):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                fetch._fetch("female_voice_old.onnx", "0" * 64, 13, self.directory)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_list_prints_every_model_without_touching_the_network(self):
        output = io.StringIO()
        with patch.object(fetch.urllib.request, "urlopen") as urlopen:
            with contextlib.redirect_stdout(output):
                self.assertEqual(fetch.main(["--list"]), 0)
        urlopen.assert_not_called()
        for name in fetch.MODELS:
            self.assertIn(name, output.getvalue())

    def test_unknown_model_names_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                fetch.main(["not-a-model.onnx"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
