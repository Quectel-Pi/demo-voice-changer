import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from voice_changer.ai.cosyvoice_tts import (
    GeneratedAudio, TtsError, synthesize, validate_model_dir, write_wave,
)
from voice_changer.ai.profiles import ProfileStore


def write_reference(path: Path) -> Path:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(np.zeros(24000, dtype="<i2").tobytes())
    return path


def make_model_dir(path: Path) -> Path:
    path.mkdir()
    for name in (
        "cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx",
        "spk2info.pt", "llm.pt", "flow.pt", "hift.pt",
    ):
        (path / name).write_bytes(b"fixture")
    (path / "CosyVoice-BlankEN").mkdir()
    return path


class FakeModel:
    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def inference_zero_shot(self, text, prompt_text, prompt_wav, *, stream):
        self.calls.append((text, prompt_text, prompt_wav, stream))
        return iter((
            {"tts_speech": SimpleNamespace(
                detach=lambda: SimpleNamespace(
                    cpu=lambda: SimpleNamespace(numpy=lambda: np.array([[0.1, 0.2]], dtype=np.float32))
                )
            )},
            {"tts_speech": np.array([[0.3]], dtype=np.float32)},
        ))


class CosyVoiceTtsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = ProfileStore(self.root / "profiles")
        self.reference = write_reference(self.root / "source.wav")
        self.profile = self.store.create(
            "本人", relation="self", consent=True, prompt_text="你好，今天天气不错。",
            source=self.reference,
        )
        self.model_dir = make_model_dir(self.root / "model")

    def tearDown(self):
        self.temporary.cleanup()

    def test_synthesizes_with_exact_reference_prompt_and_local_audio(self):
        model = FakeModel()
        loaded = []

        def loader(model_dir):
            loaded.append(model_dir)
            return model

        result = synthesize(self.profile, "  测试朗读  ", self.model_dir, model_loader=loader)

        self.assertEqual(loaded, [self.model_dir])
        self.assertEqual(model.calls, [(
            "测试朗读", "You are a helpful assistant.<|endofprompt|>你好，今天天气不错。",
            str(self.profile.reference_path), False,
        )])
        np.testing.assert_array_equal(result.samples, np.array([0.1, 0.2, 0.3], dtype=np.float32))
        self.assertEqual(result.sample_rate, 24000)

    def test_rejects_empty_text_and_empty_prompt_before_loading_model(self):
        called = []
        with self.assertRaisesRegex(TtsError, "Enter text"):
            synthesize(self.profile, " ", self.model_dir, model_loader=lambda _: called.append(True))
        self.assertEqual(called, [])

        without_prompt = self.store.create(
            "空提示", relation="self", consent=True, prompt_text="", source=self.reference,
        )
        with self.assertRaisesRegex(TtsError, "matching prompt text"):
            synthesize(without_prompt, "Hello", self.model_dir, model_loader=lambda _: called.append(True))
        self.assertEqual(called, [])

    def test_rejects_incomplete_or_symlink_model_directory(self):
        incomplete = self.root / "incomplete"
        incomplete.mkdir()
        with self.assertRaisesRegex(TtsError, "missing"):
            validate_model_dir(incomplete)

        link = self.root / "model-link"
        link.symlink_to(self.model_dir, target_is_directory=True)
        with self.assertRaisesRegex(TtsError, "symbolic link"):
            validate_model_dir(link)

    def test_model_loader_is_not_called_when_profile_is_damaged(self):
        damaged_dir = self.profile.directory
        (damaged_dir / "profile.json").write_text("{", encoding="utf-8")
        damaged = next(item for item in self.store.list() if item.profile_id == self.profile.profile_id)
        called = []
        with self.assertRaisesRegex(TtsError, "valid local voice profile"):
            synthesize(damaged, "Hello", self.model_dir, model_loader=lambda _: called.append(True))
        self.assertEqual(called, [])

    def test_write_wave_is_pcm16_mono_and_refuses_unrequested_overwrite(self):
        destination = self.root / "generated.wav"
        result = write_wave(
            destination, GeneratedAudio(np.array([-1.0, 0.0, 1.0], dtype=np.float32), 24000),
        )
        self.assertEqual(result, destination)
        with wave.open(str(destination), "rb") as reader:
            self.assertEqual(
                (reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getnframes()),
                (24000, 1, 2, 3),
            )
            self.assertEqual(np.frombuffer(reader.readframes(3), dtype="<i2").tolist(), [-32767, 0, 32767])
        with self.assertRaisesRegex(TtsError, "already exists"):
            write_wave(destination, GeneratedAudio(np.ones(1, dtype=np.float32), 24000))
        write_wave(
            destination, GeneratedAudio(np.array([0.5], dtype=np.float32), 24000),
            overwrite=True,
        )
        with wave.open(str(destination), "rb") as reader:
            self.assertEqual(reader.getnframes(), 1)

    def test_write_wave_rejects_invalid_audio_and_symlink(self):
        with self.assertRaisesRegex(TtsError, "invalid audio"):
            write_wave(
                self.root / "bad.wav",
                GeneratedAudio(np.array([np.nan], dtype=np.float32), 24000),
            )
        target = self.root / "target.wav"
        target.write_bytes(b"keep")
        link = self.root / "link.wav"
        link.symlink_to(target)
        with self.assertRaisesRegex(TtsError, "symbolic link"):
            write_wave(link, GeneratedAudio(np.ones(1, dtype=np.float32), 24000), overwrite=True)
        self.assertEqual(target.read_bytes(), b"keep")

    def test_import_does_not_load_optional_audio_or_model_runtime(self):
        source_root = str(Path(__file__).resolve().parents[2] / "src")
        code = (
            "import sys\n"
            "import voice_changer.ai.cosyvoice_tts\n"
            "print(','.join(name for name in ('numpy', 'torch', 'torchaudio', 'cosyvoice') "
            "if name in sys.modules))\n"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONPATH": source_root},
        )
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
