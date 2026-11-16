import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from voice_changer.ai import hevc, model_catalog
from voice_changer.ai.llvc import MODEL_SHA256


class ModelCatalogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environ = patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.root)})
        environ.start()
        self.addCleanup(environ.stop)
        self.names_file = self.root / "voice-changer" / "model-names.json"

    def test_published_models_are_matched_by_filename_or_sha256(self):
        by_name = model_catalog.find_entry(Path("/models/FEMALE_VOICE_OLD_HQ.ONNX"))
        self.assertEqual(by_name.entry_id, "hevc_old_hq")
        by_digest = model_catalog.find_entry(Path("/models/renamed.onnx"), MODEL_SHA256)
        self.assertEqual(by_digest.entry_id, "llvc_koe")
        self.assertIsNone(
            model_catalog.find_entry(Path("/models/female_voice_old.onnx"), None, "llvc_onnx"),
        )

    def test_default_labels_describe_tone_not_live_capability(self):
        self.assertEqual(model_catalog.configured_label(), "晓晓 / Xiaoxiao")
        self.assertEqual(
            model_catalog.display_label(Path("/models/female_voice_hfg_new.onnx")),
            "自然女生 / Natural Female",
        )
        self.assertEqual(
            model_catalog.display_label(Path("/models/female_voice_old.onnx")),
            "柔和女声 / Gentle Female",
        )
        for entry_id in ("hevc_hfg_new", "hevc_old_hq", "hevc_old"):
            self.assertIs(model_catalog.ENTRY_BY_ID[entry_id].live_reason, hevc.LIVE_REASON)
        self.assertIsNone(model_catalog.ENTRY_BY_ID["llvc_koe"].live_reason)

    def test_unknown_models_fall_back_to_their_filename_stem(self):
        self.assertEqual(
            model_catalog.display_label(Path("/models/my-model.onnx")), "my-model",
        )

    def test_names_file_path_follows_xdg_config_home(self):
        self.assertEqual(
            model_catalog.names_path(), self.root / "voice-changer" / "model-names.json",
        )

    def test_overrides_round_trip_through_the_names_file(self):
        model_catalog.save_name_overrides({"hevc_old": "柔和女声（本地）"})
        payload = json.loads(self.names_file.read_text(encoding="utf-8"))
        self.assertEqual(payload["format"], model_catalog.SCHEMA)
        overrides, problem = model_catalog.load_name_overrides()
        self.assertIsNone(problem)
        self.assertEqual(overrides, {"hevc_old": "柔和女声（本地）"})
        self.assertEqual(
            model_catalog.display_label(
                Path("/models/female_voice_old.onnx"), None, "hevc_onnx", overrides,
            ),
            "柔和女声（本地）",
        )
        self.assertEqual(
            model_catalog.display_label(
                Path("/models/female_voice_old.onnx"), None, "hevc_onnx", {},
            ),
            "柔和女声 / Gentle Female",
        )

    def test_sha_override_wins_for_an_unrecognised_filename(self):
        digest = "a" * 64
        overrides = {f"sha:{digest}": "自定义音色"}
        self.assertEqual(
            model_catalog.display_label(Path("/models/custom.onnx"), digest, "hevc_onnx", overrides),
            "自定义音色",
        )

    def test_damaged_or_mismatched_name_files_are_reported_not_fatal(self):
        self.names_file.parent.mkdir(parents=True)
        self.names_file.write_text("{not json", encoding="utf-8")
        overrides, problem = model_catalog.load_name_overrides()
        self.assertEqual(overrides, {})
        self.assertIn("not valid JSON", problem)

        self.names_file.write_text(
            json.dumps({"format": "someone-else", "names": {}}), encoding="utf-8",
        )
        overrides, problem = model_catalog.load_name_overrides()
        self.assertEqual(overrides, {})
        self.assertIn("format mismatch", problem)

    def test_invalid_entries_are_skipped_and_valid_ones_kept(self):
        self.names_file.parent.mkdir(parents=True)
        self.names_file.write_text(json.dumps({
            "format": model_catalog.SCHEMA,
            "names": {
                "hevc_old": "x" * (model_catalog.NAME_LIMIT + 1),
                "llvc_koe": "  好名字  ",
            },
        }), encoding="utf-8")
        overrides, problem = model_catalog.load_name_overrides()
        self.assertIsNone(problem)
        self.assertEqual(overrides, {"llvc_koe": "好名字"})

    def test_rename_persists_a_label_and_empty_restores_the_default(self):
        overrides = {}
        path = Path("/models/female_voice_old_hq.onnx")
        digest = model_catalog.ENTRY_BY_ID["hevc_old_hq"].sha256[0]
        label = model_catalog.rename(path, digest, "hevc_onnx", "清亮女声·改", overrides)
        self.assertEqual(label, "清亮女声·改")
        self.assertEqual(overrides, {"hevc_old_hq": "清亮女声·改"})
        stored, problem = model_catalog.load_name_overrides()
        self.assertIsNone(problem)
        self.assertEqual(stored, overrides)
        restored = model_catalog.rename(path, digest, "hevc_onnx", "", overrides)
        self.assertEqual(restored, "清亮女声 / Bright Female")
        self.assertEqual(overrides, {})
        self.assertEqual(
            model_catalog.load_name_overrides()[0], {},
        )

    def test_rename_rejects_unidentified_models_and_overlong_names(self):
        with self.assertRaisesRegex(ValueError, "not identified"):
            model_catalog.rename(Path("/models/unknown.onnx"), None, "hevc_onnx", "x", {})
        with self.assertRaisesRegex(ValueError, "48"):
            model_catalog.rename(
                Path("/models/unknown.onnx"), "b" * 64, "hevc_onnx",
                "x" * (model_catalog.NAME_LIMIT + 1), {},
            )
        overrides = {}
        model_catalog.rename(
            Path("/models/unknown.onnx"), "b" * 64, "hevc_onnx", "自定义", overrides,
        )
        self.assertEqual(overrides, {f"sha:{'b' * 64}": "自定义"})


    def test_rknn_packages_inherit_their_source_onnx_name(self):
        for filename in (
            "female_voice_hfg_new_rknn.rknn",
            "female_voice_hfg_new-rknn.rknn",
            "female_voice_hfg_new_rknn_fp16.rknn",
            "female_voice_hfg_new_rknn_fp16_model.rknn",
        ):
            with self.subTest(filename=filename):
                path = Path("/models") / filename
                entry = model_catalog.find_entry(path, None, "hevc_rknn")
                self.assertIsNotNone(entry)
                self.assertEqual(entry.entry_id, "hevc_hfg_new")
                self.assertEqual(
                    model_catalog.display_label(path, None, "hevc_rknn"),
                    "自然女生 / Natural Female",
                )

    def test_rknn_rename_is_stored_under_the_catalog_entry_id(self):
        rknn_digest = hashlib.sha256(b"converted rknn package").hexdigest()
        path = Path("/models/female_voice_hfg_new_rknn.rknn")
        overrides = {}
        label = model_catalog.rename(
            path, rknn_digest, "hevc_rknn", "我的自然女声", overrides,
        )
        self.assertEqual(label, "我的自然女声")
        self.assertEqual(overrides, {"hevc_hfg_new": "我的自然女声"})
        stored, problem = model_catalog.load_name_overrides()
        self.assertIsNone(problem)
        self.assertEqual(stored, overrides)
        self.assertEqual(
            model_catalog.display_label(path, rknn_digest, "hevc_rknn", overrides),
            "我的自然女声",
        )


if __name__ == "__main__":
    unittest.main()
