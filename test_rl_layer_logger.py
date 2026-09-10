from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rl_layer_logger import (
    ACTION_SPECS,
    LayerLabelMap,
    LayerLabelSession,
    LayerTracker,
    VisualDefectAssessment,
    apply_action,
    apply_layer_labels_to_csv,
    assess_layer_photo,
    assess_layer_visual,
    base_print_settings,
    build_webcam_snapshot_url,
    calculate_reward,
    capture_layer_photo,
    configure_cnn,
    discretise,
    layer_photo_path_for,
    normalize_visual_label,
    publish_rl_speed_status,
    visual_bin,
)


class LayerTrackerTests(unittest.TestCase):
    def test_first_layer_after_second_starts(self) -> None:
        tracker = LayerTracker()
        self.assertIsNone(tracker.consume_layer_started(1))
        self.assertEqual(tracker.consume_layer_started(2), 1)
        self.assertEqual(tracker.consume_layer_started(3), 2)

    def test_height_fallback(self) -> None:
        tracker = LayerTracker()
        self.assertIsNone(tracker.consume_height(0.2, 0.2))
        self.assertEqual(tracker.consume_height(0.4, 0.2), 1)


class RlSupportTests(unittest.TestCase):
    def test_base_print_settings(self) -> None:
        speed, flow, fan = base_print_settings()
        self.assertEqual((speed, flow, fan), (170.0, 100.0, 100.0))

    def test_discretise_includes_visual_state(self) -> None:
        visual = VisualDefectAssessment("normal", 0.9, "cnn")
        state = discretise(75.0, 2, 2, 2, visual)
        self.assertEqual(state, (((2 * 6 + 2) * 6 + 2) * 5 + 2) * 3 + visual_bin(visual))

    def test_apply_action_changes_only_target_parameter(self) -> None:
        speed_action = next(index for index, spec in enumerate(ACTION_SPECS) if spec[0] == "speed" and spec[1] == 1)
        speed, flow, fan = apply_action(speed_action, 2, 3, 4)
        self.assertEqual((speed, flow, fan), (3, 3, 4))

    def test_reward_inside_window(self) -> None:
        self.assertEqual(calculate_reward(75.0), 1.0)

    def test_reward_penalizes_cold(self) -> None:
        self.assertLess(calculate_reward(40.0), 0.0)

    def test_reward_penalizes_cnn_defect(self) -> None:
        visual = VisualDefectAssessment("defective", 1.0, "cnn")
        self.assertLess(calculate_reward(75.0, visual), calculate_reward(75.0))

    def test_reward_bonus_for_normal_visual_and_temp_window(self) -> None:
        visual = VisualDefectAssessment("normal", 1.0, "cnn")
        self.assertGreater(calculate_reward(75.0, visual), calculate_reward(75.0))

    def test_publish_rl_speed_status_posts_json(self) -> None:
        calls: list[dict] = []

        def fake_post(url: str, json: dict, timeout: float) -> object:  # noqa: ANN001
            calls.append({"url": url, "json": json, "timeout": timeout})
            response = mock.Mock()
            response.raise_for_status.return_value = None
            return response

        with mock.patch("rl_layer_logger.requests.post", fake_post):
            publish_rl_speed_status(120.0, 3, layer=7)

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["url"].endswith("/api/rl-status"))
        self.assertEqual(calls[0]["json"]["speed_mms"], 120.0)
        self.assertEqual(calls[0]["json"]["print_number"], 3)
        self.assertEqual(calls[0]["json"]["layer"], 7)


class LayerPhotoTests(unittest.TestCase):
    def test_build_webcam_snapshot_url(self) -> None:
        self.assertEqual(
            build_webcam_snapshot_url("http://cimpi-prusamk4s.local"),
            "http://cimpi-prusamk4s.local/webcam/?action=snapshot",
        )

    def test_layer_photo_path_for(self) -> None:
        path = layer_photo_path_for(
            3,
            42,
            runs_root="print_runs",
            control_mode="rl",
            print_angle_deg=60.0,
        )
        self.assertEqual(
            path.as_posix(),
            "print_runs/print_003_rl_a60/photos/layer_0042.jpg",
        )

    def test_capture_layer_photo_writes_jpeg(self) -> None:
        jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 8 + b"\xff\xd9"
        response = mock.Mock()
        response.content = jpeg
        response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch("rl_layer_logger.requests.get", return_value=response):
                with mock.patch("rl_layer_logger.time.sleep"):
                    saved = capture_layer_photo(
                        "http://example.local",
                        1,
                        5,
                        runs_root=temp_dir,
                        settle_time=0.0,
                    )

            self.assertIsNotNone(saved)
            path = Path(saved)
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), jpeg)

    def test_assess_layer_photo_without_model_returns_unclassified(self) -> None:
        configure_cnn(enabled=True, model_path="models/does_not_exist.pt")
        assessment = assess_layer_photo("print_runs/print_001/photos/layer_0001.jpg")
        self.assertEqual(assessment, VisualDefectAssessment())

    def test_assess_layer_photo_uses_trained_model(self) -> None:
        from layer_defect_cnn import LabeledPhoto, save_model, train_model
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            normal_path = root / "print_001" / "photos" / "layer_0001.jpg"
            defective_path = root / "print_001" / "photos" / "layer_0002.jpg"
            normal_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (80, 60), color=(20, 180, 40)).save(normal_path)
            Image.new("RGB", (80, 60), color=(220, 40, 40)).save(defective_path)
            model_path = root / "model.pt"
            train_model(
                [
                    LabeledPhoto(normal_path, 1, 1, "normal"),
                    LabeledPhoto(defective_path, 1, 2, "defective"),
                    LabeledPhoto(normal_path, 1, 3, "normal"),
                ],
                output_path=model_path,
                epochs=2,
                batch_size=2,
                device="cpu",
            )
            configure_cnn(enabled=True, model_path=model_path, min_confidence=0.0, device="cpu")
            assessment = assess_layer_photo(normal_path)
            self.assertEqual(assessment.source, "cnn")
            self.assertIn(assessment.label, {"normal", "defective"})
            self.assertIsNotNone(assessment.confidence)


class LayerLabelTests(unittest.TestCase):
    def test_parse_range_spec(self) -> None:
        label_map = LayerLabelMap.parse("1-30:normal,31-50:defective,51-60:normal")
        self.assertEqual(label_map.lookup(10), VisualDefectAssessment("normal", 1.0, "manual"))
        self.assertEqual(label_map.lookup(40), VisualDefectAssessment("defective", 1.0, "manual"))
        self.assertEqual(label_map.lookup(55), VisualDefectAssessment("normal", 1.0, "manual"))
        self.assertIsNone(label_map.lookup(61))

    def test_parse_single_layer(self) -> None:
        label_map = LayerLabelMap.parse("7:defective")
        self.assertEqual(label_map.lookup(7).label, "defective")
        self.assertIsNone(label_map.lookup(6))

    def test_normalize_visual_label_aliases(self) -> None:
        self.assertEqual(normalize_visual_label("sagging"), "defective")
        self.assertEqual(normalize_visual_label("OK"), "normal")

    def test_assess_layer_visual_prefers_manual_labels(self) -> None:
        label_map = LayerLabelMap.parse("1-10:defective")
        assessment = assess_layer_visual(
            5,
            "print_runs/print_001/photos/layer_0005.jpg",
            manual_labels=label_map,
        )
        self.assertEqual(assessment, VisualDefectAssessment("defective", 1.0, "manual"))

    def test_load_label_session_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.json"
            path.write_text(
                json.dumps(
                    {
                        "default": "1-5:normal",
                        "print 2": "1-20:normal,21-40:defective",
                    }
                ),
                encoding="utf-8",
            )
            session = LayerLabelSession.load_file(path)
            self.assertEqual(session.for_print(1).lookup(3).label, "normal")
            self.assertEqual(session.for_print(2).lookup(25).label, "defective")

    def test_load_label_session_top_level_print_folder_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.json"
            path.write_text(
                json.dumps(
                    {
                        "print_001": "1-10:normal,11-20:defective",
                        "print 2": "1-5:defective",
                    }
                ),
                encoding="utf-8",
            )
            session = LayerLabelSession.load_file(path)
            self.assertEqual(session.for_print(1).lookup(15).label, "defective")
            self.assertEqual(session.for_print(2).lookup(3).label, "defective")

    def test_apply_layer_labels_to_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "print.csv"
            path.write_text(
                "print_number,layer,visual_defect,visual_confidence\n"
                "1,1,,\n"
                "1,2,,\n"
                "1,3,,\n",
                encoding="utf-8",
            )
            updated = apply_layer_labels_to_csv(
                path,
                LayerLabelMap.parse("1-2:normal,3:defective"),
            )
            self.assertEqual(updated, 3)
            text = path.read_text(encoding="utf-8")
            self.assertIn("1,normal,1.0000,manual", text.replace("\r\n", "\n"))
            self.assertIn("3,defective,1.0000,manual", text.replace("\r\n", "\n"))


if __name__ == "__main__":
    unittest.main()
