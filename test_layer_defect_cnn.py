from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from layer_defect_cnn import (
    CLASS_NAMES,
    LayerDefectCNN,
    LayerPhotoDataset,
    LabeledPhoto,
    build_labeled_dataset,
    load_model,
    parse_layer_photo_path,
    predict_layer_photo,
    preprocess_image,
    save_model,
    summarize_dataset,
    train_model,
)


def _write_photo(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 64), color=color).save(path, format="JPEG")


def _make_labeled_samples(root: Path) -> list[LabeledPhoto]:
    normal_path = root / "print_001" / "photos" / "layer_0001.jpg"
    defective_path = root / "print_001" / "photos" / "layer_0002.jpg"
    extra_normal = root / "print_001" / "photos" / "layer_0003.jpg"
    _write_photo(normal_path, (20, 180, 40))
    _write_photo(defective_path, (220, 40, 40))
    _write_photo(extra_normal, (30, 170, 50))
    return [
        LabeledPhoto(normal_path, 1, 1, "normal"),
        LabeledPhoto(defective_path, 1, 2, "defective"),
        LabeledPhoto(extra_normal, 1, 3, "normal"),
    ]


class LayerDefectCnnTests(unittest.TestCase):
    def test_parse_layer_photo_path_schema_folder(self) -> None:
        parsed = parse_layer_photo_path(
            "print_runs/print_003_mpc_a70/photos/layer_0042.jpg"
        )
        self.assertEqual(parsed, (3, 42))

    def test_parse_layer_photo_path_custom_run_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "base_overhang60" / "photos"
            run_dir.mkdir(parents=True)
            photo_path = run_dir / "layer_0015.jpg"
            _write_photo(photo_path, (100, 120, 140))
            (root / "base_overhang60" / "run_meta.json").write_text(
                '{"print_number": 1}',
                encoding="utf-8",
            )
            parsed = parse_layer_photo_path(photo_path)
            self.assertEqual(parsed, (1, 15))

    def test_model_forward_pass(self) -> None:
        model = LayerDefectCNN()
        batch = torch.randn(2, 3, 128, 128)
        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (2, len(CLASS_NAMES)))

    def test_preprocess_image_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.jpg"
            _write_photo(path, (100, 120, 140))
            tensor = preprocess_image(path, input_mode="luminance")
            self.assertEqual(tuple(tensor.shape), (3, 128, 128))

    def test_luminance_mode_differs_from_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.jpg"
            _write_photo(path, (200, 40, 40))
            rgb_tensor = preprocess_image(path, input_mode="rgb")
            lum_tensor = preprocess_image(path, input_mode="luminance")
            self.assertFalse(torch.allclose(rgb_tensor, lum_tensor))

    def test_train_and_predict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            samples = _make_labeled_samples(root)
            model_path = root / "model.pt"
            result = train_model(
                samples,
                output_path=model_path,
                epochs=3,
                batch_size=2,
                val_fraction=0.34,
                device="cpu",
            )
            self.assertTrue(model_path.exists())
            self.assertGreaterEqual(result["best_val_accuracy"], 0.0)

            prediction = predict_layer_photo(
                samples[0].path,
                model_path=model_path,
                min_confidence=0.0,
                device="cpu",
            )
            self.assertIsNotNone(prediction)
            assert prediction is not None
            self.assertIn(prediction.label, CLASS_NAMES)
            self.assertGreater(prediction.confidence, 0.0)

    def test_build_labeled_dataset_from_label_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _make_labeled_samples(root)
            samples = build_labeled_dataset(
                root,
                label_spec="1:normal,2:defective,3:normal",
            )
            self.assertEqual(len(samples), 3)
            counts = summarize_dataset(samples)
            self.assertEqual(counts["normal"], 2)
            self.assertEqual(counts["defective"], 1)

    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            model = LayerDefectCNN()
            save_model(model, path, metadata={"test": True})
            loaded = load_model(path, device=torch.device("cpu"))
            source = torch.randn(1, 3, 128, 128)
            torch.testing.assert_close(model(source), loaded(source))

    def test_dataset_returns_class_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            samples = _make_labeled_samples(root)
            dataset = LayerPhotoDataset(samples, augment=False)
            image, label_index = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 128, 128))
            self.assertIn(label_index, (0, 1))


if __name__ == "__main__":
    unittest.main()
