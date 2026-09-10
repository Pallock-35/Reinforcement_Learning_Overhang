"""
layer_defect_cnn.py
===================
Small CNN for classifying per-layer webcam photos as normal or defective.

Training data comes from:
  - print_runs/print_NNN_<mode>_a<angle>/photos/layer_MMMM.jpg images
  - manual labels in CSV logs (visual_defect column)
  - layer label range specs (LayerLabelMap / layer_labels.json)

By default, photos are converted to contrast-normalized luminance before the CNN
so different filament colors are less likely to confuse the model. Set input mode
to "rgb" to use raw color (not recommended for multi-filament training).

Saved weights default to models/layer_defect_cnn.pt.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from print_run_naming import parse_print_number_from_run_key, print_run_folder_name

InputMode = Literal["luminance", "rgb"]
VALID_INPUT_MODES: tuple[str, ...] = ("luminance", "rgb")
DEFAULT_INPUT_MODE: InputMode = (
    "luminance"
    if os.environ.get("PRUSASAM_CNN_INPUT_MODE", "luminance").strip().lower() != "rgb"
    else "rgb"
)


def normalize_visual_label(raw: str) -> str:
    token = raw.strip().lower()
    if token in {"normal", "ok", "good"}:
        return "normal"
    if token in {"defective", "defect", "sagging", "bad"}:
        return "defective"
    raise ValueError(f"Unknown layer label '{raw}'.")


IMAGE_SIZE = 128
CLASS_NAMES: tuple[str, ...] = ("normal", "defective")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
DEFAULT_MODEL_PATH = Path("models/layer_defect_cnn.pt")

LAYER_PHOTO_PATH_RE = re.compile(
    r"[/\\]([^/\\]+)[/\\](?:(?:photos)[/\\])?layer_(\d+)\.jpe?g$",
    re.IGNORECASE,
)
@dataclass(frozen=True)
class LabeledPhoto:
    path: Path
    print_number: int
    layer: int
    label: str


@dataclass(frozen=True)
class CnnPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class LayerDefectCNN(nn.Module):
    """Compact CNN for 128x128 RGB webcam stills."""

    def __init__(self, num_classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        flattened = torch.flatten(features, 1)
        return self.classifier(flattened)


class LayerPhotoDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        samples: Sequence[LabeledPhoto],
        augment: bool = False,
        *,
        input_mode: InputMode = DEFAULT_INPUT_MODE,
    ) -> None:
        self._samples = list(samples)
        self._augment = augment
        self._input_mode = input_mode

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self._samples[index]
        tensor = preprocess_image(
            sample.path,
            augment=self._augment,
            input_mode=self._input_mode,
        )
        return tensor, CLASS_TO_INDEX[sample.label]


def _parse_print_number_from_folder(folder: str) -> int | None:
    return parse_print_number_from_run_key(folder)


def _print_number_from_run_meta(photo_path: Path) -> int | None:
    run_dir = photo_path.parent.parent if photo_path.parent.name == "photos" else photo_path.parent
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    print_number = payload.get("print_number")
    if isinstance(print_number, int):
        return print_number
    return None


def parse_layer_photo_run_folder(path: str | Path) -> str | None:
    match = LAYER_PHOTO_PATH_RE.search(str(path).replace("\\", "/"))
    return match.group(1) if match else None


def parse_layer_photo_path(path: str | Path) -> tuple[int, int] | None:
    photo_path = Path(path)
    match = LAYER_PHOTO_PATH_RE.search(str(photo_path).replace("\\", "/"))
    if not match:
        return None
    print_number = _parse_print_number_from_folder(match.group(1))
    if print_number is None:
        print_number = _print_number_from_run_meta(photo_path)
    if print_number is None:
        return None
    return print_number, int(match.group(2))


def normalize_input_mode(value: str | None) -> InputMode:
    mode = (value or DEFAULT_INPUT_MODE).strip().lower()
    if mode not in VALID_INPUT_MODES:
        raise ValueError(f"input_mode must be one of {VALID_INPUT_MODES}, got '{value}'.")
    return mode  # type: ignore[return-value]


def filter_layer_photo_image(image: Image.Image, input_mode: InputMode) -> Image.Image:
    """Reduce filament-color bias before CNN training or inference."""
    rgb = image.convert("RGB")
    if input_mode == "rgb":
        return rgb

    array = np.asarray(rgb, dtype=np.float32)
    luminance = (
        0.299 * array[:, :, 0]
        + 0.587 * array[:, :, 1]
        + 0.114 * array[:, :, 2]
    )
    lo = float(np.percentile(luminance, 5))
    hi = float(np.percentile(luminance, 95))
    if hi > lo:
        luminance = (luminance - lo) / (hi - lo) * 255.0
    luminance = np.clip(luminance, 0.0, 255.0).astype(np.uint8)
    gray = Image.fromarray(luminance, mode="L")
    return gray.convert("RGB")


def preprocess_image(
    path: str | Path,
    *,
    augment: bool = False,
    input_mode: InputMode | None = None,
) -> torch.Tensor:
    mode = normalize_input_mode(input_mode)
    with Image.open(path) as image:
        filtered = filter_layer_photo_image(image, mode)
        if augment and random.random() < 0.5:
            filtered = filtered.transpose(Image.FLIP_LEFT_RIGHT)
        resized = filtered.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor


def discover_layer_photos(photo_root: str | Path) -> list[Path]:
    root = Path(photo_root)
    if not root.exists():
        return []
    paths: list[Path] = []
    for path in root.rglob("layer_*.jpg"):
        if path.is_file() and parse_layer_photo_path(path) is not None:
            paths.append(path)
    return sorted(paths)


def labels_from_csv_logs(log_paths: Iterable[str | Path]) -> dict[tuple[int, int], str]:
    labels: dict[tuple[int, int], str] = {}
    for log_path in log_paths:
        path = Path(log_path)
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label_text = (row.get("visual_defect") or "").strip().lower()
                if not label_text:
                    continue
                try:
                    label = normalize_visual_label(label_text)
                except ValueError:
                    continue

                photo_path = (row.get("layer_photo") or "").strip()
                parsed = parse_layer_photo_path(photo_path) if photo_path else None
                if parsed is None:
                    layer_text = (row.get("layer") or "").strip()
                    print_text = (row.get("print_number") or "").strip()
                    if layer_text.isdigit() and print_text.isdigit():
                        parsed = (int(print_text), int(layer_text))
                if parsed is None:
                    continue
                labels[parsed] = label
    return labels


def labels_from_label_session(
    photo_paths: Sequence[Path],
    labels_file: str | Path,
) -> dict[tuple[int, int], str]:
    from rl_layer_logger import LayerLabelSession

    session = LayerLabelSession.load_file(labels_file)
    labels: dict[tuple[int, int], str] = {}
    for photo_path in photo_paths:
        parsed = parse_layer_photo_path(photo_path)
        if parsed is None:
            continue
        print_number, layer = parsed
        folder = parse_layer_photo_run_folder(photo_path)
        if folder is None:
            folder = print_run_folder_name(print_number)
        assessment = session.for_run_folder(folder).lookup(layer)
        if assessment is not None and assessment.label:
            labels[parsed] = assessment.label
    return labels


def build_labeled_dataset(
    photo_root: str | Path,
    *,
    log_paths: Sequence[str | Path] | None = None,
    labels_file: str | Path | None = None,
    label_spec: str | None = None,
) -> list[LabeledPhoto]:
    photo_paths = discover_layer_photos(photo_root)
    label_lookup: dict[tuple[int, int], str] = {}

    if log_paths:
        label_lookup.update(labels_from_csv_logs(log_paths))
    if labels_file:
        label_lookup.update(labels_from_label_session(photo_paths, labels_file))
    if label_spec:
        from rl_layer_logger import LayerLabelMap

        label_map = LayerLabelMap.parse(label_spec)
        for photo_path in photo_paths:
            parsed = parse_layer_photo_path(photo_path)
            if parsed is None:
                continue
            print_number, layer = parsed
            assessment = label_map.lookup(layer)
            if assessment is not None and assessment.label:
                label_lookup.setdefault(parsed, assessment.label)

    samples: list[LabeledPhoto] = []
    for photo_path in photo_paths:
        parsed = parse_layer_photo_path(photo_path)
        if parsed is None:
            continue
        label = label_lookup.get(parsed)
        if label is None:
            continue
        print_number, layer = parsed
        samples.append(
            LabeledPhoto(
                path=photo_path,
                print_number=print_number,
                layer=layer,
                label=label,
            )
        )
    return samples


def choose_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(
    model: LayerDefectCNN,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    input_mode: InputMode = DEFAULT_INPUT_MODE,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "class_names": list(CLASS_NAMES),
        "image_size": IMAGE_SIZE,
        "input_mode": normalize_input_mode(input_mode),
        "metadata": metadata or {},
    }
    torch.save(payload, output_path)
    return output_path


def load_model_checkpoint(path: str | Path) -> dict[str, Any]:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"CNN model not found: {model_path}")
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid model checkpoint: {model_path}")
    return payload


def checkpoint_input_mode(payload: dict[str, Any]) -> InputMode:
    stored = payload.get("input_mode")
    if stored:
        return normalize_input_mode(str(stored))
    nested = payload.get("metadata")
    if isinstance(nested, dict) and nested.get("input_mode"):
        return normalize_input_mode(str(nested["input_mode"]))
    return "rgb"


def load_model(path: str | Path, device: torch.device | None = None) -> LayerDefectCNN:
    resolved_device = device or choose_device()
    payload = load_model_checkpoint(path)
    model = LayerDefectCNN(num_classes=len(payload.get("class_names", CLASS_NAMES)))
    model.load_state_dict(payload["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model


class LayerDefectClassifier:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        min_confidence: float = 0.5,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.min_confidence = min_confidence
        self.device = choose_device(device)
        self._model: LayerDefectCNN | None = None
        self._input_mode: InputMode = DEFAULT_INPUT_MODE

    def _ensure_model(self) -> LayerDefectCNN:
        if self._model is None:
            payload = load_model_checkpoint(self.model_path)
            self._input_mode = checkpoint_input_mode(payload)
            self._model = load_model(self.model_path, self.device)
        return self._model

    def predict(self, photo_path: str | Path) -> CnnPrediction | None:
        path = Path(photo_path)
        if not path.exists():
            return None

        model = self._ensure_model()
        tensor = preprocess_image(
            path,
            augment=False,
            input_mode=self._input_mode,
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        probs = {
            CLASS_NAMES[index]: float(probabilities[index].item())
            for index in range(len(CLASS_NAMES))
        }
        best_index = int(torch.argmax(probabilities).item())
        confidence = probs[CLASS_NAMES[best_index]]
        if confidence < self.min_confidence:
            return None
        return CnnPrediction(
            label=CLASS_NAMES[best_index],
            confidence=confidence,
            probabilities=probs,
        )


_CLASSIFIER_CACHE: dict[tuple[str, float, str | None], LayerDefectClassifier] = {}


def get_classifier(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    min_confidence: float = 0.5,
    device: str | None = None,
) -> LayerDefectClassifier:
    key = (str(Path(model_path)), min_confidence, device)
    if key not in _CLASSIFIER_CACHE:
        _CLASSIFIER_CACHE[key] = LayerDefectClassifier(
            model_path,
            min_confidence=min_confidence,
            device=device,
        )
    return _CLASSIFIER_CACHE[key]


def predict_layer_photo(
    photo_path: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    min_confidence: float = 0.5,
    device: str | None = None,
) -> CnnPrediction | None:
    classifier = get_classifier(
        model_path,
        min_confidence=min_confidence,
        device=device,
    )
    return classifier.predict(photo_path)


def train_model(
    samples: Sequence[LabeledPhoto],
    *,
    output_path: str | Path = DEFAULT_MODEL_PATH,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.2,
    device: str | None = None,
    input_mode: InputMode = DEFAULT_INPUT_MODE,
) -> dict[str, Any]:
    if len(samples) < 2:
        raise ValueError("Need at least 2 labeled photos to train the CNN.")
    if len({sample.label for sample in samples}) < 2:
        raise ValueError("Need at least one normal and one defective labeled photo.")

    resolved_input_mode = normalize_input_mode(input_mode)
    resolved_device = choose_device(device)
    dataset = LayerPhotoDataset(samples, augment=True, input_mode=resolved_input_mode)

    val_size = max(1, int(round(len(dataset) * val_fraction)))
    if val_size >= len(dataset):
        val_size = 1
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, train_size), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, val_size), shuffle=False)

    model = LayerDefectCNN().to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images = images.to(resolved_device)
            labels = labels.to(resolved_device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * images.size(0)
            predictions = torch.argmax(logits, dim=1)
            train_correct += int((predictions == labels).sum().item())
            train_total += images.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(resolved_device)
                labels = labels.to(resolved_device)
                logits = model(images)
                predictions = torch.argmax(logits, dim=1)
                val_correct += int((predictions == labels).sum().item())
                val_total += images.size(0)

        train_accuracy = train_correct / max(train_total, 1)
        val_accuracy = val_correct / max(val_total, 1)
        epoch_summary = {
            "epoch": float(epoch),
            "train_loss": train_loss / max(train_total, 1),
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        }
        history.append(epoch_summary)
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    metadata = {
        "epochs": epochs,
        "num_samples": len(samples),
        "best_val_accuracy": best_val_accuracy,
        "input_mode": resolved_input_mode,
        "history": history,
    }
    saved_path = save_model(
        model,
        output_path,
        metadata=metadata,
        input_mode=resolved_input_mode,
    )
    return {
        "model_path": str(saved_path),
        "best_val_accuracy": best_val_accuracy,
        "num_samples": len(samples),
        "input_mode": resolved_input_mode,
        "history": history,
    }


def apply_cnn_to_csv(
    csv_path: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    min_confidence: float = 0.5,
    device: str | None = None,
    overwrite_manual: bool = False,
) -> int:
    path = Path(csv_path)
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")
        fieldnames = list(reader.fieldnames)
        for column in ("visual_defect", "visual_confidence", "visual_defect_source"):
            if column not in fieldnames:
                fieldnames.append(column)
        rows.extend(reader)

    classifier = get_classifier(model_path, min_confidence=min_confidence, device=device)
    updated = 0
    for row in rows:
        source = (row.get("visual_defect_source") or "").strip().lower()
        if source == "manual" and not overwrite_manual:
            continue
        photo_path = (row.get("layer_photo") or "").strip()
        if not photo_path:
            continue
        prediction = classifier.predict(photo_path)
        if prediction is None:
            continue
        row["visual_defect"] = prediction.label
        row["visual_confidence"] = f"{prediction.confidence:.4f}"
        row["visual_defect_source"] = "cnn"
        updated += 1

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return updated


def summarize_dataset(samples: Sequence[LabeledPhoto]) -> dict[str, int]:
    counts = {label: 0 for label in CLASS_NAMES}
    for sample in samples:
        counts[sample.label] = counts.get(sample.label, 0) + 1
    return counts
