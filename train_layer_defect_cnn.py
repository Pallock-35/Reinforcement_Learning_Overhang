#!/usr/bin/env python3
"""
train_layer_defect_cnn.py
=========================
Train the per-layer defect CNN from labeled webcam photos.

Examples:
  python train_layer_defect_cnn.py --photo-root print_runs --layer-labels-file layer_labels.json
  python train_layer_defect_cnn.py --photo-root print_runs --logs print_runs/print_001/log.csv
  python train_layer_defect_cnn.py --photo-root print_runs --layer-labels "1-30:normal,31-50:defective"
"""

from __future__ import annotations

import argparse
import glob

from layer_defect_cnn import (
    DEFAULT_MODEL_PATH,
    build_labeled_dataset,
    summarize_dataset,
    train_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the layer defect CNN")
    parser.add_argument(
        "--photo-root",
        default="print_runs",
        help="Root folder containing print_NNN/photos/layer_MMMM.jpg images",
    )
    parser.add_argument(
        "--logs",
        nargs="*",
        default=[],
        help="CSV logs with layer_photo and visual_defect columns",
    )
    parser.add_argument(
        "--layer-labels-file",
        help="JSON file with per-print layer label ranges",
    )
    parser.add_argument(
        "--layer-labels",
        metavar="SPEC",
        help='Label ranges applied to every photo, e.g. "1-30:normal,31-50:defective"',
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_MODEL_PATH),
        help="Where to save the trained model weights",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Training device (default: cuda when available)",
    )
    parser.add_argument(
        "--input-mode",
        choices=("luminance", "rgb"),
        default="luminance",
        help="Photo filter before CNN (default: luminance for filament-color invariance)",
    )
    args = parser.parse_args()

    log_paths: list[str] = []
    for pattern in args.logs:
        log_paths.extend(glob.glob(pattern))

    samples = build_labeled_dataset(
        args.photo_root,
        log_paths=log_paths,
        labels_file=args.layer_labels_file,
        label_spec=args.layer_labels,
    )
    if not samples:
        print(
            "No labeled photos found. Provide labels via --logs, "
            "--layer-labels-file, or --layer-labels."
        )
        return 1

    counts = summarize_dataset(samples)
    print(f"Training dataset: {len(samples)} photos")
    print(f"  normal     : {counts.get('normal', 0)}")
    print(f"  defective  : {counts.get('defective', 0)}")
    print(f"  input mode : {args.input_mode}")

    try:
        result = train_model(
            samples,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_fraction=args.val_fraction,
            device=args.device,
            input_mode=args.input_mode,  # type: ignore[arg-type]
        )
    except ValueError as error:
        print(f"Training failed: {error}")
        return 1

    print(f"Saved model to {result['model_path']}")
    print(f"Input mode: {result['input_mode']}")
    print(f"Best validation accuracy: {result['best_val_accuracy']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
