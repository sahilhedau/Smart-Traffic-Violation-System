#!/usr/bin/env python3
"""
Model Evaluation and Accuracy Benchmark Script
Smart Traffic Violation Detection System (TrafficGuard AI)

Features:
- Evaluates YOLO models on the Validation or Test split of the helmet/plate dataset
- Outputs full detection accuracy metrics:
    - Precision (P)
    - Recall (R)
    - mAP@50 (Mean Average Precision at IoU 0.50)
    - mAP@50-95 (COCO metric)
    - F1-Score
    - Per-class accuracy breakdown (With Helmet vs Without Helmet)
- Saves confusion matrix and performance curves
"""

import os
import sys
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_model")

def evaluate(
    model_path: str,
    data_yaml: str = "data/helmet_dataset.yaml",
    split: str = "test",
    imgsz: int = 640,
    device: str = "auto"
):
    from ultralytics import YOLO
    import torch

    yaml_file = PROJECT_ROOT / data_yaml if not Path(data_yaml).is_absolute() else Path(data_yaml)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Dataset config {yaml_file} not found.")

    model_file = Path(model_path)
    if not model_file.is_absolute():
        model_file = PROJECT_ROOT / model_path

    if not model_file.exists():
        # Check in models/
        if (PROJECT_ROOT / "models" / model_path).exists():
            model_file = PROJECT_ROOT / "models" / model_path
        else:
            raise FileNotFoundError(f"Model file {model_file} not found.")

    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    logger.info("Evaluating model: %s on split '%s' (device: %s)...", model_file, split, device)
    model = YOLO(str(model_file))

    metrics = model.val(
        data=str(yaml_file),
        split=split,
        imgsz=imgsz,
        device=device,
        project=str(PROJECT_ROOT / "runs" / "val"),
        name=f"eval_{split}",
        exist_ok=True,
        verbose=True
    )

    # Extract accuracy figures
    precision = metrics.box.mp  # mean precision
    recall = metrics.box.mr     # mean recall
    map50 = metrics.box.map50   # mAP @ 0.50
    map50_95 = metrics.box.map  # mAP @ 0.50:0.95
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    print("\n" + "=" * 65)
    print(f"       YOLO ACCURACY & PERFORMANCE REPORT ({split.upper()} SET)      ")
    print("=" * 65)
    print(f"Overall Metrics:")
    print(f"  * Precision (P):       {precision * 100:.2f}%")
    print(f"  * Recall (R):          {recall * 100:.2f}%")
    print(f"  * F1-Score:            {f1 * 100:.2f}%")
    print(f"  * mAP @ 0.50:          {map50 * 100:.2f}%")
    print(f"  * mAP @ 0.50:0.95:     {map50_95 * 100:.2f}%")
    print("-" * 65)

    # Per-class breakdown
    class_names = getattr(metrics, 'names', {0: 'With Helmet', 1: 'Without Helmet'})
    print(f"Per-Class Metrics:")
    for i, c_name in class_names.items():
        if i < len(metrics.box.p) and i < len(metrics.box.r) and i < len(metrics.box.all_ap):
            cp = metrics.box.p[i]
            cr = metrics.box.r[i]
            cmap50 = metrics.box.all_ap[i][0] if len(metrics.box.all_ap[i]) > 0 else 0.0
            cf1 = 2 * (cp * cr) / (cp + cr + 1e-6)
            print(f"  [{i}] {c_name:16s} | P: {cp*100:5.1f}% | R: {cr*100:5.1f}% | F1: {cf1*100:5.1f}% | mAP50: {cmap50*100:5.1f}%")

    print("=" * 65 + "\n")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO Model Accuracy on Validation or Test Set")
    parser.add_argument("--model", type=str, default="models/yolov8n.pt", help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default="data/helmet_dataset.yaml", help="Path to dataset YAML")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"], help="Split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="auto", help="Device ('cpu', '0', 'auto')")

    args = parser.parse_args()
    try:
        evaluate(
            model_path=args.model,
            data_yaml=args.data,
            split=args.split,
            imgsz=args.imgsz,
            device=args.device
        )
    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
