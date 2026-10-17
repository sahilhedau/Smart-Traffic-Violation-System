#!/usr/bin/env python3
"""
Specialized YOLO Training Script for Helmet & No-Helmet Detection
Smart Traffic Violation Detection System (TrafficGuard AI)

Features:
- Auto-detects local dataset in Downloads or project data directory
- Validates image and label files (YOLO format: class x y w h)
- Generates data/helmet_dataset.yaml with verified paths
- Configurable model (yolov8n, yolo11n, etc.), epochs, batch size, imgsz, and device
- Automatically copies trained weights 'best.pt' to models/helmet_yolov8n.pt
- Supports dry-run validation mode for instant environment verification
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_helmet")

DEFAULT_CANDIDATE_PATHS = [
    PROJECT_ROOT / "data" / "helmet_dataset",
    Path("C:/Users/sahil/Downloads/Bike-Helmets-Detection.v1i.yolov5pytorch"),
    Path.home() / "Downloads" / "Bike-Helmets-Detection.v1i.yolov5pytorch",
]

def find_dataset(custom_path: str = None) -> Path:
    """Finds the helmet dataset directory from custom argument or known candidate paths."""
    if custom_path:
        p = Path(custom_path).resolve()
        if p.exists() and (p / "train").exists():
            return p
        raise FileNotFoundError(f"Specified dataset path does not exist or missing 'train': {custom_path}")

    for candidate in DEFAULT_CANDIDATE_PATHS:
        if candidate.exists() and (candidate / "train").exists():
            logger.info("Found helmet dataset at: %s", candidate)
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not automatically locate helmet dataset. Please provide --dataset-path <path_to_dataset>"
    )

def validate_split(split_name: str, images_dir: Path, labels_dir: Path) -> Dict[str, Any]:
    """Validates images and YOLO label files for a single split."""
    if not images_dir.exists():
        return {"count": 0, "labels": 0, "class_counts": {}, "errors": [f"Missing {split_name}/images"]}

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [f for f in images_dir.iterdir() if f.suffix.lower() in image_exts]
    class_counts = {0: 0, 1: 0}
    errors = []

    valid_labels = 0
    for img in images:
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        valid_labels += 1
        try:
            with open(lbl, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        errors.append(f"{lbl.name}:{line_num} Invalid token count ({len(parts)} != 5)")
                        continue
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        errors.append(f"{lbl.name}:{line_num} Coordinates out of bounds: {parts}")
                        continue
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
        except Exception as e:
            errors.append(f"Error reading {lbl.name}: {e}")

    return {
        "count": len(images),
        "labels": valid_labels,
        "class_counts": class_counts,
        "errors": errors[:10]  # sample first 10 errors
    }

def generate_yaml(dataset_dir: Path, output_yaml: Path) -> Path:
    """Generates a standalone YAML configuration file with absolute paths."""
    train_images = dataset_dir / "train" / "images"
    val_images = dataset_dir / "valid" / "images"
    test_images = dataset_dir / "test" / "images"

    if not val_images.exists() and (dataset_dir / "val" / "images").exists():
        val_images = dataset_dir / "val" / "images"

    yaml_content = f"""# Auto-generated dataset configuration for TrafficGuard AI Helmet Detection
path: {dataset_dir.as_posix()}
train: {train_images.as_posix()}
val: {val_images.as_posix()}
test: {test_images.as_posix() if test_images.exists() else val_images.as_posix()}

nc: 2
names: ['With Helmet', 'Without Helmet']
"""
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    logger.info("Generated YOLO dataset config: %s", output_yaml)
    return output_yaml

def run_training(
    dataset_path: str = None,
    base_model: str = "yolov8n.pt",
    epochs: int = 25,
    batch_size: int = 8,
    imgsz: int = 640,
    device: str = "auto",
    dry_run: bool = False
) -> Path:
    """Main execution function for dataset validation and training."""
    dataset_dir = find_dataset(dataset_path)
    logger.info("Validating dataset structure in: %s", dataset_dir)

    train_stats = validate_split("train", dataset_dir / "train" / "images", dataset_dir / "train" / "labels")
    val_stats = validate_split("valid", dataset_dir / "valid" / "images", dataset_dir / "valid" / "labels")

    logger.info("--- Helmet Dataset Summary ---")
    logger.info("Train: %d images, %d label files", train_stats["count"], train_stats["labels"])
    logger.info("Train class annotations: 'With Helmet'=%d, 'Without Helmet'=%d",
                train_stats["class_counts"].get(0, 0), train_stats["class_counts"].get(1, 0))
    logger.info("Validation: %d images, %d label files", val_stats["count"], val_stats["labels"])

    if train_stats["errors"]:
        logger.warning("Found %d annotation warnings in train split", len(train_stats["errors"]))

    yaml_path = PROJECT_ROOT / "data" / "helmet_dataset.yaml"
    generate_yaml(dataset_dir, yaml_path)

    if dry_run:
        logger.info("[DRY RUN] Dataset validation passed. Training skipped.")
        return yaml_path

    # Proceed to train with Ultralytics YOLO
    from ultralytics import YOLO

    # Resolve device
    if device == "auto":
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
    logger.info("Using inference/training device: %s", device)

    # Resolve base model weights
    local_weights = PROJECT_ROOT / "models" / base_model
    weights_source = str(local_weights) if local_weights.exists() else base_model
    logger.info("Initializing base YOLO model from: %s", weights_source)

    model = YOLO(weights_source)
    run_name = "helmet_yolo_train"
    project_dir = PROJECT_ROOT / "runs" / "detect"

    logger.info("Starting training: epochs=%d, batch=%d, imgsz=%d, device=%s...", epochs, batch_size, imgsz, device)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        workers=2,
        verbose=True
    )

    best_weights = project_dir / run_name / "weights" / "best.pt"
    if best_weights.exists():
        dest = PROJECT_ROOT / "models" / "helmet_yolov8n.pt"
        shutil.copy2(best_weights, dest)
        logger.info("SUCCESS: Best model weights saved to %s", dest)
        return dest
    else:
        logger.warning("Training completed but best.pt not found at %s", best_weights)
        return yaml_path

def main():
    parser = argparse.ArgumentParser(description="Train specialized YOLO model for Helmet Detection")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to Roboflow/YOLO helmet dataset")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights (e.g. yolov8n.pt, yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (reduce to 4 or 8 for CPU)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution size (e.g. 640 or 416)")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', or GPU index '0'")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and generate YAML without training")

    args = parser.parse_args()
    try:
        run_training(
            dataset_path=args.dataset_path,
            base_model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            dry_run=args.dry_run
        )
    except Exception as e:
        logger.exception("Failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
