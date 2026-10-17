#!/usr/bin/env python3
"""
Specialized YOLO Training Script for Number Plate Detection
Smart Traffic Violation Detection System (TrafficGuard AI)

Features:
- Configurable base model (yolov8n.pt, yolo11n.pt, etc.)
- Auto-detects data/plate_dataset.yaml or custom YAML path
- Configurable epochs, batch size, image resolution, and device
- Automatically copies 'best.pt' weights to models/plate_yolov8n.pt
- Supports dry-run validation mode
"""

import sys
import shutil
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_plate")

def run_plate_training(
    data_yaml: str = None,
    base_model: str = "yolov8n.pt",
    epochs: int = 30,
    batch_size: int = 8,
    imgsz: int = 640,
    device: str = "auto",
    dry_run: bool = False
) -> Path:
    yaml_path = Path(data_yaml) if data_yaml else PROJECT_ROOT / "data" / "plate_dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset config {yaml_path} not found. Please run scripts/prepare_plate_dataset.py first."
        )

    logger.info("Using plate dataset configuration: %s", yaml_path)

    if dry_run:
        logger.info("[DRY RUN] Dataset configuration verified. Training skipped.")
        return yaml_path

    from ultralytics import YOLO
    import torch

    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    local_weights = PROJECT_ROOT / "models" / base_model
    weights_source = str(local_weights) if local_weights.exists() else base_model
    logger.info("Loading base weights from: %s", weights_source)

    model = YOLO(weights_source)
    run_name = "plate_yolo_train"
    project_dir = PROJECT_ROOT / "runs" / "detect"

    logger.info("Starting plate detector training: epochs=%d, batch=%d, imgsz=%d...", epochs, batch_size, imgsz)
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
        dest = PROJECT_ROOT / "models" / "plate_yolov8n.pt"
        shutil.copy2(best_weights, dest)
        logger.info("SUCCESS: Best license plate model saved to %s", dest)
        return dest
    else:
        logger.warning("Training finished but best.pt not found at %s", best_weights)
        return yaml_path

def main():
    parser = argparse.ArgumentParser(description="Train specialized YOLO model for License Plate Detection")
    parser.add_argument("--data", type=str, default=None, help="Path to plate dataset YAML")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights (e.g. yolov8n.pt, yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution size")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', or '0'")
    parser.add_argument("--dry-run", action="store_true", help="Verify config and environment without training")

    args = parser.parse_args()
    try:
        run_plate_training(
            data_yaml=args.data,
            base_model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            dry_run=args.dry_run
        )
    except Exception as e:
        logger.exception("Plate training failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
