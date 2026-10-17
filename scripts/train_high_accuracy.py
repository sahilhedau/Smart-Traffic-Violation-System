#!/usr/bin/env python3
"""
High-Accuracy YOLO Training Script for TrafficGuard AI
Smart Traffic Violation Detection System

Key Accuracy-Maximizing Techniques:
1. Architecture: YOLO11n (Enhanced C3k2 feature pyramid + SPPF)
2. Resolution: 640x640 high-resolution feature mapping
3. Cosine Annealing LR schedule (cos_lr=True) for smooth loss convergence
4. Mosaic deactivation in final epochs (close_mosaic=3) for precise bounding box regression
5. Class loss re-weighting (cls=1.2) to boost minority 'Without Helmet' detection recall
6. Post-training automated evaluation on unseen Test split with full mAP/Precision/Recall report
7. Auto-deploys best.pt to models/helmet_yolov8n.pt
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_high_accuracy")

def train_for_high_accuracy(
    data_yaml: str = "data/helmet_archive.yaml",
    base_model: str = "models/yolo11n.pt",
    epochs: int = 15,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "auto"
):
    from ultralytics import YOLO
    import torch

    yaml_path = Path(data_yaml) if Path(data_yaml).is_absolute() else PROJECT_ROOT / data_yaml
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset config {yaml_path} not found.")

    model_path = Path(base_model) if Path(base_model).is_absolute() else PROJECT_ROOT / base_model
    if not model_path.exists():
        model_path = Path("yolo11n.pt")  # fallback to auto-download

    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    logger.info("==================================================")
    logger.info("  TRAFFICGUARD AI — HIGH-ACCURACY TRAINING PIPELINE")
    logger.info("==================================================")
    logger.info("Base Model:    %s", model_path)
    logger.info("Dataset:       %s", yaml_path)
    logger.info("Epochs:        %d", epochs)
    logger.info("Image Size:    %d", imgsz)
    logger.info("Batch Size:    %d", batch_size)
    logger.info("Device:        %s", device)
    logger.info("==================================================")

    model = YOLO(str(model_path))
    project_dir = PROJECT_ROOT / "runs" / "detect"
    run_name = "helmet_high_accuracy"

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
        cos_lr=True,          # Cosine learning rate schedule
        close_mosaic=3,       # Disable mosaic in final 3 epochs for sharp boundary detection
        cls=1.2,              # Boost classification penalty for 'Without Helmet'
        box=7.5,              # Bounding box regression loss gain
        hsv_h=0.015,          # Color jitter for variable lighting
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,           # 50% horizontal flip
        verbose=True
    )

    best_weights = project_dir / run_name / "weights" / "best.pt"
    if best_weights.exists():
        dest = PROJECT_ROOT / "models" / "helmet_yolov8n.pt"
        shutil.copy2(best_weights, dest)
        logger.info("SUCCESS: Deployed high-accuracy model to %s", dest)

        # Run Test Set Evaluation
        from scripts.evaluate_model import evaluate
        logger.info("Running final evaluation on unseen Test set...")
        evaluate(
            model_path=str(dest),
            data_yaml=str(yaml_path),
            split="test",
            imgsz=imgsz,
            device=device
        )
    return best_weights

def main():
    parser = argparse.ArgumentParser(description="Train High-Accuracy Helmet Detection YOLO Model")
    parser.add_argument("--data", type=str, default="data/helmet_archive.yaml", help="Dataset config YAML")
    parser.add_argument("--model", type=str, default="models/yolo11n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="auto", help="Inference device")

    args = parser.parse_args()
    try:
        train_for_high_accuracy(
            data_yaml=args.data,
            base_model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device
        )
    except Exception as e:
        logger.exception("Training failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
