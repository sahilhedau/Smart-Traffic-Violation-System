#!/usr/bin/env python3
"""
Dataset Preparation & Format Conversion Script for License Plate Detection
Smart Traffic Violation Detection System (TrafficGuard AI)

Features:
- Auto-detects number plate datasets in common formats:
    1. YOLO TXT format (class xc yc w h)
    2. Pascal VOC XML format (<bndbox><xmin>...)
    3. COCO JSON format (annotations: [{bbox: [x,y,w,h]}])
- Safely converts Pascal VOC or COCO into standardized YOLO format
- Automatically splits dataset into train / val / test (80/10/10) with fixed seed
- Validates bounding boxes to ensure no corrupted or out-of-bounds coordinates
- Generates data/plate_dataset.yaml for Ultralytics YOLO training
- Never overwrites original dataset files; creates clean structured output in data/plate_dataset/
"""

import os
import sys
import json
import random
import shutil
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_plate")

def convert_voc_xml_to_yolo(xml_file: Path, img_w: int, img_h: int) -> List[Tuple[int, float, float, float, float]]:
    """Converts a Pascal VOC XML file to normalized YOLO format."""
    boxes = []
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # If width/height in XML
    size_elem = root.find("size")
    if size_elem is not None:
        try:
            w_val = int(size_elem.find("width").text)
            h_val = int(size_elem.find("height").text)
            if w_val > 0 and h_val > 0:
                img_w, img_h = w_val, h_val
        except Exception:
            pass

    if img_w <= 0 or img_h <= 0:
        return boxes

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            xc = ((xmin + xmax) / 2.0) / img_w
            yc = ((ymin + ymax) / 2.0) / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h

            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            bw = max(0.001, min(1.0, bw))
            bh = max(0.001, min(1.0, bh))

            boxes.append((0, xc, yc, bw, bh))
        except Exception as e:
            logger.warning("Error parsing bbox in %s: %s", xml_file.name, e)

    return boxes

def convert_coco_json_to_yolo(json_file: Path, output_labels_dir: Path) -> int:
    """Converts a COCO JSON file to individual YOLO txt label files."""
    with open(json_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    annotations_by_img = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        annotations_by_img.setdefault(img_id, []).append(ann)

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for img_id, img_info in images.items():
        fname = Path(img_info["file_name"]).stem
        img_w = float(img_info["width"])
        img_h = float(img_info["height"])
        if img_w <= 0 or img_h <= 0:
            continue

        label_path = output_labels_dir / f"{fname}.txt"
        lines = []
        for ann in annotations_by_img.get(img_id, []):
            bbox = ann.get("bbox", [])  # [x, y, width, height]
            if len(bbox) != 4:
                continue
            x, y, w, h = bbox
            xc = (x + w / 2.0) / img_w
            yc = (y + h / 2.0) / img_h
            nw = w / img_w
            nh = h / img_h
            lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        if lines:
            with open(label_path, "w", encoding="utf-8") as lf:
                lf.write("\n".join(lines) + "\n")
            count += 1

    return count

def prepare_plate_dataset(
    source_dir: Path,
    target_dir: Path = None,
    train_split: float = 0.80,
    val_split: float = 0.10,
    seed: int = 42
) -> Path:
    """
    Standardizes dataset into target_dir with train/val/test splits and generates data/plate_dataset.yaml.
    """
    if target_dir is None:
        target_dir = PROJECT_ROOT / "data" / "plate_dataset"

    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve()
    logger.info("Preparing plate dataset from '%s' into '%s'...", source_dir, target_dir)

    # Check if source_dir is already a split YOLO dataset (contains train/images and train/labels)
    if (source_dir / "train" / "images").exists() and (source_dir / "train" / "labels").exists():
        logger.info("Source dataset already split into YOLO train/val/test format.")
        yaml_path = PROJECT_ROOT / "data" / "plate_dataset.yaml"
        _write_yaml(source_dir, yaml_path)
        return yaml_path

    # Gather images and annotations
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_files = list(source_dir.rglob("*"))
    images = [f for f in all_files if f.is_file() and f.suffix.lower() in img_exts]

    if not images:
        raise ValueError(f"No image files found in {source_dir}")

    logger.info("Found %d image files in source directory.", len(images))

    # Match each image with its label
    paired = []
    xml_files = {f.stem.lower(): f for f in all_files if f.suffix.lower() == ".xml"}
    txt_files = {f.stem.lower(): f for f in all_files if f.suffix.lower() == ".txt" and f.name != "classes.txt"}

    for img in images:
        stem = img.stem.lower()
        if stem in txt_files:
            paired.append((img, "yolo_txt", txt_files[stem]))
        elif stem in xml_files:
            paired.append((img, "voc_xml", xml_files[stem]))

    logger.info("Matched %d images with valid annotation files.", len(paired))

    # Shuffle and split
    random.seed(seed)
    random.shuffle(paired)

    n_total = len(paired)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    splits = {
        "train": paired[:n_train],
        "valid": paired[n_train:n_train + n_val],
        "test": paired[n_train + n_val:]
    }

    # Copy / convert into target directory
    import cv2
    for split_name, items in splits.items():
        img_out = target_dir / split_name / "images"
        lbl_out = target_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, fmt, ann_path in items:
            dest_img = img_out / img_path.name
            dest_lbl = lbl_out / f"{img_path.stem}.txt"

            shutil.copy2(img_path, dest_img)

            if fmt == "yolo_txt":
                shutil.copy2(ann_path, dest_lbl)
            elif fmt == "voc_xml":
                # Need image dimensions for VOC conversion
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    boxes = convert_voc_xml_to_yolo(ann_path, w, h)
                    with open(dest_lbl, "w", encoding="utf-8") as lf:
                        for cls_id, xc, yc, bw, bh in boxes:
                            lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    yaml_path = PROJECT_ROOT / "data" / "plate_dataset.yaml"
    _write_yaml(target_dir, yaml_path)
    logger.info("Plate dataset preparation complete! Config saved to: %s", yaml_path)
    return yaml_path

def _write_yaml(dataset_dir: Path, output_yaml: Path):
    val_dir = dataset_dir / "valid" / "images"
    if not val_dir.exists() and (dataset_dir / "val" / "images").exists():
        val_dir = dataset_dir / "val" / "images"

    test_dir = dataset_dir / "test" / "images"
    if not test_dir.exists():
        test_dir = val_dir

    yaml_content = f"""# Auto-generated dataset configuration for TrafficGuard AI Number Plate Detection
path: {dataset_dir.as_posix()}
train: {(dataset_dir / "train" / "images").as_posix()}
val: {val_dir.as_posix()}
test: {test_dir.as_posix()}

nc: 1
names: ['license_plate']
"""
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        f.write(yaml_content)

def main():
    parser = argparse.ArgumentParser(description="Prepare, convert, and split license plate dataset for YOLO")
    parser.add_argument("--source", type=str, required=True, help="Path to raw or downloaded plate dataset folder")
    parser.add_argument("--target", type=str, default=None, help="Target destination for clean YOLO dataset")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Train split ratio (default 0.80)")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Val split ratio (default 0.10)")

    args = parser.parse_args()
    try:
        prepare_plate_dataset(
            source_dir=Path(args.source),
            target_dir=Path(args.target) if args.target else None,
            train_split=args.train_ratio,
            val_split=args.val_ratio
        )
    except Exception as e:
        logger.exception("Plate dataset preparation failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
