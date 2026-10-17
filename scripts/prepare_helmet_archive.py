#!/usr/bin/env python3
"""
Pascal VOC XML to YOLO Conversion Script for Helmet Archive Dataset
Smart Traffic Violation Detection System (TrafficGuard AI)

Features:
- Converts 764 Pascal VOC XML annotations into normalized YOLO txt format
- Maps classes: 0 -> 'With Helmet', 1 -> 'Without Helmet'
- Splits dataset cleanly into Train (80%), Val (10%), Test (10%) with fixed seed (no leakage)
- Generates data/helmet_archive.yaml with absolute/portable paths
- Verifies bounding box coordinates and image dimensions
"""

import os
import sys
import random
import shutil
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_helmet_archive")

CLASS_MAP = {
    "With Helmet": 0,
    "Without Helmet": 1,
}

def parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[Tuple[int, float, float, float, float]]]:
    """Parses VOC XML file and returns (width, height, list of [cls_id, xc, yc, w, h])."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is not None:
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)
    else:
        raise ValueError(f"No size tag in {xml_path}")

    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid dimensions {img_w}x{img_h} in {xml_path}")

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_MAP:
            logger.warning("Unknown class '%s' in %s, skipping object.", name, xml_path.name)
            continue
        cls_id = CLASS_MAP[name]

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Normalize to YOLO format
        xc = ((xmin + xmax) / 2.0) / img_w
        yc = ((ymin + ymax) / 2.0) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        # Clamp to [0, 1]
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        bw = max(0.001, min(1.0, bw))
        bh = max(0.001, min(1.0, bh))

        boxes.append((cls_id, xc, yc, bw, bh))

    return img_w, img_h, boxes

def convert_and_split(
    archive_dir: Path,
    output_dir: Path = None,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42
) -> Path:
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "helmet_archive_yolo"

    archive_dir = Path(archive_dir).resolve()
    output_dir = Path(output_dir).resolve()

    xml_dir = archive_dir / "annotations"
    img_dir = archive_dir / "images"

    if not xml_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(f"Missing annotations or images in {archive_dir}")

    xml_files = sorted(list(xml_dir.glob("*.xml")))
    logger.info("Found %d XML annotation files in %s", len(xml_files), xml_dir)

    # Collect valid matched image-xml pairs
    pairs = []
    for xf in xml_files:
        stem = xf.stem
        # Try .png, .jpg, .jpeg
        matched_img = None
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
            cand = img_dir / f"{stem}{ext}"
            if cand.exists():
                matched_img = cand
                break
        if matched_img:
            pairs.append((matched_img, xf))

    logger.info("Matched %d image-annotation pairs.", len(pairs))

    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:]
    }

    class_stats = {s: {0: 0, 1: 0} for s in splits}

    for split_name, items in splits.items():
        s_img_dir = output_dir / split_name / "images"
        s_lbl_dir = output_dir / split_name / "labels"
        s_img_dir.mkdir(parents=True, exist_ok=True)
        s_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, xml_path in items:
            # Copy image
            shutil.copy2(img_path, s_img_dir / img_path.name)

            # Convert XML to YOLO txt
            try:
                _, _, boxes = parse_voc_xml(xml_path)
                lbl_file = s_lbl_dir / f"{img_path.stem}.txt"
                with open(lbl_file, "w", encoding="utf-8") as f:
                    for cls_id, xc, yc, bw, bh in boxes:
                        f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                        class_stats[split_name][cls_id] += 1
            except Exception as e:
                logger.error("Error processing %s: %s", xml_path.name, e)

    # Write data/helmet_archive.yaml
    yaml_path = PROJECT_ROOT / "data" / "helmet_archive.yaml"
    yaml_content = f"""# Auto-generated dataset config for Helmet Archive Dataset (764 raw samples)
path: {output_dir.as_posix()}
train: {(output_dir / "train" / "images").as_posix()}
val: {(output_dir / "val" / "images").as_posix()}
test: {(output_dir / "test" / "images").as_posix()}

nc: 2
names: ['With Helmet', 'Without Helmet']
"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    logger.info("=== Dataset Conversion & Split Complete ===")
    for s, counts in class_stats.items():
        logger.info("  * Split '%s': %d images | With Helmet: %d | Without Helmet: %d",
                    s, len(splits[s]), counts[0], counts[1])
    logger.info("Config saved to: %s", yaml_path)
    return yaml_path

if __name__ == "__main__":
    archive_path = Path("C:/Users/sahil/Downloads/archive")
    convert_and_split(archive_path)
