#!/usr/bin/env python3
"""
Create a representative subset YAML for fine-tuning YOLO plate detector on available hardware.
"""
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Create subset file lists for Ultralytics YOLO
downloads_dir = Path("C:/Users/sahil/Downloads")
img_train = downloads_dir / "images" / "train"
img_val = downloads_dir / "images" / "val"
lbl_train = downloads_dir / "labels" / "train"
lbl_val = downloads_dir / "labels" / "val"

# Find train images that have non-empty labels
train_files = []
for p in img_train.glob("*.jpg"):
    lbl_p = lbl_train / f"{p.stem}.txt"
    if lbl_p.exists() and lbl_p.stat().st_size > 0:
        train_files.append(p.as_posix())

val_files = []
for p in img_val.glob("*.jpg"):
    lbl_p = lbl_val / f"{p.stem}.txt"
    if lbl_p.exists() and lbl_p.stat().st_size > 0:
        val_files.append(p.as_posix())

random.seed(42)
random.shuffle(train_files)
random.shuffle(val_files)

# Select 600 train and 150 val
selected_train = train_files[:600]
selected_val = val_files[:150]

subset_dir = PROJECT_ROOT / "data" / "plate_subset"
subset_dir.mkdir(parents=True, exist_ok=True)

train_list_txt = subset_dir / "train_subset.txt"
val_list_txt = subset_dir / "val_subset.txt"

with open(train_list_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(selected_train) + "\n")

with open(val_list_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(selected_val) + "\n")

yaml_content = f"""# Curated Plate Detection Subset for Fast, Accurate Fine-Tuning
path: {PROJECT_ROOT.as_posix()}
train: {train_list_txt.as_posix()}
val: {val_list_txt.as_posix()}
test: C:/Users/sahil/Downloads/images/test

nc: 1
names: ['license_plate']
"""

yaml_path = PROJECT_ROOT / "data" / "plate_subset.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"Created subset YAML at {yaml_path}: {len(selected_train)} train images, {len(selected_val)} val images.")
