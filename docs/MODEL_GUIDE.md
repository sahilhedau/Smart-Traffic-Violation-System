# Model Management and Training Guide

## 1. Deployed Models & Pretrained Weights

| Model | Purpose | Architecture | Source / License | Size |
|---|---|---|---|---|
| `yolov8n.pt` | Object Detection & Tracking (Person, Vehicles) | YOLOv8 Nano (CNN + C2f) | Ultralytics / AGPL-3.0 | ~6.2 MB |
| `bytetrack.yaml` | Multi-Object Temporal Tracking | Kalman Filter + Hungarian Matching | Ultralytics Built-in | Included |
| `EasyOCR Reader` | Character Recognition on Plates | CRAFT + ResNet-BiLSTM-CTC | JaidedAI / Apache-2.0 | ~80 MB (cached) |
| `helmet_yolov8n.pt` (Optional) | Specialized Helmet Detection | YOLOv8 Nano Custom | Custom Fine-Tuned | ~6 MB |

---

## 2. Model Download & Verification Process

The project includes an automated, self-verifying downloader:
```bash
python scripts/download_models.py
```
- Downloads only verified release binaries directly from official repository mirrors.
- Downloads to a temporary `.tmp` file and validates minimum byte size before atomically replacing the destination weight file.
- Idempotent: Skips download if valid weight file already exists on disk.

---

## 3. Specialized Fine-Tuning Guide

While the system provides computer-vision baselines for helmet inspection and license plate recognition out of the box, accuracy can be customized with domain-specific datasets.

### A. Helmet & Rider Classification
1. **Dataset Sources**:
   - Roboflow Two-Wheeler Helmet Detection Dataset (public permissive domain).
   - AI City Challenge Traffic Safety Benchmark.
2. **Annotation Classes**: `helmet`, `without_helmet`.
3. **Training Command**:
   ```bash
   yolo detect train data=helmet_dataset.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=16
   ```
4. **Deployment**:
   Place the exported `best.pt` inside the `models/` folder as `models/helmet_yolov8n.pt`. The application automatically detects and loads it upon startup.

### B. Indian License Plate Localisation
1. **Dataset Sources**:
   - Open Datasets India Number Plate Collection (Kaggle / CC-BY-4.0).
2. **Annotation Classes**: `license_plate`.
3. **Training Command**:
   ```bash
   yolo detect train data=plate_dataset.yaml model=yolov8n.pt epochs=40 imgsz=640
   ```
4. **Deployment**:
   Place the trained model in `models/plate_yolov8n.pt`.

---

## 4. Hardware and Inference Device

The system automatically detects hardware acceleration:
- **NVIDIA GPU / CUDA**: Detected via `torch.cuda.is_available()`. Inference automatically runs on the GPU.
- **CPU Fallback**: When no compatible GPU is present, execution seamlessly routes to CPU with frame sampling (`frame_skip=2` or `3`) to ensure high throughput on standard laptops.
