# Smart Traffic Violation Detection System (TrafficGuard AI)

A complete, production-grade, autonomous Computer Vision & AI system built to detect, track, and audit road traffic violations from surveillance video recordings and camera feeds.

Developed with **Ultralytics YOLOv8**, **ByteTrack**, **OpenCV**, **EasyOCR**, and **FastAPI**, featuring a responsive real-time web dashboard for law-enforcement and traffic management auditing.

---

## 🚦 Key Features

1. **General Object Detection & Vehicle Tracking**
   - Detects and tracks vehicles (`car`, `motorcycle`, `bus`, `truck`, `bicycle`) and `person` instances.
   - Powered by ByteTrack multi-object tracking to assign persistent tracking IDs across video frames.
   - Generates fully annotated output videos with bounding boxes, IDs, and trajectory paths.

2. **Red-Light / Signal-Jump Violation Detection**
   - Multiple calibrated traffic-signal regions, each paired with its own virtual stop line.
   - Real-time HSV color segmentation to distinguish **RED**, **YELLOW**, and **GREEN** signal states.
   - Direction-aware line crossing detection using vector cross-products (`ccw` determinant checks).
   - Generates alerts only when a vehicle crosses a stop line paired with an active RED signal.
   - Eliminates duplicate alerts for the same crossing vehicle.

3. **Triple-Riding Violation Detection**
   - Detects two-wheelers and associates riders using vertical alignment, horizontal clustering, and bounding box overlap.
   - Flags an alert when **three or more riders** are seated on a motorcycle.
   - Temporal smoothing over consecutive frames eliminates momentary detector flickers.

4. **Helmet & No-Helmet Violation Detection**
   - Automatic rider head-region localization (upper 25% crop).
   - Computer-vision analysis (skin tone ratio in HSV and Laplacian texture entropy) and dedicated model hook.
   - Temporal consistency smoothing across multiple frames before raising candidate violations.

5. **License Plate Localisation & OCR Pipeline**
   - Triggered selectively on candidate violation frames to optimize throughput.
   - Morphological top-hat gradient analysis to crop vehicle number plate candidate regions.
   - Multi-scale OCR inputs use cubic resizing, bilateral denoising, CLAHE, and adaptive thresholding.
   - EasyOCR results are compared across preprocessing variants, preferring repeated reads and Indian registration formats.
   - Formatting and validation for standard Indian registration plates (e.g., `MH12AB1234`, `DL04C1111`, `KA01MJ5050`).
   - Human-in-the-loop review interface allowing manual verification and correction of OCR strings.

6. **Traffic Control Dashboard**
   - **Executive Dashboard**: Real-time violation counters, category breakdowns, and system health status.
   - **Video Analysis**: Drag-and-drop video upload, real-time FPS/progress tracking, and direct in-browser playback of annotated output videos.
   - **Violations Review Center**: Evidence inspection modal, cropped plate viewer, and reviewer decision controls (`Confirm`, `Dismiss`, `Edit Plate Number`).
   - **Settings & Calibration**: Web-based sliders for confidence thresholds, frame skipping, and paired stop-line/signal regions.

---

## 💻 Tech Stack & Environment

- **Language**: Python 3.13 / 3.14 compatible
- **Deep Learning**: PyTorch 2.14+, Torchvision 0.29+, Ultralytics YOLOv8
- **Tracking**: ByteTrack (`lap` assignment + Kalman Filter)
- **Computer Vision**: OpenCV 5.0 (Canny, morphological transforms, CLAHE, HSV masks)
- **OCR Engine**: EasyOCR (CRAFT + ResNet-BiLSTM)
- **Backend**: FastAPI, Uvicorn, SQLite (WAL mode, parameterized queries)
- **Frontend**: Responsive Single-Page Application (HTML5, Modern CSS, Vanilla JS)
- **Automated Testing**: Pytest, Pytest-Asyncio, HTTPX

---

## ⚡ Quick Start (Windows)

### 1. One-Click Startup
Double-click or run the startup batch script in your terminal:
```cmd
run.bat
```
This script will:
- Activate the local Python virtual environment (`.venv`).
- Verify model weights (`models/yolov8n.pt`).
- Initialize the SQLite database and output directories.
- Launch the FastAPI server at `http://127.0.0.1:8000`.

### 2. Open the Dashboard
Navigate to:
```
http://127.0.0.1:8000
```
Interactive API documentation is accessible at:
```
http://127.0.0.1:8000/docs
```

### 3. Upload, Calibrate, and Start

In **Video Analysis**, upload a traffic video. Uploading stages the file without running violation analysis and opens **Settings & Calibration** with the same video. The app suggests a stop line using image-edge detection and signal boxes using the pretrained YOLO traffic-light class. Review those suggestions, drag line endpoints or signal boxes on the preview, or use the X/Y and box-edge sliders. Each approach has its own signal box and paired stop line; adding or removing an approach updates both together. Calibration changes save automatically. Return to Video Analysis and press **Start Analysis** when ready. Signal-state classification uses the saved regions and HSV color detection.

The general-purpose YOLO weights are not custom-trained for a specific camera. Reliable stop-line detection depends on the scene, so its computer-vision suggestion should be checked and adjusted before processing. Multi-pass OCR improves resilience to small or low-contrast crops, but no accuracy increase is claimed without a representative labeled benchmark.

---

## 🛠️ Manual Installation & Setup

If you prefer setting up the environment step-by-step:

```powershell
# 1. Create Python virtual environment
py -3.13 -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\activate

# 3. Upgrade pip and install dependencies
pip install -r requirements.txt

# 4. Download pretrained models
python scripts/download_models.py

# 5. Generate sample demonstration video (Optional)
python scripts/generate_sample_video.py

# 6. Run automated test suite
pytest -v

# 7. Start the backend server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

---

## 🧪 Automated Testing

The project includes an automated test suite covering configuration, database operations, rule engines, OCR normalization, and API endpoints:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

**Test Coverage Summary:**
- `test_config.py`: Directory and setting verification.
- `test_database.py`: Schema migration and settings persistence.
- `test_event_service.py`: Event CRUD, deduplication windows, and review updates.
- `test_signal_jump.py`: CCW geometry, paired signal/stop-line crossings, and HSV color detection.
- `test_triple_riding.py`: Rider-motorcycle spatial clustering and temporal thresholds.
- `test_helmet.py`: Head crop extraction and baseline texture/skin classification.
- `test_plate_ocr.py`: Plate normalization, preprocessing, and multi-pass OCR candidate selection.
- `test_api.py`: FastAPI health, stats, settings, and file upload endpoints.
- `test_video_pipeline.py`: Full end-to-end video processing and DB status update.

- `test_helmet_dataset.py`: Helmet dataset detection, annotation validation & YAML config.
- `test_image_api.py`: Snapshot image upload, YOLO detection, OCR & evidence endpoints.
- `test_plate_engine.py`: Plate recognition engine, verification rules & multi-vehicle OCR.

---

## 🎯 Custom YOLO Training & Datasets

### 1. Helmet & No-Helmet Detection
Your Roboflow/YOLO dataset is automatically recognized. Run:
```powershell
# Validate dataset and generate data/helmet_dataset.yaml (Dry run)
.\.venv\Scripts\python.exe scripts/train_helmet.py --dry-run

# Start local YOLO training (configurable epochs, batch, imgsz)
.\.venv\Scripts\python.exe scripts/train_helmet.py --epochs 25 --batch 8 --imgsz 640
```
Trained weights (`best.pt`) are automatically copied to `models/helmet_yolov8n.pt` for live system inference!

### 2. License Plate Detection
Prepare, convert (VOC XML / COCO JSON / YOLO format), and train:
```powershell
# Prepare and split plate dataset into data/plate_dataset/
.\.venv\Scripts\python.exe scripts/prepare_plate_dataset.py --source <path_to_raw_dataset>

# Train specialized license plate detector
.\.venv\Scripts\python.exe scripts/train_plate.py --epochs 30 --batch 8
```
Trained weights are automatically saved to `models/plate_yolov8n.pt`.

### 3. Free Google Colab GPU Training (10–15 min)
See [`scripts/colab_training_guide.md`](file:///d:/Smart-Traffic-Violation-System/scripts/colab_training_guide.md) for ready-to-run Colab code with a free NVIDIA Tesla T4 GPU.

---

## 📸 Snapshot / Image Detection API

Upload any traffic photograph for instant analysis:
- **API Endpoint**: `POST /api/image/detect`
- **Supported Formats**: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`
- **Returns**: Bounding boxes, vehicle classifications, helmet compliance (With / Without Helmet), license plate OCR text, and traffic violations.
- **Web UI**: Access the **Snapshot Detection** tab in the dashboard at `http://127.0.0.1:8000`.

---

## 📂 Storage Locations

- **Database**: `traffic_system.db` (SQLite in project root)
- **Model Weights**: `models/` (`yolov8n.pt`, `helmet_yolov8n.pt`, `plate_yolov8n.pt`)
- **Annotated Videos**: `outputs/annotated_videos/`
- **Violation Evidence Frames**: `outputs/evidence/`
- **Uploaded Videos**: `outputs/uploads/`
- **Demonstration Footage**: `data/samples/sample_traffic.mp4`

---

## ⚙️ Calibration & Settings

- **Stop Line**: `[x1, y1, x2, y2]`. In **Settings & Calibration**, load a representative camera image/video, select a frame, and click the two ends of the painted stop bar for each approach.
- **Signal Area**: `[x1, y1, x2, y2]`. Drag a tight box around the signal lens, avoiding the housing and background; color is read from this area and YOLO traffic-light boxes are constrained to it. Each signal area is paired with its own stop line.
- **Signal Modes**:
  - `auto_detect` (default): Detects red/yellow/green from traffic-light boxes or the calibrated signal area.
  - `red`: Forced RED mode for test and verification scenarios.
  - `green`: Forced GREEN mode (no crossing violations triggered).

Calibration is saved per configured approach. The stop-line endpoints should span the stop bar for the lane controlled by its paired signal; verify each line/box pair against the camera view before processing.

---

## ⚖️ Legal & Operational Disclaimer

This software generates **candidate violation alerts** subject to verification by authorized human traffic reviewers. It is intended for academic research and college demonstration purposes and must not be used for legal summons issuance without statutory validation and certified camera calibration.
