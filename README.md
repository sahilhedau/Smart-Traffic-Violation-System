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
   - Calibrated virtual stop-line geometry and traffic signal region of interest (ROI).
   - Real-time HSV color segmentation to distinguish **RED**, **YELLOW**, and **GREEN** signal states.
   - Direction-aware line crossing detection using vector cross-products (`ccw` determinant checks).
   - Generates alerts only when a vehicle crosses the stop line during an active RED signal.
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
   - Image enhancement pipeline: cubic resizing, bilateral denoising, and CLAHE (Contrast Limited Adaptive Histogram Equalization).
   - High-accuracy alphanumeric OCR via EasyOCR.
   - Formatting and validation for standard Indian registration plates (e.g., `MH12AB1234`, `DL04C1111`, `KA01MJ5050`).
   - Human-in-the-loop review interface allowing manual verification and correction of OCR strings.

6. **Traffic Control Dashboard**
   - **Executive Dashboard**: Real-time violation counters, category breakdowns, and system health status.
   - **Video Analysis**: Drag-and-drop video upload, real-time FPS/progress tracking, and direct in-browser playback of annotated output videos.
   - **Violations Review Center**: Evidence inspection modal, cropped plate viewer, and reviewer decision controls (`Confirm`, `Dismiss`, `Edit Plate Number`).
   - **Settings & Calibration**: Web-based sliders for confidence thresholds, frame skipping, stop-line coordinates, and traffic light modes.

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
- `test_signal_jump.py`: CCW orientation geometry, stop-line intersection, and HSV color detection.
- `test_triple_riding.py`: Rider-motorcycle spatial clustering and temporal thresholds.
- `test_helmet.py`: Head crop extraction and baseline texture/skin classification.
- `test_plate_ocr.py`: Plate normalization regex and CLAHE preprocessing.
- `test_api.py`: FastAPI health, stats, settings, and file upload endpoints.
- `test_video_pipeline.py`: Full end-to-end video processing and DB status update.

---

## 📂 Storage Locations

- **Database**: `traffic_system.db` (SQLite in project root)
- **Model Weights**: `models/` (e.g., `yolov8n.pt`)
- **Annotated Videos**: `outputs/annotated_videos/`
- **Violation Evidence Frames**: `outputs/evidence/`
- **Uploaded Videos**: `outputs/uploads/`
- **Demonstration Footage**: `data/samples/sample_traffic.mp4`

---

## ⚙️ Calibration & Settings

Stop line and signal ROI coordinates are normalized between `0.0` and `1.0`:
- **Stop Line**: `[x1, y1, x2, y2]` (Default: `[0.05, 0.65, 0.95, 0.65]`)
- **Signal ROI**: `[x1, y1, x2, y2]` (Default: `[0.75, 0.05, 0.95, 0.30]`)
- **Signal Modes**:
  - `auto_detect`: Automatic HSV thresholding of the light in the ROI.
  - `red`: Forced RED mode for test and verification scenarios.
  - `green`: Forced GREEN mode (no crossing violations triggered).

---

## ⚖️ Legal & Operational Disclaimer

This software generates **candidate violation alerts** subject to verification by authorized human traffic reviewers. It is intended for academic research and college demonstration purposes and must not be used for legal summons issuance without statutory validation and certified camera calibration.
