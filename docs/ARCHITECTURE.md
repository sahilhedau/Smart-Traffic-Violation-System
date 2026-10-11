# System Architecture & Technical Specifications

## 1. Overview

The **Smart Traffic Violation Detection System** is an end-to-end, multi-stage computer vision platform designed for real-time monitoring and automated audit of urban vehicular violations.

The system handles 4 core violation categories:
1. **Red-Light Crossing / Signal Jumping**: Vector-trajectory intersection against a calibrated virtual stop line coupled with HSV traffic signal state analysis.
2. **Triple Riding**: Detection of 3 or more riders seated on two-wheelers (motorcycles/scooters) with vertical/horizontal bounding box overlap and temporal smoothing.
3. **Helmet Violation**: Head-crop extraction, skin-to-hair texture analysis, and specialized model interface to detect motorcycle riders riding without protective headgear.
4. **License Plate Detection & OCR**: Automated candidate plate localisation, CLAHE/contrast enhancement, EasyOCR recognition, and Indian vehicle registration formatting/normalization.

---

## 2. Component Pipeline

```
+--------------------------------------------------------------------------+
|                            Video Frame Input                             |
|               (Uploaded MP4/AVI file or Live Stream/Webcam)              |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                  Ultralytics YOLOv8 Inference & Detection                |
|           Classes: person, bicycle, car, motorcycle, bus, truck           |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|               ByteTrack Multi-Object Tracking & Trajectories             |
|                  (Assigns consistent track IDs and centroids)             |
+--------------------------------------------------------------------------+
          |                         |                           |
          v                         v                           v
+-------------------+     +--------------------+     +---------------------+
|   Signal Jump     |     |   Triple Riding    |     |   Helmet Violation  |
|     Engine        |     |      Engine        |     |       Engine        |
| - HSV Color ROI   |     | - Spatial overlap  |     | - Head crop extract |
| - Stop-line cross |     | - Rider clustering |     | - Texture/skin tone |
| - Direction check |     | - Multi-frame hold |     | - Multi-frame hold  |
+-------------------+     +--------------------+     +---------------------+
          |                         |                           |
          +-------------------------+---------------------------+
                                    |
                        [ Candidate Violation? ]
                                    | (Yes)
                                    v
+--------------------------------------------------------------------------+
|               License Plate Localisation & OCR Preprocessing             |
|   1. Vehicle crop extraction (lower region)                              |
|   2. Morphological top-hat gradient & contour bounding                   |
|   3. Bilateral denoising + CLAHE contrast boost                          |
|   4. EasyOCR alphanumeric inference                                      |
|   5. Normalization regex (e.g. MH12AB1234, DL04C1111)                    |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                        Violation Event Engine                            |
|   - Deduplication (prevents multiple alerts for same tracked vehicle)    |
|   - Saves annotated evidence frame & cropped plate image                 |
|   - Assigns review status: PENDING_REVIEW                                |
|   - Commits to SQLite Database                                           |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                FastAPI REST Service & Dashboard Interface                |
|   - Real-time statistics, review workflow (Confirm/Dismiss/Edit plate)   |
|   - Annotated video playback streaming                                  |
|   - Configurable calibration (Stop line, signal ROI, privacy blur)       |
+--------------------------------------------------------------------------+
```

---

## 3. Database Schema (SQLite)

### Table: `violation_events`
- `id` (TEXT PRIMARY KEY): Unique UUID identifier.
- `event_type` (TEXT): `RED_LIGHT_CROSSING`, `TRIPLE_RIDING`, or `NO_HELMET`.
- `timestamp` (TEXT): ISO 8601 UTC timestamp.
- `camera_id` (TEXT): Camera source identifier (`CAM_MAIN_01`).
- `tracking_id` (INTEGER): ByteTrack persistent object ID.
- `confidence` (REAL): Detection confidence (0.0 to 1.0).
- `severity` (TEXT): `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`.
- `status` (TEXT): `PENDING_REVIEW`, `CONFIRMED_BY_REVIEWER`, `DISMISSED`, `UNREADABLE_PLATE`.
- `evidence_path` (TEXT): Path to full annotated evidence JPEG on disk.
- `crop_path` (TEXT): Path to cropped license plate JPEG on disk.
- `plate_text` (TEXT): Normalized registration plate text or `UNREADABLE`.
- `plate_raw_text` (TEXT): Raw unedited OCR string.
- `plate_confidence` (REAL): OCR character recognition confidence.
- `details_json` (TEXT): JSON payload containing bounding boxes and debug notes.
- `reviewer_notes` (TEXT): Notes recorded during automated detection or manual audit.
- `created_at` / `updated_at` (TEXT): Timestamps.

### Table: `processing_jobs`
- `id` (TEXT PRIMARY KEY): Job UUID.
- `source_filename` (TEXT): Path to source uploaded video.
- `status` (TEXT): `QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`.
- `progress` (REAL): Processing percentage (0 to 100).
- `current_frame` / `total_frames` (INTEGER): Progress counters.
- `fps` (REAL): Current processing throughput (Frames Per Second).
- `annotated_video_path` (TEXT): Path to output video.
- `violations_count` (INTEGER): Number of violations caught in job.
- `error_message` (TEXT): Diagnostic exception details if failed.
