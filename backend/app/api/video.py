import os
import uuid
import shutil
import logging
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from backend.app.config import config
from backend.app.database.db import get_db_connection
from backend.app.schemas.video import JobStatusResponse, VideoUploadResponse, JobStatus
from ai.detection.detector import ObjectDetector
from ai.video.processor import VideoProcessor

logger = logging.getLogger("traffic_system.api.video")
router = APIRouter(prefix="/video", tags=["video"])

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
_calibration_detector: Optional[ObjectDetector] = None


def suggest_stop_line(frame: np.ndarray) -> Optional[list[float]]:
    """Suggest a likely road stop bar from long, near-horizontal edges."""
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(20, width // 12),
        minLineLength=max(30, width // 8),
        maxLineGap=max(10, width // 50),
    )
    if segments is None:
        return None

    candidates = []
    for segment in segments.reshape(-1, 4):
        x1, y1, x2, y2 = map(int, segment)
        delta_x, delta_y = x2 - x1, y2 - y1
        length = math.hypot(delta_x, delta_y)
        midpoint_y = (y1 + y2) / 2
        if length < width * 0.12 or not height * 0.35 <= midpoint_y <= height * 0.92:
            continue
        if abs(delta_y) > max(12, abs(delta_x) * 0.25):
            continue
        candidates.append((length * (0.5 + midpoint_y / height), x1, y1, x2, y2))

    if not candidates:
        return None

    _, x1, y1, x2, y2 = max(candidates)
    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    return [x1 / width, y1 / height, x2 / width, y2 / height]


@router.get("/calibrate/{job_id}")
def suggest_video_calibration(job_id: str):
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT source_filename FROM processing_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        source_path = Path(row["source_filename"])
        if not source_path.is_file():
            raise HTTPException(status_code=404, detail="Source video file missing on disk")
    finally:
        conn.close()

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise HTTPException(status_code=422, detail="Could not open the uploaded video for calibration")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 4)
        success, frame = capture.read()
        if not success:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = capture.read()
        if not success or frame is None:
            raise HTTPException(status_code=422, detail="Could not read a frame from the uploaded video")
        frame_time_seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
    finally:
        capture.release()

    global _calibration_detector
    if _calibration_detector is None:
        _calibration_detector = ObjectDetector(
            model_name=str(config.MODELS_DIR / config.YOLO_MODEL_NAME),
            device=config.DEVICE,
            conf_threshold=0.15,
        )

    detections = _calibration_detector.detect_and_track(frame, persist=False)
    height, width = frame.shape[:2]
    signal_rois = [
        [
            max(0.0, min(1.0, detection.box[0] / width)),
            max(0.0, min(1.0, detection.box[1] / height)),
            max(0.0, min(1.0, detection.box[2] / width)),
            max(0.0, min(1.0, detection.box[3] / height)),
        ]
        for detection in detections
        if detection.class_name == "traffic light"
    ]
    signal_rois.sort(key=lambda roi: (roi[2] - roi[0]) * (roi[3] - roi[1]), reverse=True)

    return {
        "width": width,
        "height": height,
        "frame_time_seconds": frame_time_seconds,
        "stop_line": suggest_stop_line(frame),
        "signal_rois": signal_rois,
        "signal_model": config.YOLO_MODEL_NAME,
    }

def run_video_job(job_id: str, video_path: str, is_demo: bool = False):
    try:
        processor = VideoProcessor(job_id=job_id, video_source=video_path, is_demo=is_demo)
        processor.process()
    except Exception as e:
        logger.exception("Background video processing failed for job %s: %s", job_id, e)

@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    job_id = str(uuid.uuid4())
    safe_filename = f"{job_id[:8]}_{Path(file.filename).name}"
    target_path = config.UPLOADS_DIR / safe_filename

    # Save uploaded file safely
    bytes_read = 0
    with open(target_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            bytes_read += len(chunk)
            if bytes_read > MAX_FILE_SIZE:
                target_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File exceeds maximum allowed size of 500 MB")
            buffer.write(chunk)

    # Insert job into database
    conn = get_db_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        with conn:
            conn.execute("""
            INSERT INTO processing_jobs (
                id, source_filename, status, progress, current_frame, total_frames,
                fps, annotated_video_path, violations_count, error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id, str(target_path), JobStatus.QUEUED.value, 0.0, 0, 0,
                0.0, None, 0, None, now, now
            ))
    finally:
        conn.close()

    return VideoUploadResponse(
        job_id=job_id,
        filename=file.filename,
        file_size_bytes=bytes_read,
        message="Video uploaded successfully and queued for processing."
    )

@router.post("/upload-sample", response_model=JobStatusResponse)
def trigger_sample_video(background_tasks: BackgroundTasks):
    sample_path = config.SAMPLES_DIR / "sample_traffic.mp4"
    if not sample_path.exists():
        from scripts.generate_sample_video import generate_sample_video
        generate_sample_video("sample_traffic.mp4")

    job_id = str(uuid.uuid4())
    conn = get_db_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        with conn:
            conn.execute("""
            INSERT INTO processing_jobs (
                id, source_filename, status, progress, current_frame, total_frames,
                fps, annotated_video_path, violations_count, error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id, str(sample_path), JobStatus.QUEUED.value, 0.0, 0, 0,
                0.0, None, 0, None, now, now
            ))
    finally:
        conn.close()

    background_tasks.add_task(run_video_job, job_id, str(sample_path), True)
    return get_job_status(job_id)

@router.post("/process/{job_id}", response_model=JobStatusResponse)
def start_processing(job_id: str, background_tasks: BackgroundTasks):
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT * FROM processing_jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")

        if row["status"] != JobStatus.QUEUED.value:
            raise HTTPException(
                status_code=409,
                detail=f"Job cannot be started while it is {row['status'].lower()}."
            )

        source_path = row["source_filename"]
        if not Path(source_path).exists():
            raise HTTPException(status_code=404, detail="Source video file missing on disk")

        # Launch processing in background task to avoid blocking the event loop
        now = datetime.now(timezone.utc).isoformat()
        with conn:
            cursor = conn.execute(
                "UPDATE processing_jobs SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (JobStatus.PROCESSING.value, now, job_id, JobStatus.QUEUED.value)
            )
            if cursor.rowcount == 0:
                raise HTTPException(status_code=409, detail="Job has already been started.")

        cursor = conn.execute("SELECT * FROM processing_jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        background_tasks.add_task(run_video_job, job_id, source_path, False)
        return _row_to_job(row)
    finally:
        conn.close()

@router.post("/cancel/{job_id}", response_model=JobStatusResponse)
def cancel_processing(job_id: str):
    conn = get_db_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        with conn:
            cursor = conn.execute(
                "UPDATE processing_jobs SET status = ?, updated_at = ? WHERE id = ?",
                (JobStatus.CANCELLED.value, now, job_id)
            )
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Job not found")

        cursor = conn.execute("SELECT * FROM processing_jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        return _row_to_job(row)
    finally:
        conn.close()

@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT * FROM processing_jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        return _row_to_job(row)
    finally:
        conn.close()

@router.get("/stream/{job_id}")
async def stream_annotated_video(job_id: str, request: Request):
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT annotated_video_path, status FROM processing_jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        if not row or not row["annotated_video_path"]:
            raise HTTPException(status_code=404, detail="Annotated video not available yet")

        path = Path(row["annotated_video_path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="Video file not found on disk")

        file_size = path.stat().st_size
        range_header = request.headers.get("Range")

        if not range_header:
            return FileResponse(
                path,
                media_type="video/mp4",
                filename=path.name,
                headers={"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
            )

        # Parse HTTP Range header: "bytes=start-end"
        range_value = range_header.strip().replace("bytes=", "")
        range_parts = range_value.split("-")
        start = int(range_parts[0]) if range_parts[0] else 0
        end = int(range_parts[1]) if len(range_parts) > 1 and range_parts[1] else file_size - 1
        end = min(end, file_size - 1)
        content_length = (end - start) + 1

        def iter_file():
            with open(path, "rb") as f:
                f.seek(start)
                bytes_left = content_length
                while bytes_left > 0:
                    chunk_size = min(bytes_left, 1024 * 512)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    bytes_left -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": "video/mp4",
            "Cache-Control": "no-cache",
        }
        return StreamingResponse(iter_file(), status_code=206, headers=headers)
    finally:
        conn.close()

def _row_to_job(row) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=row["id"],
        source_filename=Path(row["source_filename"]).name,
        status=JobStatus(row["status"]),
        progress=row["progress"],
        current_frame=row["current_frame"],
        total_frames=row["total_frames"],
        fps=row["fps"],
        annotated_video_path=row["annotated_video_path"],
        violations_count=row["violations_count"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"]
    )
