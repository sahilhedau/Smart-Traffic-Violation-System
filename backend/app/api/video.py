import os
import uuid
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from backend.app.config import config
from backend.app.database.db import get_db_connection
from backend.app.schemas.video import JobStatusResponse, VideoUploadResponse, JobStatus
from ai.video.processor import VideoProcessor

logger = logging.getLogger("traffic_system.api.video")
router = APIRouter(prefix="/video", tags=["video"])

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

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

        source_path = row["source_filename"]
        if not Path(source_path).exists():
            raise HTTPException(status_code=404, detail="Source video file missing on disk")

        # Launch processing in background task to avoid blocking the event loop
        background_tasks.add_task(run_video_job, job_id, source_path, False)

        # Update status to PROCESSING
        now = datetime.now(timezone.utc).isoformat()
        with conn:
            conn.execute("UPDATE processing_jobs SET status = ?, updated_at = ? WHERE id = ?",
                         (JobStatus.PROCESSING.value, now, job_id))

        cursor = conn.execute("SELECT * FROM processing_jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
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
