import pytest
import io
import uuid
import cv2
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import config
from backend.app.database.db import get_db_connection, init_db
from backend.app.services.event_service import EventService
from backend.app.schemas.event import ViolationEventCreate, EventType, Severity
from backend.app.schemas.video import JobStatus
from ai.video.processor import VideoProcessor

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    init_db()

def test_upload_creates_job():
    sample_file = config.SAMPLES_DIR / "sample_traffic.mp4"
    assert sample_file.exists(), "Sample video file required for upload test"

    with open(sample_file, "rb") as f:
        response = client.post(
            "/api/video/upload",
            files={"file": ("test_upload.mp4", f, "video/mp4")}
        )

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["filename"] == "test_upload.mp4"

    # Verify job in DB
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT * FROM processing_jobs WHERE id = ?", (data["job_id"],)).fetchone()
        assert row is not None
        assert row["status"] == JobStatus.QUEUED.value
    finally:
        conn.close()

def test_processing_creates_playable_h264_output():
    sample_file = config.SAMPLES_DIR / "sample_traffic.mp4"
    job_id = f"test_h264_{uuid.uuid4().hex[:8]}"

    # Process video
    processor = VideoProcessor(job_id=job_id, video_source=str(sample_file), is_demo=False)
    result = processor.process()

    assert result["status"] == "COMPLETED"
    output_path = Path(result["annotated_video_path"])
    assert output_path.exists()
    assert output_path.stat().st_size > 5000

    # Test that OpenCV can reopen and decode the output video
    cap = cv2.VideoCapture(str(output_path))
    assert cap.isOpened() is True
    ret, frame = cap.read()
    assert ret is True
    assert frame is not None
    assert frame.shape[0] > 0 and frame.shape[1] > 0
    cap.release()

def test_stream_api_byte_ranges():
    sample_file = config.SAMPLES_DIR / "sample_traffic.mp4"
    job_id = f"test_stream_{uuid.uuid4().hex[:8]}"

    processor = VideoProcessor(job_id=job_id, video_source=str(sample_file), is_demo=False)
    res = processor.process()

    # Test full GET
    resp_full = client.get(f"/api/video/stream/{job_id}")
    assert resp_full.status_code == 200
    assert resp_full.headers.get("accept-ranges") == "bytes"

    # Test partial Range request
    resp_range = client.get(
        f"/api/video/stream/{job_id}",
        headers={"Range": "bytes=0-1023"}
    )
    assert resp_range.status_code == 206
    assert "bytes 0-1023/" in resp_range.headers.get("content-range", "")
    assert len(resp_range.content) == 1024

def test_failed_processing_reported_as_failed():
    job_id = f"test_fail_{uuid.uuid4().hex[:8]}"
    # Non-existent video file
    bad_source = "non_existent_video_file_xyz.mp4"

    processor = VideoProcessor(job_id=job_id, video_source=bad_source)
    with pytest.raises(ValueError):
        processor.process()

    conn = get_db_connection()
    try:
        row = conn.execute("SELECT status, error_message FROM processing_jobs WHERE id = ?", (job_id,)).fetchone()
        assert row is not None
        assert row["status"] == "FAILED"
        assert "Cannot open source video" in row["error_message"]
    finally:
        conn.close()

def test_clear_history_requires_confirmation_and_clears():
    # Insert a dummy event
    EventService.create_event(ViolationEventCreate(
        event_type=EventType.RED_LIGHT_CROSSING,
        timestamp="2026-10-11T00:00:00Z",
        confidence=0.9
    ))

    # Calling DELETE without confirm must fail with 400
    res_no_confirm = client.delete("/api/events/clear")
    assert res_no_confirm.status_code == 400

    # Calling DELETE with confirm=true must succeed
    res_confirmed = client.delete("/api/events/clear?confirm=true")
    assert res_confirmed.status_code == 200

    # Check stats are reset
    stats = client.get("/api/statistics").json()
    assert stats["total_violations"] == 0

def test_cancel_processing_endpoint():
    job_id = f"test_cancel_{uuid.uuid4().hex[:8]}"
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO processing_jobs (id, source_filename, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (job_id, "dummy.mp4", "PROCESSING", "now", "now")
            )
    finally:
        conn.close()

    res = client.post(f"/api/video/cancel/{job_id}")
    assert res.status_code == 200
    assert res.json()["status"] == "CANCELLED"
