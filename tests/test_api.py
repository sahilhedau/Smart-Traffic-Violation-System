import pytest
import io
import uuid
from fastapi import BackgroundTasks, HTTPException
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.api.video import start_processing
from backend.app.database.db import get_db_connection, init_db

client = TestClient(app)

def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "cuda_available" in data
    assert "device" in data

def test_api_statistics():
    response = client.get("/api/statistics")
    assert response.status_code == 200
    data = response.json()
    assert "total_violations" in data
    assert "red_light_count" in data

def test_api_settings():
    get_res = client.get("/api/settings")
    assert get_res.status_code == 200
    current_settings = get_res.json()

    # Update confidence threshold
    current_settings["confidence_threshold"] = 0.42
    patch_res = client.patch("/api/settings", json=current_settings)
    assert patch_res.status_code == 200
    assert patch_res.json()["confidence_threshold"] == 0.42

def test_api_events_pagination():
    response = client.get("/api/events?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data

def test_api_upload_invalid_extension():
    # Attempt to upload a .txt file
    fake_file = io.BytesIO(b"fake text content")
    response = client.post(
        "/api/video/upload",
        files={"file": ("malicious.txt", fake_file, "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_video_upload_waits_in_queue_until_started():
    response = client.post(
        "/api/video/upload",
        files={"file": ("staged-video.mp4", io.BytesIO(b"video placeholder"), "video/mp4")}
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    try:
        status = client.get(f"/api/video/status/{job_id}")
        assert status.status_code == 200
        assert status.json()["status"] == "QUEUED"
    finally:
        from backend.app.database.db import get_db_connection
        conn = get_db_connection()
        try:
            row = conn.execute(
                "SELECT source_filename FROM processing_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row:
                from pathlib import Path
                Path(row["source_filename"]).unlink(missing_ok=True)
                conn.execute("DELETE FROM processing_jobs WHERE id = ?", (job_id,))
                conn.commit()
        finally:
            conn.close()

def test_start_processing_starts_queued_job_once(tmp_path):
    init_db()
    job_id = f"test_{uuid.uuid4().hex}"
    source_path = tmp_path / "queued-video.mp4"
    source_path.write_bytes(b"test video")
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
            INSERT INTO processing_jobs (
                id, source_filename, status, progress, current_frame, total_frames,
                fps, annotated_video_path, violations_count, error_message, created_at, updated_at
            ) VALUES (?, ?, 'QUEUED', 0.0, 0, 0, 0.0, NULL, 0, NULL, 'now', 'now')
            """, (job_id, str(source_path)))
    finally:
        conn.close()

    try:
        tasks = BackgroundTasks()
        response = start_processing(job_id, tasks)
        assert response.status == "PROCESSING"
        assert len(tasks.tasks) == 1

        with pytest.raises(HTTPException) as error:
            start_processing(job_id, BackgroundTasks())
        assert error.value.status_code == 409
    finally:
        conn = get_db_connection()
        try:
            conn.execute("DELETE FROM processing_jobs WHERE id = ?", (job_id,))
            conn.commit()
        finally:
            conn.close()
