import pytest
import os
import uuid
from pathlib import Path
from backend.app.config import config
from backend.app.database.db import get_db_connection, init_db
from ai.video.processor import VideoProcessor

@pytest.fixture(autouse=True)
def setup_db():
    init_db()

def test_video_pipeline_end_to_end():
    sample_video = config.SAMPLES_DIR / "sample_traffic.mp4"
    assert sample_video.exists(), "Sample video should exist for pipeline test"

    job_id = f"test_{uuid.uuid4().hex[:8]}"

    # Insert job into DB
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
            INSERT INTO processing_jobs (
                id, source_filename, status, progress, current_frame, total_frames,
                fps, annotated_video_path, violations_count, error_message, created_at, updated_at
            ) VALUES (?, ?, 'QUEUED', 0.0, 0, 0, 0.0, NULL, 0, NULL, 'now', 'now')
            """, (job_id, str(sample_video)))
    finally:
        conn.close()

    # Run processor
    processor = VideoProcessor(
        job_id=job_id,
        video_source=str(sample_video)
    )

    result = processor.process()
    assert result["status"] == "COMPLETED"
    assert result["total_frames"] > 0
    assert result["annotated_video_path"] is not None

    annotated_path = Path(result["annotated_video_path"])
    assert annotated_path.exists()
    assert annotated_path.stat().st_size > 1000

    # Verify database was updated
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT * FROM processing_jobs WHERE id = ?", (job_id,))
        job_row = cursor.fetchone()
        assert job_row["status"] == "COMPLETED"
        assert job_row["current_frame"] == result["total_frames"]
    finally:
        conn.close()
