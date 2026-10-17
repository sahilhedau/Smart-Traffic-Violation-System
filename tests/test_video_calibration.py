import cv2
import numpy as np
import uuid

import backend.app.api.video as video_api
from ai.detection.detector import DetectionResult
from backend.app.api.video import suggest_stop_line, suggest_video_calibration
from backend.app.database.db import get_db_connection, init_db


def test_stop_line_suggestion_finds_prominent_road_bar():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.line(frame, (80, 260), (560, 260), (255, 255, 255), 8)

    suggestion = suggest_stop_line(frame)

    assert suggestion is not None
    assert abs(suggestion[1] - 260 / 360) < 0.04
    assert abs(suggestion[3] - 260 / 360) < 0.04
    assert suggestion[2] - suggestion[0] > 0.6


def test_stop_line_suggestion_returns_none_without_long_road_edges():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    assert suggest_stop_line(frame) is None


def test_video_calibration_returns_normalized_signal_boxes(tmp_path, monkeypatch):
    init_db()
    job_id = f"calibration_{uuid.uuid4().hex}"
    source_path = tmp_path / "calibration.mp4"
    source_path.touch()
    conn = get_db_connection()
    try:
        conn.execute("""
        INSERT INTO processing_jobs (
            id, source_filename, status, progress, current_frame, total_frames,
            fps, annotated_video_path, violations_count, error_message, created_at, updated_at
        ) VALUES (?, ?, 'QUEUED', 0.0, 0, 0, 0.0, NULL, 0, NULL, 'now', 'now')
        """, (job_id, str(source_path)))
        conn.commit()
    finally:
        conn.close()

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.line(frame, (80, 260), (560, 260), (255, 255, 255), 8)

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            return 8 if prop == cv2.CAP_PROP_FRAME_COUNT else 250

        def set(self, prop, value):
            return True

        def read(self):
            return True, frame.copy()

        def release(self):
            pass

    class FakeDetector:
        def detect_and_track(self, image, persist):
            return [DetectionResult(
                box=[100, 20, 140, 100],
                class_id=9,
                class_name="traffic light",
                confidence=0.9,
            )]

    monkeypatch.setattr(video_api.cv2, "VideoCapture", lambda path: FakeCapture())
    monkeypatch.setattr(video_api, "_calibration_detector", FakeDetector())
    try:
        result = suggest_video_calibration(job_id)

        assert result["signal_rois"] == [[100 / 640, 20 / 360, 140 / 640, 100 / 360]]
        assert result["stop_line"] is not None
        assert result["frame_time_seconds"] == 0.25
    finally:
        conn = get_db_connection()
        try:
            conn.execute("DELETE FROM processing_jobs WHERE id = ?", (job_id,))
            conn.commit()
        finally:
            conn.close()
