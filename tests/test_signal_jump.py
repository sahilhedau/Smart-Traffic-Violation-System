import pytest
import numpy as np
from ai.signal_jump.signal_engine import (
    segments_intersect,
    ccw,
    SignalJumpDetector
)
from ai.detection.detector import DetectionResult

def test_segments_intersect():
    # Horizontal line from (0, 10) to (20, 10)
    A = (0.0, 10.0)
    B = (20.0, 10.0)

    # Vertical trajectory from (10, 5) to (10, 15) crosses AB
    C = (10.0, 5.0)
    D = (10.0, 15.0)
    assert segments_intersect(A, B, C, D) is True

    # Trajectory that does not reach AB
    E = (10.0, 2.0)
    F = (10.0, 8.0)
    assert segments_intersect(A, B, E, F) is False

    # Parallel trajectory
    G = (0.0, 15.0)
    H = (20.0, 15.0)
    assert segments_intersect(A, B, G, H) is False

def test_signal_state_hsv_detection():
    detector = SignalJumpDetector(
        stop_line=[0.1, 0.5, 0.9, 0.5],
        signal_roi=[0.8, 0.1, 0.9, 0.3],
        signal_mode="auto_detect"
    )

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # In ROI (y: 10-30, x: 80-90), fill with bright RED
    # OpenCV BGR: Red is (0, 0, 255)
    frame[10:30, 80:90] = (0, 0, 255)
    state, conf = detector.determine_signal_state(frame)
    assert state == "RED"
    assert conf > 0.5

    # Fill ROI with bright GREEN (0, 255, 0)
    frame[10:30, 80:90] = (0, 255, 0)
    state_green, conf_green = detector.determine_signal_state(frame)
    assert state_green == "GREEN"

    # Blank frame -> UNKNOWN
    blank_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    state_blank, _ = detector.determine_signal_state(blank_frame)
    assert state_blank == "UNKNOWN"

def test_signal_jump_violation_trigger():
    detector = SignalJumpDetector(
        stop_line=[0.0, 0.5, 1.0, 0.5],
        signal_roi=[0.8, 0.1, 0.9, 0.3],
        signal_mode="red"
    )

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Stop line at y = 100

    det = DetectionResult(
        box=[40.0, 80.0, 80.0, 120.0],
        class_id=2,
        class_name="car",
        confidence=0.85,
        track_id=1
    )

    # Trajectory crosses from y=90 to y=110
    trajectories = {
        1: [(60.0, 90.0), (60.0, 110.0)]
    }

    violations = detector.check_violations(frame, [det], trajectories)
    assert len(violations) == 1
    assert violations[0]["event_type"] == "RED_LIGHT_CROSSING"
    assert violations[0]["tracking_id"] == 1

    # Second check for same track ID should be deduplicated
    violations_again = detector.check_violations(frame, [det], trajectories)
    assert len(violations_again) == 0
