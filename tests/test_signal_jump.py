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

def test_signal_state_checks_multiple_calibrated_boxes():
    detector = SignalJumpDetector(
        stop_line=[0.1, 0.5, 0.9, 0.5],
        signal_roi=[0.0, 0.0, 0.4, 1.0],
        signal_rois=[[0.0, 0.0, 0.4, 1.0], [0.6, 0.0, 1.0, 1.0]],
        signal_mode="auto_detect"
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:40, 10:30] = (0, 0, 255)
    frame[50:70, 70:90] = (0, 255, 0)

    state, confidence = detector.determine_signal_state(frame)

    assert state == "RED"
    assert confidence == 1.0


def test_red_signal_only_triggers_its_paired_stop_line():
    detector = SignalJumpDetector(
        stop_line=[0.1, 0.5, 0.4, 0.5],
        signal_roi=[0.0, 0.0, 0.4, 0.4],
        signal_rois=[[0.0, 0.0, 0.4, 0.4], [0.6, 0.0, 1.0, 0.4]],
        stop_lines=[[0.25, 0.4, 0.25, 0.6], [0.75, 0.4, 0.75, 0.6]],
        signal_mode="auto_detect"
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:30, 10:30] = (0, 0, 255)
    frame[10:30, 70:90] = (0, 255, 0)
    states = detector.determine_signal_states(frame)
    assert [state for state, _ in states] == ["RED", "GREEN"]

    red_approach = DetectionResult(
        box=[10.0, 40.0, 40.0, 60.0], class_id=2,
        class_name="car", confidence=0.9, track_id=21
    )
    green_approach = DetectionResult(
        box=[60.0, 40.0, 90.0, 60.0], class_id=2,
        class_name="car", confidence=0.9, track_id=22
    )
    violations = detector.check_violations(
        frame,
        [red_approach, green_approach],
        {21: [(10.0, 50.0), (40.0, 50.0)], 22: [(60.0, 50.0), (90.0, 50.0)]},
        signal_states=states
    )

    assert len(violations) == 1
    assert violations[0]["tracking_id"] == 21
    assert violations[0]["signal_index"] == 0

def test_signal_state_uses_detected_traffic_light_box():
    detector = SignalJumpDetector(
        stop_line=[0.1, 0.5, 0.9, 0.5],
        signal_roi=[0.4, 0.2, 0.6, 0.8],
        signal_mode="auto_detect"
    )
    light = DetectionResult(
        box=[40.0, 20.0, 60.0, 80.0],
        class_id=9,
        class_name="traffic light",
        confidence=0.8
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[30:60, 45:55] = (0, 0, 255)

    state_red, confidence_red = detector.determine_signal_state(frame, [light])
    assert state_red == "RED"
    assert confidence_red > 0.9

    frame[30:60, 45:55] = (0, 255, 0)
    state_green, _ = detector.determine_signal_state(frame, [light])
    assert state_green == "GREEN"

def test_dim_green_background_is_not_treated_as_active_signal():
    detector = SignalJumpDetector(
        stop_line=[0.1, 0.5, 0.9, 0.5],
        signal_roi=[0.0, 0.0, 1.0, 1.0],
        signal_mode="auto_detect"
    )
    dim_green = np.full((9, 9, 3), (0, 70, 0), dtype=np.uint8)

    state, confidence = detector.determine_signal_state(dim_green)
    assert state == "UNKNOWN"
    assert confidence == 0.0

def test_signal_state_ignores_traffic_lights_outside_calibrated_area():
    detector = SignalJumpDetector(
        stop_line=[0.1, 0.5, 0.9, 0.5],
        signal_roi=[0.8, 0.1, 0.9, 0.3],
        signal_mode="auto_detect"
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:30, 80:90] = (0, 0, 255)
    frame[10:30, 10:20] = (0, 255, 0)
    unrelated_light = DetectionResult(
        box=[10.0, 10.0, 20.0, 30.0],
        class_id=9,
        class_name="traffic light",
        confidence=0.8
    )

    state, _ = detector.determine_signal_state(frame, [unrelated_light])
    assert state == "RED"

def test_angled_stop_line_detects_crossing_in_either_direction():
    detector = SignalJumpDetector(
        stop_line=[0.2, 0.8, 0.8, 0.2],
        signal_roi=[0.8, 0.1, 0.9, 0.3],
        signal_mode="red"
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    vehicle = DetectionResult(
        box=[45.0, 45.0, 55.0, 55.0],
        class_id=2,
        class_name="car",
        confidence=0.9,
        track_id=10
    )

    forward = detector.check_violations(
        frame,
        [vehicle],
        {10: [(50.0, 40.0), (50.0, 60.0)]}
    )
    reverse = detector.check_violations(
        frame,
        [DetectionResult(
            box=vehicle.box,
            class_id=2,
            class_name="car",
            confidence=0.9,
            track_id=12
        )],
        {12: [(50.0, 60.0), (50.0, 40.0)]}
    )

    assert len(forward) == 1
    assert len(reverse) == 1

def test_green_signal_does_not_raise_red_light_violation():
    detector = SignalJumpDetector(
        stop_line=[0.0, 0.5, 1.0, 0.5],
        signal_roi=[0.8, 0.1, 0.9, 0.3],
        signal_mode="green"
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    vehicle = DetectionResult(
        box=[40.0, 40.0, 60.0, 60.0],
        class_id=2,
        class_name="car",
        confidence=0.9,
        track_id=11
    )

    violations = detector.check_violations(
        frame,
        [vehicle],
        {11: [(50.0, 40.0), (50.0, 60.0)]}
    )
    assert violations == []

def test_auto_detected_red_signal_triggers_crossing_violation():
    detector = SignalJumpDetector(
        stop_line=[0.0, 0.5, 1.0, 0.5],
        signal_roi=[0.8, 0.1, 0.9, 0.3],
        signal_mode="auto_detect"
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:30, 80:90] = (0, 0, 255)
    traffic_light = DetectionResult(
        box=[80.0, 10.0, 90.0, 30.0],
        class_id=9,
        class_name="traffic light",
        confidence=0.8
    )
    vehicle = DetectionResult(
        box=[40.0, 40.0, 60.0, 60.0],
        class_id=2,
        class_name="car",
        confidence=0.9,
        track_id=13
    )

    violations = detector.check_violations(
        frame,
        [vehicle],
        {13: [(50.0, 40.0), (50.0, 60.0)]},
        traffic_light_detections=[traffic_light]
    )

    assert len(violations) == 1
    assert violations[0]["signal_state"] == "RED"
    assert violations[0]["signal_confidence"] > 0.9

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
