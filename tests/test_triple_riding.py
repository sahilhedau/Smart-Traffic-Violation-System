import pytest
from ai.triple_riding.triple_riding_engine import TripleRidingDetector
from ai.detection.detector import DetectionResult

def test_rider_motorcycle_association():
    detector = TripleRidingDetector(min_riders=3, min_frames=3)

    # Motorcycle at [100, 150, 180, 250]
    moto = DetectionResult(
        box=[100.0, 150.0, 180.0, 250.0],
        class_id=3,
        class_name="motorcycle",
        confidence=0.90,
        track_id=10
    )

    # Rider seated directly above/on the motorcycle: [110, 110, 140, 200]
    rider = DetectionResult(
        box=[110.0, 110.0, 140.0, 200.0],
        class_id=0,
        class_name="person",
        confidence=0.85
    )

    # Distant pedestrian walking on the sidewalk: [400, 100, 440, 220]
    pedestrian = DetectionResult(
        box=[400.0, 100.0, 440.0, 220.0],
        class_id=0,
        class_name="person",
        confidence=0.80
    )

    assert detector.is_rider_on_motorcycle(rider, moto) is True
    assert detector.is_rider_on_motorcycle(pedestrian, moto) is False

def test_triple_riding_temporal_smoothing():
    detector = TripleRidingDetector(min_riders=3, min_frames=3)

    moto = DetectionResult(
        box=[100.0, 150.0, 220.0, 250.0],
        class_id=3,
        class_name="motorcycle",
        confidence=0.90,
        track_id=55
    )

    # 3 riders on the bike
    riders = [
        DetectionResult(box=[110.0, 110.0, 140.0, 200.0], class_id=0, class_name="person", confidence=0.85),
        DetectionResult(box=[145.0, 110.0, 175.0, 200.0], class_id=0, class_name="person", confidence=0.85),
        DetectionResult(box=[180.0, 110.0, 210.0, 200.0], class_id=0, class_name="person", confidence=0.85),
    ]

    # Frame 1: count is 3, but should not trigger immediately (needs min_frames)
    v1, counts1 = detector.process_frame([moto], riders)
    assert counts1[55] == 3
    assert len(v1) == 0

    # Frame 2:
    v2, _ = detector.process_frame([moto], riders)
    # Temporal smoothing requires consecutive confirmations
    # Frame 3:
    v3, _ = detector.process_frame([moto], riders)
    assert len(v3) == 1
    assert v3[0]["event_type"] == "TRIPLE_RIDING"
    assert v3[0]["tracking_id"] == 55
