import pytest
import numpy as np
from ai.license_plate.plate_engine import PlateRecognitionEngine

def test_plate_engine_initialization():
    engine = PlateRecognitionEngine()
    assert engine is not None
    # normalize_plate_text tests
    assert engine.normalize_plate_text("KA 05 AB 1234") == "KA05AB1234"
    assert engine.normalize_plate_text("MH 02 BG 4444") == "MH02BG4444"
    assert engine.normalize_plate_text("INVALID") == "UNREADABLE"

def test_detect_and_recognize_all():
    engine = PlateRecognitionEngine()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Synthetic vehicle bounding box
    vehicle_boxes = [[100.0, 150.0, 300.0, 400.0]]
    results = engine.detect_and_recognize_all(frame, vehicle_boxes)
    assert isinstance(results, list)
    if results:
        res = results[0]
        assert "vehicle_box" in res
        assert "plate_text" in res
        assert "confidence" in res
        assert "is_verified" in res

def test_is_verified_plate():
    engine = PlateRecognitionEngine()
    assert engine.is_verified_plate("MH12AB1234", 0.85) is True
    assert engine.is_verified_plate("UNREADABLE", 0.90) is False
    assert engine.is_verified_plate("MH12AB1234", 0.10) is False
