import pytest
import numpy as np
from ai.license_plate.plate_engine import PlateRecognitionEngine

def test_plate_normalization():
    engine = PlateRecognitionEngine()

    assert engine.normalize_plate_text("MH 12 AB 1234") == "MH12AB1234"
    assert engine.normalize_plate_text("dl-04c-1111") == "DL04C1111"
    assert engine.normalize_plate_text("KA.01.MJ.5050") == "KA01MJ5050"
    assert engine.normalize_plate_text("  hr26dq5551  ") == "HR26DQ5551"

    # Garbage / unreadable inputs
    assert engine.normalize_plate_text("") == "UNREADABLE"
    assert engine.normalize_plate_text("$$$") == "UNREADABLE"
    assert engine.normalize_plate_text("12") == "UNREADABLE"
    assert engine.normalize_plate_text("XYZ") == "UNREADABLE"

def test_plate_image_preprocessing():
    engine = PlateRecognitionEngine()
    dummy_crop = np.random.randint(0, 255, (40, 120, 3), dtype=np.uint8)

    preprocessed = engine.preprocess_plate_image(dummy_crop)
    assert preprocessed is not None
    assert len(preprocessed.shape) == 2  # Grayscale
    assert preprocessed.shape[0] == 70  # Standardized target height
