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


def test_ocr_compares_multiscale_variants_and_prefers_valid_plate_format():
    engine = PlateRecognitionEngine()

    class MockReader:
        def __init__(self):
            self.images = []
            self.results = [
                [(None, "ABC1234", 0.95)],
                [(None, "MH12AB1234", 0.42)],
                [(None, "MH12AB1234", 0.45)],
            ]

        def readtext(self, image, **kwargs):
            self.images.append(image)
            return self.results[len(self.images) - 1]

    engine.reader = MockReader()
    crop = np.full((28, 120, 3), 160, dtype=np.uint8)

    text, raw_text, confidence = engine.recognize_crop(crop)

    assert text == "MH12AB1234"
    assert raw_text == "MH12AB1234"
    assert confidence == 0.45
    assert len(engine.reader.images) == 3
    assert engine.reader.images[1].shape[0] == 140
