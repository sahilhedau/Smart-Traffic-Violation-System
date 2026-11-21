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
    assert preprocessed.shape[0] == 80  # Upscales small crops while preserving resolution


def test_ocr_compares_multiscale_variants_and_prefers_valid_plate_format():
    engine = PlateRecognitionEngine()

    class MockReader:
        def __init__(self):
            self.images = []
            self.results = [
                [(None, "ABC1234", 0.95)],
                [(None, "MH12AB1234", 0.42)],
                [(None, "MH12AB1234", 0.45)],
                [(None, "MH12AB1234", 0.39)],
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
    assert len(engine.reader.images) == 4
    assert engine.reader.images[1].shape[0] == 140


def test_ocr_returns_low_confidence_valid_guess_for_manual_review():
    engine = PlateRecognitionEngine()

    class MockReader:
        def readtext(self, image, **kwargs):
            return [(None, "MH12AB1234", 0.12)]

    engine.reader = MockReader()

    text, raw_text, confidence = engine.recognize_crop(
        np.full((12, 80, 3), 160, dtype=np.uint8)
    )

    assert text == "MH12AB1234"
    assert raw_text == "MH12AB1234"
    assert confidence == 0.12
    assert engine.is_verified_plate(text, confidence) is False


def test_plate_recognition_retries_heuristic_crop_when_yolo_crop_is_unreadable():
    engine = PlateRecognitionEngine()
    frame = np.full((100, 160, 3), 100, dtype=np.uint8)
    heuristic_crop = np.full((18, 70, 3), 200, dtype=np.uint8)
    engine.detect_plates_yolo = lambda image, confidence_threshold=None: [
        {"box": [10, 10, 40, 25], "confidence": 0.18}
    ]
    engine.find_plate_candidate_region = lambda image, include_yolo=True: heuristic_crop

    ocr_calls = []

    def recognize_crop(crop):
        ocr_calls.append(crop)
        if len(ocr_calls) == 1:
            return "UNREADABLE", "", 0.0
        return "MH12AB1234", "MH12AB1234", 0.22

    engine.recognize_crop = recognize_crop

    text, raw_text, confidence, crop = engine.recognize_plate(
        frame,
        [20, 10, 100, 70]
    )

    assert text == "MH12AB1234"
    assert raw_text == "MH12AB1234"
    assert confidence == 0.22
    assert crop is heuristic_crop
    assert len(ocr_calls) == 3
    assert engine.last_plate_box is None
