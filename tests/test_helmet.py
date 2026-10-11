import pytest
import numpy as np
from ai.helmet.helmet_engine import HelmetDetector
from ai.detection.detector import DetectionResult

def test_head_crop_extraction():
    detector = HelmetDetector(min_confidence=0.4)
    frame = np.zeros((300, 300, 3), dtype=np.uint8)

    # Valid person box
    person_box = [50.0, 50.0, 100.0, 200.0]
    crop = detector.extract_head_crop(frame, person_box)
    assert crop is not None
    # Upper 25% of height (150 * 0.25 = 37.5px)
    assert crop.shape[0] >= 30
    assert crop.shape[1] == 50

    # Low resolution person box (tiny box)
    tiny_box = [10.0, 10.0, 20.0, 25.0]
    tiny_crop = detector.extract_head_crop(frame, tiny_box)
    assert tiny_crop is None

def test_helmet_baseline_classifier():
    detector = HelmetDetector()

    # Synthetic bare head with skin tone in HSV: (H: 15, S: 100, V: 200) -> BGR
    # High skin ratio should classify as NO_HELMET
    head_skin = np.full((40, 40, 3), (130, 170, 225), dtype=np.uint8) # BGR skin
    # Add texture noise to simulate hair/face features
    noise = np.random.randint(-20, 20, (40, 40, 3), dtype=np.int16)
    head_skin = np.clip(head_skin.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cls_skin, conf_skin = detector.classify_head_baseline(head_skin)
    assert cls_skin in ["NO_HELMET", "UNKNOWN"]
