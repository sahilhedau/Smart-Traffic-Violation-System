import io
import cv2
import pytest
import numpy as np
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config import config

client = TestClient(app)

def create_test_image_bytes(width=640, height=480) -> io.BytesIO:
    """Creates a sample test JPEG image with geometric shapes."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Background road
    img[200:480, :] = (60, 60, 60)
    # Road lane markings
    cv2.line(img, (320, 200), (320, 480), (255, 255, 255), 4)
    # Synthetic vehicle box
    cv2.rectangle(img, (150, 240), (350, 400), (0, 0, 180), -1)
    
    success, encoded = cv2.imencode(".jpg", img)
    assert success
    return io.BytesIO(encoded.tobytes())

def test_image_detect_invalid_extension():
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/api/image/detect", files=files)
    assert response.status_code == 400
    assert "Unsupported image format" in response.json()["detail"]

def test_image_detect_success():
    img_bytes = create_test_image_bytes()
    files = {"file": ("sample_traffic.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/image/detect", files=files, data={"save_violations_to_db": "false"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "sample_traffic.jpg"
    assert data["width"] == 640
    assert data["height"] == 480
    assert "annotated_image_url" in data
    assert "counts" in data
    assert isinstance(data["detections"], list)
    assert isinstance(data["helmet_detections"], list)
    assert isinstance(data["plate_detections"], list)
    assert isinstance(data["violations"], list)
    assert data["inference_time_ms"] > 0

    # Test evidence image retrieval
    evidence_url = data["annotated_image_url"]
    ev_response = client.get(evidence_url)
    assert ev_response.status_code == 200
    assert ev_response.headers["content-type"] == "image/jpeg"

def test_image_detect_records_whole_frame_no_helmet_violation(monkeypatch):
    from backend.app.api import image as image_api

    class StubDetector:
        def detect_and_track(self, frame, persist=False):
            return []

    class StubHelmetDetector:
        def detect_in_frame(self, frame):
            return [{
                "box": [200.0, 100.0, 240.0, 145.0],
                "class_name": "Without Helmet",
                "confidence": 0.91,
                "is_violation": True,
            }]

    class StubPlateEngine:
        pass

    class StubTripleRidingDetector:
        def associate_riders(self, motorcycles, persons):
            return {}

    monkeypatch.setattr(
        image_api,
        "get_detectors",
        lambda: (
            StubDetector(),
            StubHelmetDetector(),
            StubPlateEngine(),
            StubTripleRidingDetector(),
        ),
    )

    response = client.post(
        "/api/image/detect",
        files={"file": ("helmet.jpg", create_test_image_bytes(), "image/jpeg")},
        data={"save_violations_to_db": "false"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["counts"]["no_helmets_detected"] == 1
    assert data["counts"]["violations"] == 1
    assert data["helmet_detections"][0]["is_violation"] is True
    assert data["violations"][0]["event_type"] == "NO_HELMET"
    assert data["violations"][0]["confidence"] == 0.91

def test_image_evidence_not_found():
    response = client.get("/api/image/evidence/non_existent_file.jpg")
    assert response.status_code == 404
