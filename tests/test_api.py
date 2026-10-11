import pytest
import io
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "cuda_available" in data
    assert "device" in data

def test_api_statistics():
    response = client.get("/api/statistics")
    assert response.status_code == 200
    data = response.json()
    assert "total_violations" in data
    assert "red_light_count" in data

def test_api_settings():
    get_res = client.get("/api/settings")
    assert get_res.status_code == 200
    current_settings = get_res.json()

    # Update confidence threshold
    current_settings["confidence_threshold"] = 0.42
    patch_res = client.patch("/api/settings", json=current_settings)
    assert patch_res.status_code == 200
    assert patch_res.json()["confidence_threshold"] == 0.42

def test_api_events_pagination():
    response = client.get("/api/events?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data

def test_api_upload_invalid_extension():
    # Attempt to upload a .txt file
    fake_file = io.BytesIO(b"fake text content")
    response = client.post(
        "/api/video/upload",
        files={"file": ("malicious.txt", fake_file, "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]
