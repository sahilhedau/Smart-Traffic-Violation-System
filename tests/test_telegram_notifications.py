import sqlite3
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app
from backend.app.schemas.event import EventType, Severity, ViolationEventCreate
from backend.app.services.event_service import EventService
from backend.app.services.telegram_notification_service import TelegramNotificationService
from backend.app.database.db import _create_tables

client = TestClient(app)


@pytest.fixture
def event():
    created = EventService.create_event(ViolationEventCreate(
        id=str(uuid.uuid4()),
        event_type=EventType.TRIPLE_RIDING,
        timestamp=datetime.now(timezone.utc).isoformat(),
        tracking_id=248,
        confidence=0.416,
        severity=Severity.HIGH,
        plate_text="TS09PB3596",
    ))
    yield created

    from backend.app.database.db import get_db_connection

    conn = get_db_connection()
    try:
        with conn:
            conn.execute("DELETE FROM violation_events WHERE id = ?", (created.id,))
    finally:
        conn.close()


def successful_telegram_response():
    return SimpleNamespace(status_code=200, json=lambda: {"ok": True})


def configure_telegram(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test-chat")


def test_confirmation_persists_before_sending_and_attaches_evidence(
    event,
    monkeypatch,
    tmp_path,
):
    configure_telegram(monkeypatch)
    image_path = tmp_path / "evidence.png"
    Image.new("RGB", (2, 2), color="red").save(image_path)
    from backend.app.database.db import get_db_connection

    conn = get_db_connection()
    try:
        conn.execute(
            "UPDATE violation_events SET evidence_path = ? WHERE id = ?",
            (str(image_path), event.id),
        )
        conn.commit()
    finally:
        conn.close()

    requests = []

    def fake_post(url, **kwargs):
        saved = EventService.get_event(event.id)
        assert saved.status.value == "CONFIRMED_BY_REVIEWER"
        assert saved.telegram_notification_status == "SENDING"
        requests.append((url, kwargs))
        return successful_telegram_response()

    monkeypatch.setattr(httpx, "post", fake_post)
    response = client.patch(
        f"/api/events/{event.id}",
        json={
            "status": "CONFIRMED_BY_REVIEWER",
            "reviewer_notes": "Reviewed",
            "plate_text": "TS09PB3596",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "CONFIRMED_BY_REVIEWER"
    assert data["telegram_notification_status"] == "SENT"
    assert len(requests) == 1
    url, kwargs = requests[0]
    assert url.endswith("/sendPhoto")
    assert kwargs["files"]["photo"][0] == "evidence.png"
    caption = kwargs["data"]["caption"]
    assert "TrafficGuard AI" in caption
    assert "Violation Type: Triple Riding" in caption
    assert f"Event ID: {event.id}" in caption
    assert "Tracking ID: 248" in caption
    assert "License Plate OCR: TS09PB3596" in caption
    assert "AI Confidence: 41.6%" in caption
    assert "Review Status: CONFIRMED BY REVIEWER" in caption


def test_dismissing_event_does_not_send_telegram(event, monkeypatch):
    attempted = []
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: attempted.append(args))

    response = client.patch(
        f"/api/events/{event.id}",
        json={"status": "DISMISSED"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "DISMISSED"
    assert attempted == []


def test_repeated_confirmation_does_not_send_duplicate(event, monkeypatch):
    configure_telegram(monkeypatch)
    attempts = []
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: (attempts.append(args), successful_telegram_response())[1],
    )

    for _ in range(2):
        response = client.patch(
            f"/api/events/{event.id}",
            json={"status": "CONFIRMED_BY_REVIEWER"},
        )
        assert response.status_code == 200

    assert len(attempts) == 1


def test_telegram_failure_keeps_confirmation_and_can_be_retried(event, monkeypatch):
    configure_telegram(monkeypatch)
    attempts = []

    def fake_post(url, **kwargs):
        attempts.append(url)
        if len(attempts) == 1:
            return SimpleNamespace(
                status_code=401,
                json=lambda: {"ok": False, "error_code": 401},
            )
        return successful_telegram_response()

    monkeypatch.setattr(httpx, "post", fake_post)
    response = client.patch(
        f"/api/events/{event.id}",
        json={"status": "CONFIRMED_BY_REVIEWER"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "CONFIRMED_BY_REVIEWER"
    assert response.json()["telegram_notification_status"] == "FAILED"
    assert response.json()["telegram_notification_error"] == "Telegram API returned HTTP 401"
    assert EventService.get_event(event.id).status.value == "CONFIRMED_BY_REVIEWER"

    sent, message = TelegramNotificationService.notify_confirmed_event(
        event.id,
        retry=True,
    )
    assert sent is True
    assert message == "Telegram notification sent"
    assert len(attempts) == 2
    assert EventService.get_event(event.id).telegram_notification_status == "SENT"


def test_missing_telegram_configuration_does_not_break_confirmation(event, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    attempted = []
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: attempted.append(args))

    response = client.patch(
        f"/api/events/{event.id}",
        json={"status": "CONFIRMED_BY_REVIEWER"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "CONFIRMED_BY_REVIEWER"
    assert response.json()["telegram_notification_status"] == "NOT_CONFIGURED"
    assert attempted == []


def test_missing_evidence_sends_text_message(event, monkeypatch):
    configure_telegram(monkeypatch)
    requests = []
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kwargs: (requests.append((url, kwargs)), successful_telegram_response())[1],
    )

    response = client.patch(
        f"/api/events/{event.id}",
        json={"status": "CONFIRMED_BY_REVIEWER"},
    )

    assert response.status_code == 200
    assert response.json()["telegram_notification_status"] == "SENT"
    assert requests[0][0].endswith("/sendMessage")
    assert "text" in requests[0][1]["data"]
    assert "files" not in requests[0][1]


def test_legacy_database_is_migrated_without_losing_events(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE violation_events (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            status TEXT,
            created_at TEXT
        )
    """)
    conn.execute(
        "INSERT INTO violation_events (id, event_type, status, created_at) VALUES (?, ?, ?, ?)",
        ("legacy-event", "TRIPLE_RIDING", "PENDING_REVIEW", "then"),
    )

    _create_tables(conn)

    row = conn.execute(
        "SELECT id, status, telegram_notification_status FROM violation_events WHERE id = ?",
        ("legacy-event",),
    ).fetchone()
    assert dict(row) == {
        "id": "legacy-event",
        "status": "PENDING_REVIEW",
        "telegram_notification_status": "NOT_SENT",
    }
    conn.close()


def test_test_message_is_clearly_labeled(monkeypatch):
    configure_telegram(monkeypatch)
    sent_messages = []
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kwargs: (sent_messages.append(kwargs["data"]["text"]), successful_telegram_response())[1],
    )

    sent, result = TelegramNotificationService.send_test_message()

    assert sent is True
    assert result == "Test message sent"
    assert "TEST MESSAGE" in sent_messages[0]
