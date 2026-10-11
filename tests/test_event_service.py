import pytest
import uuid
from datetime import datetime, timezone
from backend.app.database.db import init_db
from backend.app.schemas.event import (
    ViolationEventCreate,
    ViolationEventUpdate,
    EventType,
    Severity,
    ReviewStatus,
)
from backend.app.services.event_service import EventService

@pytest.fixture(autouse=True)
def setup_db():
    init_db()

def test_event_creation_and_retrieval():
    event_id = str(uuid.uuid4())
    event_create = ViolationEventCreate(
        id=event_id,
        event_type=EventType.RED_LIGHT_CROSSING,
        timestamp=datetime.now(timezone.utc).isoformat(),
        camera_id="CAM_TEST_01",
        tracking_id=101,
        confidence=0.88,
        severity=Severity.HIGH,
        plate_text="MH12AB1234",
        plate_confidence=0.92,
        reviewer_notes="Test crossing event"
    )

    created = EventService.create_event(event_create)
    assert created.id == event_id
    assert created.event_type == EventType.RED_LIGHT_CROSSING
    assert created.plate_text == "MH12AB1234"
    assert created.status == ReviewStatus.PENDING_REVIEW

    fetched = EventService.get_event(event_id)
    assert fetched is not None
    assert fetched.id == event_id

def test_event_deduplication():
    track_id = 999
    event_create = ViolationEventCreate(
        event_type=EventType.TRIPLE_RIDING,
        timestamp=datetime.now(timezone.utc).isoformat(),
        tracking_id=track_id,
        confidence=0.85
    )
    EventService.create_event(event_create)

    # Immediately checking deduplication should report true
    assert EventService.is_duplicate(EventType.TRIPLE_RIDING.value, track_id, window_seconds=10.0) is True
    # Different track ID should report false
    assert EventService.is_duplicate(EventType.TRIPLE_RIDING.value, 888, window_seconds=10.0) is False

def test_event_update():
    event_create = ViolationEventCreate(
        event_type=EventType.NO_HELMET,
        timestamp=datetime.now(timezone.utc).isoformat(),
        tracking_id=102,
        confidence=0.75,
        plate_text="UNREADABLE"
    )
    created = EventService.create_event(event_create)

    # Reviewer confirms and corrects plate
    update = ViolationEventUpdate(
        status=ReviewStatus.CONFIRMED_BY_REVIEWER,
        plate_text="DL04C1111",
        reviewer_notes="Confirmed by manual inspection"
    )
    updated = EventService.update_event(created.id, update)
    assert updated.status == ReviewStatus.CONFIRMED_BY_REVIEWER
    assert updated.plate_text == "DL04C1111"
    assert updated.reviewer_notes == "Confirmed by manual inspection"

def test_statistics_and_filtering():
    stats = EventService.get_statistics()
    assert stats.total_violations >= 0
    assert stats.pending_review_count >= 0

    # Filter test
    filtered = EventService.list_events(event_type=EventType.NO_HELMET.value)
    for item in filtered.items:
        assert item.event_type == EventType.NO_HELMET
