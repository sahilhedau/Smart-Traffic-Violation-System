import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse

from backend.app.schemas.event import (
    ViolationEventResponse,
    ViolationEventUpdate,
    PaginatedEventsResponse,
    EventType,
    ReviewStatus,
)
from backend.app.services.event_service import EventService

router = APIRouter(prefix="/events", tags=["events"])

@router.get("", response_model=PaginatedEventsResponse)
def list_events(
    event_type: Optional[EventType] = Query(None, description="Filter by violation type"),
    status: Optional[ReviewStatus] = Query(None, description="Filter by review status"),
    search: Optional[str] = Query(None, description="Search by plate number, notes, or event ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    type_str = event_type.value if event_type else None
    status_str = status.value if status else None
    return EventService.list_events(
        event_type=type_str,
        status=status_str,
        search=search,
        page=page,
        page_size=page_size
    )

@router.get("/{event_id}", response_model=ViolationEventResponse)
def get_event(event_id: str):
    event = EventService.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Violation event not found")
    return event

@router.patch("/{event_id}", response_model=ViolationEventResponse)
def update_event(event_id: str, update: ViolationEventUpdate):
    updated = EventService.update_event(event_id, update)
    if not updated:
        raise HTTPException(status_code=404, detail="Violation event not found")
    return updated

@router.get("/{event_id}/evidence")
def get_event_evidence(event_id: str):
    event = EventService.get_event(event_id)
    if not event or not event.evidence_path:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    path = Path(event.evidence_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Evidence file missing on disk")
    return FileResponse(path, media_type="image/jpeg")

@router.get("/{event_id}/crop")
def get_event_crop(event_id: str):
    event = EventService.get_event(event_id)
    if not event or not event.crop_path:
        raise HTTPException(status_code=404, detail="Plate crop not found")

    path = Path(event.crop_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Crop file missing on disk")
    return FileResponse(path, media_type="image/jpeg")

@router.delete("/clear", tags=["events"])
def clear_all_events(confirm: bool = Query(False, description="Must be true to confirm deletion")):
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Pass '?confirm=true' to explicitly delete all historical violation events."
        )
    EventService.clear_all_events()
    return {"message": "All violation events have been cleared from database."}
