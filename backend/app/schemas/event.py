from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class EventType(str, Enum):
    RED_LIGHT_CROSSING = "RED_LIGHT_CROSSING"
    TRIPLE_RIDING = "TRIPLE_RIDING"
    NO_HELMET = "NO_HELMET"

class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ReviewStatus(str, Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    CONFIRMED_BY_REVIEWER = "CONFIRMED_BY_REVIEWER"
    DISMISSED = "DISMISSED"
    UNREADABLE_PLATE = "UNREADABLE_PLATE"

class ViolationEventBase(BaseModel):
    event_type: EventType
    timestamp: str
    camera_id: str = "CAM_01"
    tracking_id: Optional[int] = None
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Severity = Severity.MEDIUM
    status: ReviewStatus = ReviewStatus.PENDING_REVIEW
    evidence_path: Optional[str] = None
    crop_path: Optional[str] = None
    plate_text: Optional[str] = None
    plate_raw_text: Optional[str] = None
    plate_confidence: Optional[float] = 0.0
    details: Optional[Dict[str, Any]] = None
    reviewer_notes: Optional[str] = None

class ViolationEventCreate(ViolationEventBase):
    id: Optional[str] = None

class ViolationEventUpdate(BaseModel):
    status: Optional[ReviewStatus] = None
    plate_text: Optional[str] = None
    reviewer_notes: Optional[str] = None

class ViolationEventResponse(ViolationEventBase):
    id: str
    created_at: str
    updated_at: str

class PaginatedEventsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[ViolationEventResponse]

class DashboardStatsResponse(BaseModel):
    total_violations: int
    red_light_count: int
    triple_riding_count: int
    no_helmet_count: int
    pending_review_count: int
    confirmed_count: int
    dismissed_count: int
    recent_events: List[ViolationEventResponse]
