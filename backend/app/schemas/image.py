from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ImageDetectionBox(BaseModel):
    box: List[float]  # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    track_id: Optional[int] = None

class PlateDetectionResult(BaseModel):
    vehicle_box: List[float]
    plate_text: str
    raw_text: str
    confidence: float
    is_verified: bool

class HelmetDetectionItem(BaseModel):
    box: List[float]
    status: str  # 'HELMET' | 'NO_HELMET' | 'UNKNOWN'
    confidence: float
    is_violation: bool

class ImageViolationItem(BaseModel):
    event_type: str
    confidence: float
    severity: str
    description: str
    associated_box: Optional[List[float]] = None

class ImageDetectionResponse(BaseModel):
    status: str = "success"
    filename: str
    width: int
    height: int
    inference_time_ms: float
    annotated_image_url: str
    counts: Dict[str, int]
    detections: List[ImageDetectionBox]
    helmet_detections: List[HelmetDetectionItem]
    plate_detections: List[PlateDetectionResult]
    violations: List[ImageViolationItem]
