from typing import List, Optional
from pydantic import BaseModel, Field

class CalibrationSettings(BaseModel):
    stop_line: List[float] = Field(default=[0.05, 0.65, 0.95, 0.65], description="[x1, y1, x2, y2] normalized 0.0 to 1.0")
    signal_roi: List[float] = Field(default=[0.75, 0.05, 0.95, 0.30], description="[x1, y1, x2, y2] normalized 0.0 to 1.0")
    signal_mode: str = Field(default="red", description="'auto_detect', 'red', 'yellow', 'green'")

class SystemSettings(BaseModel):
    confidence_threshold: float = Field(default=0.35, ge=0.1, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.1, le=1.0)
    frame_skip: int = Field(default=1, ge=1, le=10)
    max_processing_fps: int = Field(default=30, ge=1, le=60)
    
    enable_signal_jump: bool = True
    enable_triple_riding: bool = True
    enable_helmet_check: bool = True
    enable_plate_ocr: bool = True
    privacy_blur: bool = False

    calibration: CalibrationSettings = CalibrationSettings()
