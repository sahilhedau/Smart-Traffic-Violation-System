from typing import List
from math import isfinite
from pydantic import BaseModel, Field
from pydantic import model_validator

class CalibrationSettings(BaseModel):
    stop_line: List[float] = Field(default=[0.05, 0.65, 0.95, 0.65], description="[x1, y1, x2, y2] normalized 0.0 to 1.0")
    stop_lines: List[List[float]] = Field(default_factory=lambda: [[0.05, 0.65, 0.95, 0.65]], description="One stop line per signal box, each [x1, y1, x2, y2]")
    signal_roi: List[float] = Field(default=[0.75, 0.05, 0.95, 0.30], description="[x1, y1, x2, y2] normalized 0.0 to 1.0")
    signal_rois: List[List[float]] = Field(default_factory=lambda: [[0.75, 0.05, 0.95, 0.30]], description="Signal regions, each [x1, y1, x2, y2] normalized 0.0 to 1.0")
    signal_mode: str = Field(default="auto_detect", description="'auto_detect', 'red', 'yellow', 'green'")

    @model_validator(mode="before")
    @classmethod
    def migrate_single_signal_roi(cls, values):
        if isinstance(values, dict):
            values = values.copy()
            if not values.get("signal_rois"):
                values["signal_rois"] = [values.get("signal_roi", [0.75, 0.05, 0.95, 0.30])]
            if values["signal_rois"]:
                values["signal_roi"] = values["signal_rois"][0]
            if not values.get("stop_lines"):
                legacy_line = values.get("stop_line", [0.05, 0.65, 0.95, 0.65])
                values["stop_lines"] = [legacy_line] * len(values["signal_rois"])
            if values["stop_lines"]:
                values["stop_line"] = values["stop_lines"][0]
        return values

    @model_validator(mode="after")
    def validate_calibration_coordinates(self):
        if len(self.stop_lines) != len(self.signal_rois):
            raise ValueError("Each signal box must have exactly one corresponding stop line")
        coordinates = [*(("stop line", line) for line in self.stop_lines), *[
            (f"signal box {index + 1}", roi)
            for index, roi in enumerate(self.signal_rois)
        ]]
        for label, values in coordinates:
            if len(values) != 4 or any(not isfinite(value) or value < 0 or value > 1 for value in values):
                raise ValueError(f"{label} must contain four normalized coordinates from 0 to 1")
        for index, roi in enumerate(self.signal_rois):
            if roi[2] <= roi[0] or roi[3] <= roi[1]:
                raise ValueError(f"signal box {index + 1} must have positive width and height")
        return self

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
