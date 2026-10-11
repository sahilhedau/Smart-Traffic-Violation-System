import os
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel, Field

# Root directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class SystemConfig(BaseModel):
    # Server settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = False

    # Paths
    BASE_DIR: Path = BASE_DIR
    MODELS_DIR: Path = BASE_DIR / "models"
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"
    EVIDENCE_DIR: Path = BASE_DIR / "outputs" / "evidence"
    ANNOTATED_VIDEOS_DIR: Path = BASE_DIR / "outputs" / "annotated_videos"
    UPLOADS_DIR: Path = BASE_DIR / "outputs" / "uploads"
    SAMPLES_DIR: Path = BASE_DIR / "data" / "samples"
    DATABASE_PATH: Path = BASE_DIR / "traffic_system.db"

    # Vision & Model Parameters
    DEVICE: str = "auto"  # 'auto', 'cuda', 'cpu'
    YOLO_MODEL_NAME: str = "yolov8n.pt"
    HELMET_MODEL_NAME: str = "helmet_yolov8n.pt"  # optional specialized or baseline
    PLATE_MODEL_NAME: str = "plate_yolov8n.pt"    # optional specialized or baseline
    CONFIDENCE_THRESHOLD: float = 0.35
    IOU_THRESHOLD: float = 0.45
    FRAME_SKIP: int = 1
    MAX_PROCESSING_FPS: int = 30
    TARGET_WIDTH: int = 1280
    TARGET_HEIGHT: int = 720

    # Violation feature toggles
    ENABLE_SIGNAL_JUMP: bool = True
    ENABLE_TRIPLE_RIDING: bool = True
    ENABLE_HELMET_CHECK: bool = True
    ENABLE_PLATE_OCR: bool = True
    PRIVACY_BLUR: bool = False

    # Red Light / Stop Line Calibration
    # Line format: [x1, y1, x2, y2] in normalized coordinates (0.0 to 1.0)
    STOP_LINE: List[float] = [0.05, 0.65, 0.95, 0.65]
    # Signal ROI: [x1, y1, x2, y2]
    SIGNAL_ROI: List[float] = [0.75, 0.05, 0.95, 0.30]
    # Signal mode: 'auto_detect', 'red', 'yellow', 'green'
    SIGNAL_MODE: str = "red"

    # Triple Riding settings
    MIN_RIDERS_FOR_ALERT: int = 3
    TRIPLE_RIDING_MIN_FRAMES: int = 3

    # Helmet settings
    HELMET_MIN_CONFIDENCE: float = 0.40
    HELMET_SMOOTHING_FRAMES: int = 3

    # Plate OCR settings
    OCR_MIN_CONFIDENCE: float = 0.30
    OCR_ENGINE: str = "easyocr"  # 'easyocr'

    def ensure_directories(self):
        """Ensure all runtime directories exist."""
        for directory in [
            self.MODELS_DIR,
            self.OUTPUTS_DIR,
            self.EVIDENCE_DIR,
            self.ANNOTATED_VIDEOS_DIR,
            self.UPLOADS_DIR,
            self.SAMPLES_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

# Global configuration instance
config = SystemConfig()
config.ensure_directories()
