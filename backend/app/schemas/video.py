from typing import Optional
from enum import Enum
from pydantic import BaseModel

class JobStatus(str, Enum):
    IDLE = "IDLE"
    UPLOADING = "UPLOADING"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class JobStatusResponse(BaseModel):
    job_id: str
    source_filename: str
    status: JobStatus
    progress: float
    current_frame: int
    total_frames: int
    fps: float
    annotated_video_path: Optional[str] = None
    violations_count: int = 0
    error_message: Optional[str] = None
    created_at: str
    updated_at: str

class VideoUploadResponse(BaseModel):
    job_id: str
    filename: str
    file_size_bytes: int
    message: str
