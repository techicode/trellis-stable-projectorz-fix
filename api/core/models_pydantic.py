from enum import Enum
from typing import Optional, Dict, List
from pydantic import BaseModel, Field

class TaskStatus(str, Enum):
    PROCESSING = "processing"
    PREVIEW_READY = "preview_ready"
    COMPLETE = "complete"
    FAILED = "failed"

class GenerationArg(BaseModel):
    seed: int = 1
    ss_guidance_strength: float = Field(7.5, gt=0)
    ss_sampling_steps: int = Field(12, gt=0, le=50)
    slat_guidance_strength: float = Field(3.0, gt=0)
    slat_sampling_steps: int = Field(12, gt=0, le=50)
    preview_frames: int = Field(300, gt=0, le=1000)
    preview_fps: int = Field(30, gt=0, le=60)
    mesh_simplify_ratio: float = Field(0.95, gt=0, le=1)
    texture_size: int = Field(1024, gt=0)
    output_format: str = "glb"

class GenerationResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int = 0
    message: str = ""
    preview_urls: Optional[Dict[str, str]] = None
    model_url: Optional[str] = None

class StatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int
    message: str