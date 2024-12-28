from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PROCESSING = "processing"
    PREVIEW_READY = "preview_ready"
    COMPLETE = "complete"
    FAILED = "failed"


class GenerationArg(BaseModel):
    seed: int = 1
    ss_guidance_strength: float = Field(7.5, gt=0, le=10)
    ss_sampling_steps: int = Field(12, gt=0, le=50)
    slat_guidance_strength: float = Field(3.0, gt=0, le=10)
    slat_sampling_steps: int = Field(12, gt=0, le=50)
    preview_frames: int = Field(300, gt=15, le=1000)
    preview_fps: int = Field(30, gt=0, le=60)
    mesh_simplify_ratio: float = Field(0.95, gt=0, le=1)
    texture_size: int = Field(1024, gt=0, le=4096)
    output_format: str = "glb"


class GenerationResponse(BaseModel):
    # No task_id anymore, we focus on a single generation
    status: TaskStatus
    progress: int = 0
    message: str = ""
    # Only used if we did “generate_preview”
    preview_urls: Optional[Dict[str, str]] = None
    # Only used if generation is complete
    model_url: Optional[str] = None


class StatusResponse(BaseModel):
    status: TaskStatus
    progress: int
    message: str
    busy: bool
