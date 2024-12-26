import os 
import sys
import io
import uuid
import shutil
from typing import Optional, Literal
from pathlib import Path
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import torch
import imageio

# Add trellis to Python path 
sys.path.append(os.getcwd())

# Configure environment
os.environ['ATTN_BACKEND'] = 'xformers'    # Can be 'flash-attn' or 'xformers'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto'

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    TEMP_DIR.mkdir(exist_ok=True)
    initialize_pipeline()
    yield
    # Shutdown 
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

app = FastAPI(lifespan=lifespan)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
pipeline = None
TASKS = {}  # Store task status
TEMP_DIR = Path("temp")

def initialize_pipeline():
    """Initialize the Trellis pipeline"""
    global pipeline
    if pipeline is None:
        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()
    return pipeline

def update_task_status(task_id: str, progress: int, message: str, status: str = "processing"):
    """Update task status in the global TASKS dictionary"""
    TASKS[task_id] = {
        "status": status,
        "progress": progress,
        "message": message
    }

def get_temp_path(task_id: str, filename: str) -> Path:
    """Get path for temporary file with task ID prefix"""
    return TEMP_DIR / f"{task_id}_{filename}"

def cleanup_old_files():
    """Clean up files older than 1 hour"""
    if TEMP_DIR.exists():
        for file in TEMP_DIR.iterdir():
            if file.stat().st_mtime < (time.time() - 3600):  # 1 hour
                file.unlink()

@app.on_event("startup")
async def startup_event():
    """Initialize on server startup"""
    TEMP_DIR.mkdir(exist_ok=True)
    initialize_pipeline()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

@app.get("/")
async def root():
    """Root endpoint to check server status"""
    return {"status": "running", "message": "Trellis API is operational"}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get status of a specific task"""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    return TASKS[task_id]

@app.post("/generate")
async def generate_3d(
    file: UploadFile,
    seed: Optional[int] = 1,
    ss_guidance_strength: Optional[float] = 7.5,
    ss_sampling_steps: Optional[int] = 12,
    slat_guidance_strength: Optional[float] = 3,
    slat_sampling_steps: Optional[int] = 12,
    preview_frames: Optional[int] = 300,
    preview_fps: Optional[int] = 30,
    mesh_simplify_ratio: Optional[float] = 0.95,
    texture_size: Optional[int] = 1024,
    output_format: Optional[Literal["glb", "obj", "both"]] = "glb",
):
    """Generate 3D model from image"""
    try:
        # Create task ID and initialize status
        task_id = str(uuid.uuid4())
        update_task_status(task_id, 0, "Starting generation...")

        # Validate and process input image
        image_data = await file.read()
        if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Validate parameters
        if ss_sampling_steps > 50 or slat_sampling_steps > 50:
            raise HTTPException(status_code=400, detail="Sampling steps cannot exceed 50")
        if preview_frames > 1000:
            raise HTTPException(status_code=400, detail="Preview frames cannot exceed 1000")
        
        # Clean up old files
        cleanup_old_files()

        # Generate 3D model
        update_task_status(task_id, 20, "Generating 3D structure...")
        outputs = pipeline.run(
            image,
            seed=seed,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )

        # Generate preview video
        update_task_status(task_id, 60, "Generating preview...")
        video = render_utils.render_video(
            outputs['gaussian'][0], 
            num_frames=preview_frames
        )['color']
        preview_path = get_temp_path(task_id, "preview.mp4")
        imageio.mimsave(str(preview_path), video, fps=preview_fps)

        # Generate 3D model file
        update_task_status(task_id, 80, "Exporting 3D model...")
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=mesh_simplify_ratio,
            texture_size=texture_size,
        )
        model_path = get_temp_path(task_id, "model.glb")
        glb.export(str(model_path))

        # Update final status
        update_task_status(task_id, 100, "Generation complete", status="complete")

        return {
            "status": "success",
            "task_id": task_id,
            "preview_url": f"/download/preview/{task_id}",
            "model_url": f"/download/model/{task_id}"
        }
    except Exception as e:
        if task_id in TASKS:
            update_task_status(task_id, 0, str(e), status="failed")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/download/preview/{task_id}")
async def download_preview(task_id: str):
    """Download preview video for a specific task"""
    preview_path = get_temp_path(task_id, "preview.mp4")
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(
        str(preview_path), 
        media_type="video/mp4",
        filename=f"preview_{task_id}.mp4"
    )

@app.get("/download/model/{task_id}")
async def download_model(task_id: str):
    """Download 3D model for a specific task"""
    model_path = get_temp_path(task_id, "model.glb")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(
        str(model_path),
        media_type="model/gltf-binary",
        filename=f"model_{task_id}.glb"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7861)