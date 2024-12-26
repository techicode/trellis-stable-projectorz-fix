import os 
import sys
import io
import base64
import uuid
import time
import shutil
from typing import Optional, Literal
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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

# Initialize FastAPI with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    TEMP_DIR.mkdir(exist_ok=True)
    initialize_pipeline()
    yield
    # Shutdown 
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

app = FastAPI(title="Trellis API", lifespan=lifespan)

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
TASKS = {}  # Store task status and data
TEMP_DIR = Path("temp")

def initialize_pipeline():
    """Initialize the Trellis pipeline"""
    global pipeline
    if pipeline is None:
        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()
    return pipeline

def update_task_status(task_id: str, progress: int, message: str, status: str = "processing", outputs=None):
    """Update task status and optionally store outputs in the global TASKS dictionary"""
    TASKS[task_id] = {
        "status": status,
        "progress": progress,
        "message": message,
        "outputs": outputs
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


def process_image_data(image_data: bytes) -> Image.Image:
    """Process raw image data into a PIL Image"""
    try:
        if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
            
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@app.post("/generate")
async def generate_3d(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    seed: int = 1,
    ss_guidance_strength: float = 7.5,
    ss_sampling_steps: int = 12,
    slat_guidance_strength: float = 3.0,
    slat_sampling_steps: int = 12,
    preview_frames: int = 300,
    preview_fps: int = 30,
    mesh_simplify_ratio: float = 0.95,
    texture_size: int = 1024,
    output_format: str = "glb",
    preview_only: bool = False  # Renamed from skip_preview for clarity
):
    task_id = None
    try:
        # Validate that we have either a file or base64 data
        if file is None and not image_base64:
            raise HTTPException(status_code=400, detail="Either file or base64 image data must be provided")

        # Validate parameters
        if ss_sampling_steps > 50 or slat_sampling_steps > 50:
            raise HTTPException(status_code=400, detail="Sampling steps cannot exceed 50")
        if preview_frames > 1000:
            raise HTTPException(status_code=400, detail="Preview frames cannot exceed 1000")
        if output_format not in ["glb", "gltf"]:  # add other supported formats
            raise HTTPException(status_code=400, detail="Unsupported output format")
        if not (0 < mesh_simplify_ratio <= 1):
            raise HTTPException(status_code=400, detail="mesh_simplify_ratio must be between 0 and 1")
        
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        update_task_status(task_id, 0, "Starting generation...")

        # Create task directory if it doesn't exist
        task_dir = os.path.join('tasks', task_id)
        os.makedirs(task_dir, exist_ok=True)

        # Process input image
        if file:
            # Handle file upload
            image_data = await file.read()
        else:
            # Handle base64 input
            try:
                # Remove potential data URL prefix
                if 'base64,' in image_base64:
                    image_base64 = image_base64.split('base64,')[1]
                image_data = base64.b64decode(image_base64)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 data: {str(e)}")

        # Process and validate the image data
        image = process_image_data(image_data)

        # Save the processed image
        image_path = os.path.join(task_dir, 'input.png')
        image.save(image_path)
        
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

        # Store outputs for potential resume
        update_task_status(task_id, 40, "Processing outputs...", outputs=outputs)

        # Initialize response with common fields
        response = {
            "task_id": task_id,
        }

        # Generate previews (unless explicitly disabled)
        if not preview_only:
            update_task_status(task_id, 60, "Generating previews...")
            
            videos = {
                'gaussian': render_utils.render_video(outputs['gaussian'][0], num_frames=preview_frames)['color'],
                'mesh': render_utils.render_video(outputs['mesh'][0], num_frames=preview_frames)['normal'],
                'radiance': render_utils.render_video(outputs['radiance_field'][0], num_frames=preview_frames)['color']
            }
            
            for name, video in videos.items():
                preview_path = get_temp_path(task_id, f"preview_{name}.mp4")
                imageio.mimsave(str(preview_path), video, fps=preview_fps)

            response["preview_urls"] = {
                "gaussian": f"/download/preview/gaussian/{task_id}",
                "mesh": f"/download/preview/mesh/{task_id}",
                "radiance": f"/download/preview/radiance/{task_id}",
            }

        # If preview_only mode, return after preview generation
        if preview_only:
            update_task_status(task_id, 100, "Preview generation complete", status="preview_ready")
            response["status"] = "preview_ready"
            return response

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

        # Update final status and add model URL
        update_task_status(task_id, 100, "Generation complete", status="complete")
        response["status"] = "complete"
        response["model_url"] = f"/download/model/{task_id}"

        return response

    except Exception as e:
        if task_id and task_id in TASKS:
            update_task_status(task_id, 0, str(e), status="failed")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
    

@app.post("/resume/{task_id}")
async def resume_generation(
    task_id: str,
    mesh_simplify_ratio: Optional[float] = 0.95,
    texture_size: Optional[int] = 1024,
):
    """Resume generation after preview to create the final model"""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = TASKS[task_id]
    if task["status"] != "preview_ready":
        raise HTTPException(status_code=400, detail="Task not in preview_ready state")
    
    try:
        outputs = task["outputs"]
        update_task_status(task_id, 80, "Exporting 3D model...")
        
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=mesh_simplify_ratio,
            texture_size=texture_size,
        )
        model_path = get_temp_path(task_id, "model.glb")
        glb.export(str(model_path))

        update_task_status(task_id, 100, "Generation complete", status="complete")

        return {
            "status": "success",
            "task_id": task_id,
            "model_url": f"/download/model/{task_id}"
        }

    except Exception as e:
        update_task_status(task_id, 0, str(e), status="failed")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/download/preview/{type}/{task_id}")
async def download_preview(type: Literal["gaussian", "mesh", "radiance"], task_id: str):
    """Download preview video for a specific task and type"""
    preview_path = get_temp_path(task_id, f"preview_{type}.mp4")
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(
        str(preview_path), 
        media_type="video/mp4",
        filename=f"preview_{type}_{task_id}.mp4"
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