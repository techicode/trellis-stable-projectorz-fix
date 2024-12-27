#The current logic in this file has two main paths:
#
#    Initial generation (/generate endpoint):
#    1. Process input image
#    2. Generate 3D structure (20%)
#    3. Process outputs (40%)
#    4. Generate previews (60%)
#    5. If not preview_only:
#       - Save preview videos
#       - Generate GLB file (80%)
#    6. Complete (100%)
#    
#    # Resume generation (/resume endpoint):
#    1. Check task exists and is in PREVIEW_READY state
#    2. Retrieve stored outputs
#    3. Generate GLB file (80%)
#    4. Complete (100%)
import io
import os
import uuid
import base64
from typing import Optional, Literal
from fastapi import APIRouter, File, Query, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import imageio
from pydantic import Field

from core.state_manage import state
from core.models_pydantic import GenerationArg, GenerationResponse, TaskStatus
from core.tasks_manage import task_manager
from core.files_manage import file_manager

from trellis.utils import render_utils, postprocessing_utils

router = APIRouter()

@router.get("/")
async def root():
    """Root endpoint to check server status"""
    return {"status": "running", "message": "Trellis API is operational"}


def gen_3d_validate_params(file, image_base64, arg):
    if file is None and not image_base64:
        raise HTTPException(status_code=400, detail="Either file or base64 image data must be provided")
    # Ensure the values fall inside the min-max range as defined in the app.py and in postprocessing_utils.py:
    if not (0 < arg.ss_guidance_strength <= 10):
        raise HTTPException(status_code=400, detail="SS guidance strength must be above 0 and less than or equal to 10")
    if not (0 < arg.ss_sampling_steps <= 50):
        raise HTTPException(status_code=400, detail="SS sampling steps must be above 0 and cannot exceed 50")
    if not (0 < arg.slat_sampling_steps <= 50):
        raise HTTPException(status_code=400, detail="Slat sampling steps must be above 0 and cannot exceed 50")
    if not (0 < arg.slat_guidance_strength <= 10):
        raise HTTPException(status_code=400, detail="Slat guidance strength must be above 0 and less than or equal to 10")
    if not (15 < arg.preview_frames <= 1000):
        raise HTTPException(status_code=400, detail="Preview frames must be above 15 and cannot exceed 1000")
    if not (0 < arg.mesh_simplify_ratio <= 1):
        raise HTTPException(status_code=400, detail="mesh_simplify_ratio must be between 0 and 1")
    if arg.output_format not in ["glb", "gltf"]:  # Add other supported formats
        raise HTTPException(status_code=400, detail="Unsupported output format")


async def gen_3d_get_image(file: Optional[UploadFile],  image_base64: Optional[str]) -> Image.Image:
    if image_base64: 
        try:# Remove potential data URL prefix:
            if 'base64,' in image_base64:
                image_base64 = image_base64.split('base64,')[1]
            image_data = base64.b64decode(image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 data: {str(e)}")
    else:# Handle file upload:
        image_data = await file.read()
    # Process and validate the image data:
    return Image.open(io.BytesIO(image_data))


async def _process_initial_generation( file: Optional[UploadFile], image_base64: Optional[str], 
                                       arg: GenerationArg = GenerationArg(), ) -> tuple[str, Image.Image, dict]:
    # Common initialization and 3D structure generation.
    # Returns a tuple: (task_id, processed_image, outputs)
    gen_3d_validate_params(file, image_base64, arg)
    task_id = str(uuid.uuid4())
    task_manager.create_task(task_id)
    task_manager.update_task(task_id, 0, "Starting generation...")
    # Process image
    image = await gen_3d_get_image(file, image_base64)
    # Generate 3D structure
    task_manager.update_task(task_id, 30, "Generating 3D structure...")
    ss_params = {
        "steps": arg.ss_sampling_steps,
        "cfg_strength": arg.ss_guidance_strength,
    }
    slat_params = {
        "steps": arg.slat_sampling_steps,
        "cfg_strength": arg.slat_guidance_strength,
    }
    outputs = state.pipeline.run(
        image,
        seed=arg.seed,
        sparse_structure_sampler_params=ss_params,
        slat_sampler_params=slat_params
    )
    return task_id, image, outputs


def _generate_preview_videos( outputs:dict, task_id:str, preview_frames:int, preview_fps:int ) -> dict:
    # Generate and save preview videos, return preview URLs.
    videos = {
        'gaussian': render_utils.render_video(
            outputs['gaussian'][0], 
            resolution=256,
            num_frames=preview_frames
        )['color'],
        'mesh': render_utils.render_video(
            outputs['mesh'][0],
            resolution=256,
            num_frames=preview_frames
        )['normal'],
        'radiance': render_utils.render_video(
            outputs['radiance_field'][0],
            resolution=256,
            num_frames=preview_frames
        )['color']
    }
    for name, video in videos.items():
        preview_path = file_manager.get_temp_path(task_id, f"preview_{name}.mp4")
        imageio.mimsave(str(preview_path), video, fps=preview_fps)
    return {
        "gaussian": f"/download/preview/gaussian/{task_id}",
        "mesh": f"/download/preview/mesh/{task_id}",
        "radiance": f"/download/preview/radiance/{task_id}"
    }


def _generate_glb( task_outputs:dict, task_id:str, mesh_simplify_ratio:float, texture_size:int) -> str:
    # Generate and save GLB file, return model URL.
    glb = postprocessing_utils.to_glb(
        task_outputs['gaussian'][0],
        task_outputs['mesh'][0],
        simplify=mesh_simplify_ratio,
        texture_size=texture_size,
    )
    model_path = file_manager.get_temp_path(task_id, "model.glb")
    glb.export(str(model_path))
    return f"/download/model/{task_id}"


@router.post("/generate_no_preview", response_model=GenerationResponse)
async def generate_no_preview(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    arg: GenerationArg = GenerationArg(),
):
    """Generate 3D model directly without previews"""
    try:
        task_id, _, outputs = await _process_initial_generation(file, image_base64, arg)
        task_manager.update_task(task_id, 70, "Exporting 3D model...")
        model_url = _generate_glb(outputs, task_id, arg.mesh_simplify_ratio, arg.texture_size)

        task_manager.update_task(task_id, 100, "Generation complete", status=TaskStatus.COMPLETE)
        return {
            "task_id": task_id,
            "status": TaskStatus.COMPLETE,
            "model_url": model_url
        }
    except Exception as e:
        if task_id and task_manager.get_task(task_id) is not None:
            task_manager.update_task(task_id, 0, str(e), status=TaskStatus.FAILED)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_preview", response_model=GenerationResponse)
async def generate_preview(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    arg: GenerationArg = GenerationArg(),
): #generate previews, and let user manually invoke 'resume_from_preview()' if they are happy
    try:
        task_id, _, outputs = await _process_initial_generation(file, image_base64, arg)
        # Store outputs for potential resume
        task_manager.update_task(task_id, 50, "Processing outputs...", outputs=outputs)

        task_manager.update_task(task_id, 70, "Generating previews...")
        preview_urls = _generate_preview_videos(outputs, task_id, arg.preview_frames, arg.preview_fps)

        task_manager.update_task(task_id, 100, "Preview generation complete", status=TaskStatus.PREVIEW_READY)
        return {  "task_id": task_id,  "status": TaskStatus.PREVIEW_READY,  "preview_urls": preview_urls  }

    except Exception as e:
        if task_id and task_manager.get_task(task_id) is not None:
            task_manager.update_task(task_id, 0, str(e), status=TaskStatus.FAILED)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume_from_preview/{task_id}", response_model=GenerationResponse)
async def resume_from_preview(
    task_id: str,
    mesh_simplify_ratio: float = Query(0.95, gt=0, le=1),
    texture_size: int = Query(1024, gt=0, le=4096)
):
    """Resume from preview to generate final GLB model"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != TaskStatus.PREVIEW_READY:
        raise HTTPException(status_code=400, detail="Task must be in preview_ready state")
    
    if not task.outputs:
        raise HTTPException(status_code=400, detail="Task outputs not found")
    
    try:
        task_manager.update_task(task_id, 70, "Generating GLB from previews...")
        model_url = _generate_glb(task.outputs, task_id, mesh_simplify_ratio, texture_size)

        task_manager.update_task(task_id, 100, "Generation complete", status=TaskStatus.COMPLETE)
        return {  "task_id": task_id,  "status": TaskStatus.COMPLETE,  "model_url": model_url  }
    except Exception as e:
        task_manager.update_task(task_id, 0, str(e), status=TaskStatus.FAILED)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task.id,
        "status": task.status,
        "progress": task.progress,
        "message": task.message
    }


@router.get("/download/preview/{type}/{task_id}")
async def download_preview(type: Literal["gaussian", "mesh", "radiance"], task_id: str):
    """Download preview video for a specific task and type"""
    preview_path = file_manager.get_temp_path(task_id, f"preview_{type}.mp4")
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(
        str(preview_path), 
        media_type="video/mp4",
        filename=f"preview_{type}_{task_id}.mp4"
    )

@router.get("/download/model/{task_id}")
async def download_model(task_id: str):
    """Download 3D model for a specific task"""
    model_path = file_manager.get_temp_path(task_id, "model.glb")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(
        str(model_path),
        media_type="model/gltf-binary",
        filename=f"model_{task_id}.glb"
    )