from typing import Optional, Literal, Dict
import asyncio
import io
import os
import uuid
import base64
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image
import torch
import imageio

from core.state_manage import state
from core.models_pydantic import GenerationArg, GenerationResponse, TaskStatus, CurrentTaskResponse
from core.tasks_manage import task_manager
from core.files_manage import file_manager

# Trellis pipeline + utils
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

router = APIRouter()

current_task_id = None

# Global lock to enforce only one generation at a time
generation_lock = asyncio.Lock()

def is_generation_in_progress() -> bool:
    return generation_lock.locked()

# -------------
# Helper: Cleanup files
# -------------
async def cleanup_task_files(task_id: str, keep_videos: bool = False, keep_model: bool = False):
    """Clean up temporary files for a task."""
    try:
        temp_files = ["input.png"]
        if not keep_videos:
            temp_files.extend([
                "preview_gaussian.mp4",
                "preview_mesh.mp4",
                "preview_radiance.mp4"
            ])
        if not keep_model:
            temp_files.append("model.glb")
            
        for file_name in temp_files:
            file_path = file_manager.get_temp_path(task_id, file_name)
            if file_path.exists():
                os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up task files: {e}")

# -------------
# Validation
# -------------
def _gen_3d_validate_params(file, image_base64, arg: GenerationArg):
    """Validate incoming parameters before generation."""
    if file is None and not image_base64:
        raise HTTPException(status_code=400, detail="Either file or base64 image data must be provided")
    # Range checks:
    if not (0 < arg.ss_guidance_strength <= 10):
        raise HTTPException(status_code=400, detail="SS guidance strength must be above 0 and <= 10")
    if not (0 < arg.ss_sampling_steps <= 50):
        raise HTTPException(status_code=400, detail="SS sampling steps must be above 0 and <= 50")
    if not (0 < arg.slat_sampling_steps <= 50):
        raise HTTPException(status_code=400, detail="Slat sampling steps must be above 0 and <= 50")
    if not (0 < arg.slat_guidance_strength <= 10):
        raise HTTPException(status_code=400, detail="Slat guidance strength must be above 0 and <= 10")
    if not (15 < arg.preview_frames <= 1000):
        raise HTTPException(status_code=400, detail="Preview frames must be above 15 and <= 1000")
    if not (0 < arg.mesh_simplify_ratio <= 1):
        raise HTTPException(status_code=400, detail="mesh_simplify_ratio must be between 0 and 1")
    if arg.output_format not in ["glb", "gltf"]:
        raise HTTPException(status_code=400, detail="Unsupported output format")
    

async def _gen_3d_get_image(file: Optional[UploadFile], image_base64: Optional[str]) -> Image.Image:
    if image_base64:
        try:# Remove potential data URL prefix:
            if 'base64,' in image_base64:
                image_base64 = image_base64.split('base64,')[1]
            image_data = base64.b64decode(image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 data: {str(e)}")
    else:
        # Handle file upload:
        image_data = await file.read()
    # Process and validate the image data:
    return Image.open(io.BytesIO(image_data))


# -------------
# Worker-like Functions (but run in-thread, to avoid serialization issues between them)
# -------------
async def _run_pipeline_generate_3d(image_path: Path, arg: GenerationArg):
    """Runs the pipeline in a thread, returns the pipeline's outputs (in-memory)."""
    def worker():
        pipeline = state.pipeline
        image = Image.open(image_path)
        outputs = pipeline.run(
            image,
            seed=arg.seed,
            sparse_structure_sampler_params={
                "steps": arg.ss_sampling_steps,
                "cfg_strength": arg.ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": arg.slat_sampling_steps,
                "cfg_strength": arg.slat_guidance_strength,
            },
        )
        torch.cuda.empty_cache()
        return outputs

    outputs = await asyncio.to_thread(worker)
    return outputs


async def _run_pipeline_generate_previews(outputs, preview_frames: int, preview_fps: int, task_id: str):
    """Given pipeline outputs, generate previews (videos) in a thread."""
    def worker():
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

    await asyncio.to_thread(worker)


async def _run_pipeline_generate_glb(outputs, mesh_simplify_ratio: float, texture_size: int, task_id: str):
    """Given pipeline outputs, generate the final GLB model in a thread."""
    def worker():
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=mesh_simplify_ratio,
            texture_size=texture_size,
        )
        model_path = file_manager.get_temp_path(task_id, "model.glb")
        glb.export(str(model_path))

    await asyncio.to_thread(worker)

# -------------
# Routes
# -------------
@router.get("/")
async def root():
    """Root endpoint to check server status."""
    busy = is_generation_in_progress()
    return {
        "status": "running",
        "message": "Trellis API is operational",
        "busy": busy
    }


@router.get("/current_task_id", response_model=CurrentTaskResponse)
async def get_current_task_id():
    """
    Retrieve the ID of the most recent generation, if any.
    This will allow you to query the progress while the generation still hasn't returned.
    """
    return CurrentTaskResponse(current_task_id=current_task_id)


@router.post("/generate_no_preview", response_model=GenerationResponse)
async def generate_no_preview(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    arg: GenerationArg = GenerationArg(),
):
    """
    Generate 3D model directly (no preview).
    """
    # Acquire the lock (non-blocking)
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server is busy with another generation task")
    
    task_id = str(uuid.uuid4())
    task_manager.create_task(task_id)

    global current_task_id
    current_task_id = task_id

    try:
        # Validate
        _gen_3d_validate_params(file, image_base64, arg)

        # Get the image
        image = await _gen_3d_get_image(file, image_base64)
        temp_image_path = file_manager.get_temp_path(task_id, "input.png")
        temp_image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(temp_image_path)

        # Start 3D generation
        task_manager.update_task(task_id, 20, "Generating 3D structure...")
        outputs = await _run_pipeline_generate_3d(temp_image_path, arg)
        task_manager.update_task(task_id, 50, "3D structure generated", outputs=outputs)

        # Generate GLB
        task_manager.update_task(task_id, 70, "Generating GLB file...")
        await _run_pipeline_generate_glb(outputs, arg.mesh_simplify_ratio, arg.texture_size, task_id)

        # Done
        task_manager.update_task(task_id, 100, "Generation complete", status=TaskStatus.COMPLETE)
        # Clean up everything except the final model
        await cleanup_task_files(task_id, keep_model=True)

        return GenerationResponse(
            task_id=task_id,
            status=TaskStatus.COMPLETE,
            progress=100,
            message="Generation complete",
            model_url=f"/download/model/{task_id}"
        )
    except Exception as e:
        task_manager.update_task(task_id, 0, str(e), status=TaskStatus.FAILED)
        await cleanup_task_files(task_id)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        current_task_id = None
        generation_lock.release()


@router.post("/generate_preview", response_model=GenerationResponse)
async def generate_preview(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    arg: GenerationArg = GenerationArg(),
):
    """
    Generate partial 3D structure + Previews, let user resume with /resume_from_preview
    """
    # Acquire the lock (non-blocking)
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server is busy with another generation task")
    
    task_id = str(uuid.uuid4())
    task_manager.create_task(task_id)

    global current_task_id 
    current_task_id = task_id

    try:
        # Validate
        _gen_3d_validate_params(file, image_base64, arg)

        # Get the image
        image = await _gen_3d_get_image(file, image_base64)
        temp_image_path = file_manager.get_temp_path(task_id, "input.png")
        temp_image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(temp_image_path)

        # Start 3D generation
        task_manager.update_task(task_id, 20, "Generating 3D structure...")
        outputs = await _run_pipeline_generate_3d(temp_image_path, arg)
        task_manager.update_task(task_id, 50, "3D structure generated", outputs=outputs)

        # Generate Previews
        task_manager.update_task(task_id, 60, "Generating previews...")
        await _run_pipeline_generate_previews(outputs, arg.preview_frames, arg.preview_fps, task_id)

        preview_urls = {
            "gaussian": f"/download/preview/gaussian/{task_id}",
            "mesh": f"/download/preview/mesh/{task_id}",
            "radiance": f"/download/preview/radiance/{task_id}",
        }

        task_manager.update_task(
            task_id,
            100,
            "Preview generation complete",
            status=TaskStatus.PREVIEW_READY,
        )

        # Keep the outputs in memory for resume
        await cleanup_task_files(task_id, keep_videos=True)

        return GenerationResponse(
            task_id=task_id,
            status=TaskStatus.PREVIEW_READY,
            progress=100,
            message="Preview generation complete",
            preview_urls=preview_urls
        )

    except Exception as e:
        task_manager.update_task(task_id, 0, str(e), status=TaskStatus.FAILED)
        await cleanup_task_files(task_id)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()
        current_task_id = None


@router.post("/resume_from_preview/{task_id}", response_model=GenerationResponse)
async def resume_from_preview(
    task_id: str,
    mesh_simplify_ratio: float = Query(0.95, gt=0, le=1),
    texture_size: int = Query(1024, gt=0, le=4096),
):
    """
    Resume from a PREVIEW_READY task, generate final GLB.
    """
    if generation_lock.locked():
        raise HTTPException(status_code=503, detail="Server is busy with another generation task")

    try:
        generation_lock.acquire_nowait()
    except:
        raise HTTPException(status_code=503, detail="Server is busy with another generation task")

    try:
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if task.status != TaskStatus.PREVIEW_READY:
            raise HTTPException(
                status_code=400,
                detail="Task must be in preview_ready state"
            )
        outputs = task.outputs
        if not outputs:
            raise HTTPException(status_code=400, detail="No pipeline outputs found in memory for this task")
        
        global current_task_id
        current_task_id = task_id

        # Generate final GLB
        task_manager.update_task(task_id, 70, "Generating GLB file...")
        await _run_pipeline_generate_glb(outputs, mesh_simplify_ratio, texture_size, task_id)

        task_manager.update_task(task_id, 100, "Generation complete", status=TaskStatus.COMPLETE)
        await cleanup_task_files(task_id, keep_model=True)

        return GenerationResponse(
            task_id=task_id,
            status=TaskStatus.COMPLETE,
            progress=100,
            message="Generation complete",
            model_url=f"/download/model/{task_id}"
        )
    except Exception as e:
        task_manager.update_task(task_id, 0, str(e), status=TaskStatus.FAILED)
        await cleanup_task_files(task_id)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()
        current_task_id = None


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get status of a generation task"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task.id,
        "status": task.status,
        "progress": task.progress,
        "message": task.message,
        "busy": is_generation_in_progress(),
    }


@router.get("/download/preview/{type}/{task_id}")
async def download_preview(
    type: Literal["gaussian", "mesh", "radiance"],
    task_id: str
):
    """Download a preview video for a given task."""
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
    """Download final 3D model for a specific task (GLB)."""
    model_path = file_manager.get_temp_path(task_id, "model.glb")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(
        str(model_path),
        media_type="model/gltf-binary",
        filename=f"model_{task_id}.glb"
    )
