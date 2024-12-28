from typing import Optional, Literal, Dict
import asyncio
import io
import base64
import os
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image
import torch
import imageio

from core.state_manage import state
from core.models_pydantic import (
    GenerationArg,
    GenerationResponse,
    TaskStatus,
    StatusResponse
)
from core.files_manage import file_manager

# Trellis pipeline + utils
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

router = APIRouter()

# A single lock to ensure only one generation at a time
generation_lock = asyncio.Lock()
def is_generation_in_progress() -> bool:
    return generation_lock.locked()

print(imageio.help('ffmpeg'))

# A single dictionary holding "current generation" metadata
current_generation = {
    "status": TaskStatus.FAILED, # default
    "progress": 0,
    "message": "",
    "outputs": None,       # pipeline outputs if we did partial gen.
    "preview_urls": None,  # dict of preview paths if relevant.
    "model_url": None      # final model path if relevant.
}


# Helper to reset the "current_generation" dictionary
# (useful to start fresh each time we begin generating)
def reset_current_generation():
    current_generation["status"] = TaskStatus.PROCESSING
    current_generation["progress"] = 0
    current_generation["message"] = ""
    current_generation["outputs"] = None
    current_generation["preview_urls"] = None
    current_generation["model_url"] = None


# Helper to update the "current_generation" dictionary
def update_current_generation(
    status: Optional[TaskStatus] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    outputs=None
):
    if status is not None:
        current_generation["status"] = status
    if progress is not None:
        current_generation["progress"] = progress
    if message is not None:
        current_generation["message"] = message
    if outputs is not None:
        current_generation["outputs"] = outputs


# Cleanup files in "current_generation" folder
async def cleanup_generation_files(keep_videos: bool = False, keep_model: bool = False):
    file_manager.cleanup_generation_files(keep_videos=keep_videos, keep_model=keep_model)


# Validate input
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
        try:
            # Remove potential data URL prefix:
            if "base64," in image_base64:
                image_base64 = image_base64.split("base64,")[1]
            image_data = base64.b64decode(image_base64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 data: {str(e)}"
            )
    else:
        # Handle file upload:
        image_data = await file.read()
    return Image.open(io.BytesIO(image_data))



# Pipeline Worker-like functions
async def _run_pipeline_generate_3d(image_path: Path, arg: GenerationArg):
    """Runs the pipeline in a thread, returns pipeline outputs (in-memory)."""
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


async def _run_pipeline_generate_previews(outputs, preview_frames: int, preview_fps: int):
    """Generate the preview videos in a thread, saves them to disk."""
    def worker():
        videos = {
            "gaussian": render_utils.render_video(
                outputs["gaussian"][0],
                resolution=256,
                num_frames=preview_frames
            )["color"],
            "mesh": render_utils.render_video(
                outputs["mesh"][0],
                resolution=256,
                num_frames=preview_frames
            )["normal"],
            "radiance": render_utils.render_video(
                outputs["radiance_field"][0],
                resolution=256,
                num_frames=preview_frames
            )["color"]
        }
        for name, video in videos.items():
            preview_path = file_manager.get_temp_path(f"preview_{name}.mp4")
            #use the video settings that unity3D can work with:
            imageio.mimsave(str(preview_path),  video,  fps=preview_fps,  codec="libx264",  format="mp4", 
                            pixelformat="yuv420p",  ffmpeg_params=["-profile:v", "baseline", "-level", "3.0"])

    await asyncio.to_thread(worker)


async def _run_pipeline_generate_glb(outputs, mesh_simplify_ratio: float, texture_size: int):
    """Generate the final GLB model in a thread."""
    def worker():
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=mesh_simplify_ratio,
            texture_size=texture_size,
        )
        model_path = file_manager.get_temp_path("model.glb")
        glb.export(str(model_path))

    await asyncio.to_thread(worker)

# --------------------------------------------------
# Routes
# --------------------------------------------------

@router.get("/")
async def root():
    """Root endpoint to check server status."""
    busy = is_generation_in_progress()
    return {
        "status": "running",
        "message": "Trellis API is operational",
        "busy": busy
    }


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get status of the single current/last generation.
    """
    return StatusResponse(
        status=current_generation["status"],
        progress=current_generation["progress"],
        message=current_generation["message"],
        busy=is_generation_in_progress(),
    )


@router.post("/generate_no_preview", response_model=GenerationResponse)
async def generate_no_preview(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    arg: GenerationArg = GenerationArg(),
):
    """
    Generate a 3D model directly (no preview).
    """
    # Acquire the lock (non-blocking)
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Server is busy with another generation"
        )
    # We have the lock => let's reset the "current_generation"
    reset_current_generation()
    try:
        _gen_3d_validate_params(file, image_base64, arg)

        # Save input image
        image = await _gen_3d_get_image(file, image_base64)
        input_image_path = file_manager.get_temp_path("input.png")
        input_image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(input_image_path)

        update_current_generation( status=TaskStatus.PROCESSING, progress=10, message="Generating 3D structure...")
        outputs = await _run_pipeline_generate_3d(input_image_path, arg)
        update_current_generation( progress=50, message="3D structure generated", outputs=outputs)

        # Generate final GLB
        update_current_generation( progress=70, message="Generating GLB file..." )
        await _run_pipeline_generate_glb(outputs, arg.mesh_simplify_ratio, arg.texture_size)

        # Done
        update_current_generation( status=TaskStatus.COMPLETE, progress=100, message="Generation complete")

        # Clean up intermediate files, keep final model
        await cleanup_generation_files(keep_model=True)

        return GenerationResponse(
            status=TaskStatus.COMPLETE,
            progress=100,
            message="Generation complete",
            model_url="/download/model"  # single endpoint
        )
    except Exception as e:
        update_current_generation( status=TaskStatus.FAILED, progress=0, message=str(e))
        await cleanup_generation_files()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
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
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server is busy with another generation" )

    reset_current_generation()

    try:
        _gen_3d_validate_params(file, image_base64, arg)

        # Save input image
        image = await _gen_3d_get_image(file, image_base64)
        input_image_path = file_manager.get_temp_path("input.png")
        input_image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(input_image_path)

        update_current_generation(status=TaskStatus.PROCESSING, progress=20, message="Generating 3D structure...")
        outputs = await _run_pipeline_generate_3d(input_image_path, arg)
        update_current_generation(progress=50, message="3D structure generated", outputs=outputs)

        # Generate Previews
        update_current_generation( progress=60, message="Generating previews..." )
        await _run_pipeline_generate_previews(outputs, arg.preview_frames, arg.preview_fps)

        # Set up preview URLs
        preview_urls = {
            "gaussian": "/download/preview/gaussian",
            "mesh": "/download/preview/mesh",
            "radiance": "/download/preview/radiance",
        }
        update_current_generation(status=TaskStatus.PREVIEW_READY, progress=100, message="Preview generation complete")
        current_generation["preview_urls"] = preview_urls

        # Clean up everything except the preview videos
        await cleanup_generation_files(keep_videos=True)

        return GenerationResponse(
            status=TaskStatus.PREVIEW_READY,
            progress=100,
            message="Preview generation complete",
            preview_urls=preview_urls
        )
    except Exception as e:
        update_current_generation(status=TaskStatus.FAILED, progress=0, message=str(e))
        await cleanup_generation_files()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()


@router.post("/resume_from_preview", response_model=GenerationResponse)
async def resume_from_preview(
    mesh_simplify_ratio: float = Query(0.95, gt=0, le=1),
    texture_size: int = Query(1024, gt=0, le=4096),
):
    """
    Resume from a PREVIEW_READY state, generate final GLB.
    """
    if is_generation_in_progress():
        raise HTTPException(
            status_code=503,
            detail="Server is busy with another generation"
        )
    
    try:# Acquire lock right away
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except:
        raise HTTPException(status_code=503, detail="Server is busy with another generation")

    try:
        if current_generation["status"] != TaskStatus.PREVIEW_READY:
            raise HTTPException(
                status_code=400,
                detail="Current generation must be in preview_ready state to resume"
            )
        outputs = current_generation["outputs"]
        if not outputs:
            raise HTTPException(
                status_code=400,
                detail="No pipeline outputs found in memory"
            )
        update_current_generation( status=TaskStatus.PROCESSING, progress=70, message="Generating final GLB...")
        await _run_pipeline_generate_glb(outputs, mesh_simplify_ratio, texture_size)

        update_current_generation( status=TaskStatus.COMPLETE, progress=100, message="Generation complete")
        await cleanup_generation_files(keep_model=True)# Cleanup everything except final model

        return GenerationResponse(
            status=TaskStatus.COMPLETE,
            progress=100,
            message="Generation complete",
            model_url="/download/model"
        )
    except Exception as e:
        update_current_generation(
            status=TaskStatus.FAILED,
            progress=0,
            message=str(e)
        )
        await cleanup_generation_files()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()


@router.get("/download/preview/{type}")
async def download_preview(
    type: Literal["gaussian", "mesh", "radiance"]
):
    """Download the preview video for the current generation."""
    preview_path = file_manager.get_temp_path(f"preview_{type}.mp4")
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(
        str(preview_path),
        media_type="video/mp4",
        filename=f"preview_{type}.mp4"
    )


@router.get("/download/model")
async def download_model():
    """Download final 3D model (GLB)."""
    model_path = file_manager.get_temp_path("model.glb")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(
        str(model_path),
        media_type="model/gltf-binary",
        filename="model.glb"
    )
