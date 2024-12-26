import io
import os
import uuid
import base64
from typing import Optional, Literal
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import imageio

from ..core.state_manage import state
from ..core.models_pydantic import GenerationParams, GenerationResponse, TaskStatus
from ..core.tasks_manage import task_manager
from ..core.files_manage import file_manager

from trellis.utils import render_utils, postprocessing_utils

router = APIRouter()

@router.get("/")
async def root():
    """Root endpoint to check server status"""
    return {"status": "running", "message": "Trellis API is operational"}


@router.post("/generate", response_model=GenerationResponse)
async def generate_3d(file: Optional[UploadFile] = File(None),
                      image_base64: Optional[str] = Form(None),
                      params: GenerationParams = GenerationParams()):
    task_id = None
    try:
        # Validate that we have either a file or base64 data
        if file is None and not image_base64:
            raise HTTPException(status_code=400, detail="Either file or base64 image data must be provided")

        # Validate parameters
        if params.ss_sampling_steps > 50 or params.slat_sampling_steps > 50:
            raise HTTPException(status_code=400, detail="Sampling steps cannot exceed 50")
        if params.preview_frames > 1000:
            raise HTTPException(status_code=400, detail="Preview frames cannot exceed 1000")
        if params.output_format not in ["glb", "gltf"]:  # add other supported formats
            raise HTTPException(status_code=400, detail="Unsupported output format")
        if not (0 < params.mesh_simplify_ratio <= 1):
            raise HTTPException(status_code=400, detail="mesh_simplify_ratio must be between 0 and 1")
        
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        task_manager.create_task(task_id)
        task_manager.update_task_status(task_id, 0, "Starting generation...")

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
        image = Image.open(io.BytesIO(image_data))

        # Save the processed image
        image_path = os.path.join(task_dir, 'input.png')
        image.save(image_path)
        
        # Clean up old files
        file_manager.cleanup_old_files()

        # Generate 3D model
        task_manager.update_task(task_id, 20, "Generating 3D structure...")
        outputs = state.pipeline.run(
            image,
            seed=params.seed,
            sparse_structure_sampler_params={
                "steps": params.ss_sampling_steps,
                "cfg_strength": params.ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": params.slat_sampling_steps,
                "cfg_strength": params.slat_guidance_strength,
            },
        )

        # Store outputs for potential resume
        task_manager.update_task(task_id, 40, "Processing outputs...", outputs=outputs)

        # Initialize response with common fields
        response = {
            "task_id": task_id,
        }

        # Generate previews (unless explicitly disabled)
        if not params.preview_only:
            task_manager.update_task(task_id, 60, "Generating previews...")
            
            videos = {
                'gaussian': render_utils.render_video(outputs['gaussian'][0], 
                                                      num_frames=params.preview_frames)['color'],
                'mesh': render_utils.render_video(outputs['mesh'][0], 
                                                  num_frames=params.preview_frames)['normal'],
                'radiance': render_utils.render_video(outputs['radiance_field'][0], 
                                                      num_frames=params.preview_frames)['color']
            }
            
            for name, video in videos.items():
                preview_path = file_manager.get_temp_path(task_id, f"preview_{name}.mp4")
                imageio.mimsave(str(preview_path), video, fps=params.preview_fps)

            response["preview_urls"] = {
                "gaussian": f"/download/preview/gaussian/{task_id}",
                "mesh": f"/download/preview/mesh/{task_id}",
                "radiance": f"/download/preview/radiance/{task_id}",
            }

        # If preview_only mode, return after preview generation
        if params.preview_only:
            task_manager.update_task(task_id, 100, "Preview generation complete", status="preview_ready")
            response["status"] = "preview_ready"
            return response

        # Generate 3D model file
        task_manager.update_task(task_id, 80, "Exporting 3D model...")
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=params.mesh_simplify_ratio,
            texture_size=params.texture_size,
        )
        model_path = file_manager.get_temp_path(task_id, "model.glb")
        glb.export(str(model_path))

        # Update final status and add model URL
        task_manager.update_task(task_id, 100, "Generation complete", status="complete")
        response["status"] = "complete"
        response["model_url"] = f"/download/model/{task_id}"
        return response

    except Exception as e:
        if task_id and task_manager.get_task(task_id) != None:
            task_manager.update_task(task_id, 0, str(e), status="failed")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@router.post("/resume/{task_id}", response_model=GenerationResponse)
async def resume_generation(task_id: str,
                            mesh_simplify_ratio: Optional[float] = 0.95,
                            texture_size: Optional[int] = 1024):
    """Resume generation after preview to create the final model"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != TaskStatus.PREVIEW_READY:
        raise HTTPException(status_code=400, detail="Task not in preview_ready state")
    
    try:
        outputs = task["outputs"]
        task_manager.update_task(task_id, 80, "Exporting 3D model...")
        
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=mesh_simplify_ratio,
            texture_size=texture_size,
        )
        model_path = file_manager.get_temp_path(task_id, "model.glb")
        glb.export(str(model_path))

        task_manager.update_task(task_id, 100, "Generation complete", status="complete")

        return {
            "status": "success",
            "task_id": task_id,
            "model_url": f"/download/model/{task_id}"
        }
    except Exception as e:
        task_manager.update_task(task_id, 0, str(e), status="failed")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get status of a specific task"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.status


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