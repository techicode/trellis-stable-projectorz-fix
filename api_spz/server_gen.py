import os 
import sys
import io
import base64
import uuid
from typing import Optional, Literal
from pathlib import Path
from fastapi import File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import imageio

# Add trellis to Python path 
sys.path.append(os.getcwd())

# Configure environment
os.environ['ATTN_BACKEND'] = 'xformers'    # Can be 'flash-attn' or 'xformers'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto'
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

from server_utils import TASKS, pipeline
from server_utils import cleanup_old_files, update_task_status, get_temp_path
