import os 
import sys
import shutil
from typing import Optional, Literal
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add trellis to Python path 
sys.path.append(os.getcwd()+"/..")


from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

from server_utils import TEMP_DIR, TASKS
from server_utils import get_temp_path
from server_gen import generate_3d, resume_generation
