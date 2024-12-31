import os 
import sys
import time
from pathlib import Path

# Add trellis to Python path 
sys.path.append(os.getcwd())

# Configure environment
os.environ['ATTN_BACKEND'] = 'xformers' # Can be 'flash-attn' or 'xformers'
os.environ['SPCONV_ALGO'] = 'native'    # Can be 'native' or 'auto'

# Global variables
pipeline = None
TASKS = {}  # Store task status and data

def update_task_status(task_id: str, progress: int, message: str, status: str = "processing", outputs=None):
    """Update task status and optionally store outputs in the global TASKS dictionary"""
    TASKS[task_id] = {
        "status": status,
        "progress": progress,
        "message": message,
        "outputs": outputs
    }
