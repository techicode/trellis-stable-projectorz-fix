import os
import sys
import platform
import torch

# -------------LOW VRAM TESTING -------------


# # only used for debugging, to emulate low-vram graphics cards:
#
# import torch
# torch.cuda.set_per_process_memory_fraction(0.43)  # Limit to 43% of my available VRAM, for testing.
# # And/or set maximum split size (in MB)
# import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'


# -------------- INFO LOGGING ----------------

print(
    f""
    f"[System Info] Python: {platform.python_version():<8} | "
    f"PyTorch: {torch.__version__:<8} | "
    f"CUDA: {'not available' if not torch.cuda.is_available() else torch.version.cuda}"
)

import logging

class TritonFilter(logging.Filter):# Custom filter to ignore Triton messages
    def filter(self, record):
        message = record.getMessage().lower()
        triton_phrases = [
            "triton is not available",
            "matching triton is not available",
            "no module named 'triton'"
        ]
        return not any(phrase in message for phrase in triton_phrases)
    
logger = logging.getLogger("trellis")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent messages from propagating to root
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',  datefmt='%H:%M:%S'))
#and to our own "trellis" logger:
handler.addFilter(TritonFilter())
logger.addHandler(handler)
# other scripts can now use this logger by doing 'logger = logging.getLogger("trellis")'

# Configure root logger:
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S', # Only show time
    handlers=[logging.StreamHandler()]
)
# Apply TritonFilter to all root handlers
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.addFilter(TritonFilter())




# -------------- PIPELINE SETUP ----------------

var_cwd = os.getcwd()
sys.path.append(var_cwd)

print('')
logger.info("Trellis API Server is starting up:")
print('')

# Configure environment, BEFORE including trellis pipeline
os.environ['ATTN_BACKEND'] = 'xformers'    # or 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'       # or 'auto'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # can use to avoid issues with async memory races

# IMPORTING FROM state_manage WILL INITIALIZE THE TRELLIS PIPELINE,
# inside state_manage file. So, ikmporting it only after the above setup:
from api_spz.core.state_manage import state 


# -------------- API SERVER SETUP AND LAUNCH ----------------

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_spz.routes import generation


@asynccontextmanager
async def lifespan(app: FastAPI):
    print('')
    logger.info("Trellis API Server is active and listening.")
    print('')
    yield
    state.cleanup()#shutdown

app = FastAPI(title="Trellis API", lifespan=lifespan)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add the generation router
app.include_router(generation.router)

if __name__ == "__main__":
    uvicorn.run( app,  host="127.0.0.1",
                 port=7960,  
                 log_level="warning" )