import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Using for now to avoid issues with async memory races

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure environment, BEFORE including trellis pipeline
os.environ['ATTN_BACKEND'] = 'xformers'  # or 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'    # or 'auto'

from core.state_manage import state
from routes import generation

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    state.initialize_pipeline()
    yield
    # Shutdown
    state.cleanup()

app = FastAPI(title="Trellis API", lifespan=lifespan)

# Add CORS middleware
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
    uvicorn.run(app, host="127.0.0.1", port=7861)
