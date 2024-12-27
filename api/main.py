import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' #MODIF using for now to avoid issues with async memory races

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure environment, BEFORE including trellis pipeline
os.environ['ATTN_BACKEND'] = 'xformers' # Can be 'flash-attn' or 'xformers'
os.environ['SPCONV_ALGO']  = 'native'    # Can be 'native' or 'auto'
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

app.include_router(generation.router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7861)