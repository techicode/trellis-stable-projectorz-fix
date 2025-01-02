import os
import sys

import logging
logger = logging.getLogger("trellis")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
# other scripts can now use this logger by doing 'logger = logging.getLogger("trellis")'

var_cwd = os.getcwd()
sys.path.append(var_cwd)

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

print('')
logger.info("Trellis API Server is starting up:")
print('')

# Configure environment, BEFORE including trellis pipeline
os.environ['ATTN_BACKEND'] = 'xformers'    # or 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'       # or 'auto'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # can use to avoid issues with async memory races

from api_spz.core.state_manage import state # this will initialize the trellis pipeline.
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