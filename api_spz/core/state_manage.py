import torch
from pathlib import Path
from trellis.pipelines import TrellisImageTo3DPipeline

class TrellisState:
    def __init__(self):
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)

    def cleanup(self):
        if self.temp_dir.exists():
            from shutil import rmtree
            rmtree(self.temp_dir)

# Initialize the pipeline once at module level
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
# DO NOT MOVE TO CUDA YET. We'll be dynamically loading parts between 'cpu' and 'cuda' soon.
# Kept for precaution:
#    pipeline.cuda()

# uncomment to reduce memory usage at the cost of numerical precision:
pipeline.to(torch.float16) 
pipeline.models['image_cond_model'].half()  #cuts memory usage in half

# Global state instance:
state = TrellisState()
state.pipeline = pipeline
print('a')