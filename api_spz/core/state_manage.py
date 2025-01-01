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
pipeline.cuda()

# pipeline.to(torch.float16) #cuts memory usage in half
# pipeline.models['image_cond_model'].half()  #cuts memory usage in half

# Global state instance 
state = TrellisState()
state.pipeline = pipeline