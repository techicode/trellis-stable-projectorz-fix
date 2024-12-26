import os
import sys
from pathlib import Path
from typing import Optional

# Add trellis to Python path 
sys.path.append(os.getcwd()+"/../..")

from trellis.pipelines import TrellisImageTo3DPipeline


class TrellisState:
    def __init__(self):
        self.pipeline: Optional[TrellisImageTo3DPipeline] = None
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)

    def initialize_pipeline(self):
        if self.pipeline is None:
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
            self.pipeline.cuda()
        return self.pipeline

    def cleanup(self):
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

state = TrellisState()