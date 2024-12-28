import os
import sys
from pathlib import Path
from typing import Optional

# Add trellis to Python path (adjust if your trellis code is located elsewhere)
sys.path.append(os.getcwd() + "/../")

import torch
from trellis.pipelines import TrellisImageTo3DPipeline


class TrellisState:
    def __init__(self):
        self.pipeline: Optional[TrellisImageTo3DPipeline] = None
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)

    def initialize_pipeline(self):
        """Load the pipeline once. We'll do it on the main process (no pickling issues)."""
        if self.pipeline is None:
            # Load your model from HuggingFace or local
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
            self.pipeline.cuda()
        return self.pipeline

    def cleanup(self):
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)

# A global state instance
state = TrellisState()
