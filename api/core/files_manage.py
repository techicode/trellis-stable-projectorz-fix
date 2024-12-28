from pathlib import Path
from .state_manage import state

class FileManager:
    def __init__(self):
        self.base_path = state.temp_dir  # "temp" folder

    def get_temp_path(self, task_id: str, filename: str) -> Path:
        task_dir = self.base_path / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir / filename

file_manager = FileManager()
