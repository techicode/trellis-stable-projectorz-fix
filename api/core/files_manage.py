from pathlib import Path
import time
from datetime import datetime, timedelta

class FileManager:
    def __init__(self, temp_dir: Path, max_age: timedelta = timedelta(hours=1)):
        self.temp_dir = temp_dir
        self.max_age = max_age

    def get_temp_path(self, task_id: str, filename: str) -> Path:
        return self.temp_dir / f"{task_id}_{filename}"

    def cleanup_old_files(self):
        cutoff = time.time() - self.max_age.total_seconds()
        for file in self.temp_dir.iterdir():
            if file.stat().st_mtime < cutoff:
                file.unlink()

file_manager = FileManager(Path("temp"))