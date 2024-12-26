import time
from typing import Dict, Optional, Any
from .models_pydantic import TaskStatus

class Task:
    def __init__(self, task_id: str):
        self.id = task_id
        self.status = TaskStatus.PROCESSING
        self.progress = 0
        self.message = ""
        self.created_at = time.time()
        self.outputs: Optional[Dict[str, Any]] = None

    def update(self, progress: int, message: str, status: Optional[TaskStatus] = None, outputs: Optional[Dict] = None):
        self.progress = progress
        self.message = message
        if status:
            self.status = status
        if outputs is not None:
            self.outputs = outputs

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}

    def create_task(self, task_id: str) -> Task:
        task = Task(task_id)
        self.tasks[task_id] = task
        return task
    
    def update_taks(self, task_id, progress: int, message: str, 
                    status: Optional[TaskStatus] = None, outputs: Optional[Dict] = None):
        self.tasks[task_id].update(progress, message, status, outputs)

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def cleanup_old_tasks(self, max_age: float = 3600):
        current_time = time.time()
        old_tasks = [tid for tid, task in self.tasks.items() 
                    if current_time - task.created_at > max_age]
        for tid in old_tasks:
            del self.tasks[tid]

task_manager = TaskManager()