from .base_task import TASKS_REGISTRY
from .classification import ClassificationTask
from .detection import DetectionTask

__all__ = [
    "TASKS_REGISTRY",
    "ClassificationTask",
    "DetectionTask",
]
