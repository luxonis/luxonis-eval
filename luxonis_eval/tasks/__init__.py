from .base_task import TASKS_REGISTRY
from .classification import ClassificationTask
from .detection import DetectionTask
from .instance_seg import InstanceSegmentationTask

__all__ = [
    "TASKS_REGISTRY",
    "ClassificationTask",
    "DetectionTask",
    "InstanceSegmentationTask",
]
