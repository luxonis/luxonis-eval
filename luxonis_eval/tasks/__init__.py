from .base_task import TASKS_REGISTRY
from .classification import ClassificationTask
from .detection import DetectionTask
from .instance_seg import InstanceSegmentationTask
from .keypoint_detection import KeypointDetectionTask

__all__ = [
    "TASKS_REGISTRY",
    "ClassificationTask",
    "DetectionTask",
    "InstanceSegmentationTask",
    "KeypointDetectionTask",
]
