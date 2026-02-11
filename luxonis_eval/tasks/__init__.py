from .classification import ClassificationTask
from .detection import DetectionTask
from .instance_seg import InstanceSegmentationTask
from .keypoint_detection import KeypointDetectionTask

__all__ = [
    "ClassificationTask",
    "DetectionTask",
    "InstanceSegmentationTask",
    "KeypointDetectionTask",
]
