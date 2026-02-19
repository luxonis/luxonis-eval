from .classification import ClassificationTask
from .detection import DetectionTask
from .instance_seg import InstanceSegmentationTask
from .keypoint_detection import KeypointDetectionTask
from .semantic_seg import SemanticSegmentationTask

__all__ = [
    "ClassificationTask",
    "DetectionTask",
    "InstanceSegmentationTask",
    "KeypointDetectionTask",
    "SemanticSegmentationTask",
]
