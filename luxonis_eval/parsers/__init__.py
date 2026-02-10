from .base_parser import PARSERS_REGISTRY
from .classification import ClassificationParser
from .detection import YOLODetectionParser
from .instance_seg import YOLOInstanceSegmentationParser

__all__ = [
    "PARSERS_REGISTRY",
    "ClassificationParser",
    "YOLODetectionParser",
    "YOLOInstanceSegmentationParser",
]
