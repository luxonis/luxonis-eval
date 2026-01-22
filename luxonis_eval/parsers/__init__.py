from .base_parser import PARSERS_REGISTRY
from .classification import ClassificationParser
from .detection import YOLODetectionParser

__all__ = [
    "PARSERS_REGISTRY",
    "ClassificationParser",
    "YOLODetectionParser",
]
