from dataclasses import dataclass

import depthai as dai
import numpy as np
from depthai_nodes import Classifications


@dataclass(frozen=True, slots=True)
class Prediction:
    """Structured parser output with optional task-specific payloads."""

    classification: Classifications | None = None
    segmentation_mask: dai.SegmentationMask | None = None
    detections: dai.ImgDetections | None = None
    instance_masks: np.ndarray | None = None

    def require_classification(self) -> Classifications:
        if self.classification is None:
            raise TypeError(
                "Prediction does not contain classification scores."
            )
        return self.classification

    def require_segmentation_mask(self) -> dai.SegmentationMask:
        if self.segmentation_mask is None:
            raise TypeError(
                "Prediction does not contain a segmentation mask."
            )
        return self.segmentation_mask

    def require_detections(self) -> dai.ImgDetections:
        if self.detections is None:
            raise TypeError(
                "Prediction does not contain detections."
            )
        return self.detections

    def require_instance_masks(self) -> np.ndarray:
        if self.instance_masks is None:
            raise TypeError(
                "Prediction does not contain per-instance masks."
            )
        return self.instance_masks

    @property
    def classes(self) -> list[str]:
        return self.require_classification().classes
