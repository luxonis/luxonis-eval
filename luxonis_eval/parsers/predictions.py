from dataclasses import dataclass
from typing import Any, cast

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

    def _require_field(self, field_name: str, description: str) -> Any:
        value = getattr(self, field_name)
        if value is None:
            raise TypeError(f"Prediction does not contain {description}.")
        return value

    def require_classification(self) -> Classifications:
        return cast(
            Classifications,
            self._require_field("classification", "classification scores"),
        )

    def require_segmentation_mask(self) -> dai.SegmentationMask:
        return cast(
            dai.SegmentationMask,
            self._require_field("segmentation_mask", "a segmentation mask"),
        )

    def require_detections(self) -> dai.ImgDetections:
        return cast(
            dai.ImgDetections,
            self._require_field("detections", "detections"),
        )

    def require_instance_masks(self) -> np.ndarray:
        return cast(
            np.ndarray,
            self._require_field("instance_masks", "per-instance masks"),
        )

    @property
    def classes(self) -> list[str]:
        return self.require_classification().classes
