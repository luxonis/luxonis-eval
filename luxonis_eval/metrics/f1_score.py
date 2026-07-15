from typing import Any, Literal

import depthai as dai
import numpy as np

from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.metrics.metrics_utils import (
    binary_segmentation_confusion,
    normalize_prediction_segmentation_mask,
    target_segmentation_to_index_mask,
)
from luxonis_eval.utils.depthai_nodes import extract_segmentation_mask


class F1Score(BaseMetric):
    """F1 score for binary semantic segmentation."""

    def __init__(
        self,
        num_classes: int | None = None,
        include_background: bool = True,
        average: Literal["micro", "macro", "weighted", "none"]
        | None = "micro",
        input_format: Literal["one-hot", "index"] = "index",
        **kwargs: Any,
    ) -> None:
        self.num_classes = num_classes
        self.include_background = include_background
        self.average = average
        self.input_format = input_format
        self.target_class_map = None
        super().__init__(**kwargs)

    def required_target_keys(self) -> list[str]:
        return ["/segmentation"]

    def reset(self) -> None:
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def update(
        self,
        predictions: dai.SegmentationMask,
        target: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> None:
        if self.target_class_map is None:
            self.target_class_map = kwargs.get("target_class_map", {})
        class_index_map = kwargs.get("class_index_map")
        target_bg = kwargs.get("target_bg")

        target_mask, binary_target = target_segmentation_to_index_mask(
            target[self.required_target_keys()[0]]
        )
        pred_mask = normalize_prediction_segmentation_mask(
            extract_segmentation_mask(predictions),
            binary_target=binary_target,
        )

        if binary_target:
            tp, fp, fn = binary_segmentation_confusion(pred_mask, target_mask)
            self.true_positives += tp
            self.false_positives += fp
            self.false_negatives += fn
            return

        del class_index_map, target_bg, pred_mask, target_mask
        raise NotImplementedError(
            "`F1Score` behavior is only implemented for "
            "binary semantic segmentation. Use "
            "`DiceCoefficient` for non-binary semantic segmentation."
        )

    def compute(self) -> dict[str, float]:
        denom = (
            2 * self.true_positives
            + self.false_positives
            + self.false_negatives
        )
        score = 0.0 if denom == 0 else (2 * self.true_positives) / denom
        return {"F1Score": float(score)}
