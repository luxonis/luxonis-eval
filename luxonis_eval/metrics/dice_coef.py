from typing import Any, Literal

import numpy as np
import torch
from depthai_nodes import SegmentationMask
from torchmetrics.segmentation import DiceScore

from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.metrics.metrics_utils import (
    mask_ignore_pixels,
    remap_prediction_mask,
)


class DiceCoefficient(BaseMetric):
    """Dice coefficient metric for semantic segmentation."""

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        average: Literal["micro", "macro", "weighted", "none"]
        | None = "micro",
        input_format: Literal["one-hot", "index"] = "index",
        **kwargs: Any,
    ) -> None:
        """Initialize the Dice coefficient metric.

        Parameters
        ----------
        num_classes : int
            Number of classes in the segmentation task.
        include_background : bool, default=True
            Whether to include the background class in the metric calculation.
        average : Literal["micro", "macro", "weighted", "none"] | None, default="micro"
            How to average the metric across classes.
        input_format : Literal["one-hot", "index"], default="index"
            Format of the input data.
        **kwargs : Any
            Additional metric configuration.
        """
        self.metric = DiceScore(
            num_classes=num_classes,
            include_background=include_background,
            average=average,
            input_format=input_format,
        )
        self.include_background = include_background
        self.input_format = input_format
        self.target_class_map = None
        super().__init__(**kwargs)

    def required_target_keys(self) -> list[str]:
        """Return the ground-truth keys required by the metric.

        Returns
        -------
        list[str]
            Ground-truth key names.
        """
        return ["/segmentation"]

    def reset(self) -> None:
        """Reset internal metric state."""
        self.metric.reset()

    def update(
        self,
        predictions: SegmentationMask,
        target: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> None:
        """Update internal metric state.

        Parameters
        ----------
        predictions : SegmentationMask
            Model predictions (logits or probabilities).
        target : dict[str, np.ndarray]
            Ground-truth labels.
        **kwargs : Any
            Additional context.
        """
        if self.target_class_map is None:
            self.target_class_map = kwargs.get("target_class_map", {})
        target_bg = kwargs.get("target_bg")
        class_index_map = kwargs.get("class_index_map")

        pred_mask: np.ndarray = predictions.mask
        target_mask = np.argmax(target[self.required_target_keys()[0]], axis=0)

        if class_index_map is not None:
            pred_mask = remap_prediction_mask(pred_mask, class_index_map)

        if not self.include_background and target_bg is not None:
            pred_mask, target_mask = mask_ignore_pixels(
                pred_mask, target_mask, ignore_index=target_bg
            )

        self.metric.update(
            torch.from_numpy(pred_mask.astype(np.int64)),
            torch.from_numpy(target_mask.astype(np.int64)),
        )

    def _compute_impl(self) -> dict[str, float]:
        """Compute final Dice coefficient metrics.

        Returns
        -------
        dict[str, float]
            Computed Dice coefficient results.
        """
        return {"Dice Score": float(self.metric.compute())}
