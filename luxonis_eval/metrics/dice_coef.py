from typing import Any, Literal

import depthai as dai
import numpy as np
from torchmetrics.segmentation import DiceScore

from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.metrics.utils import prepare_segmentation_metric_inputs


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
        predictions: dai.SegmentationMask,
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
        prepared = prepare_segmentation_metric_inputs(
            predictions,
            target,
            include_background=self.include_background,
            target_bg=kwargs.get("target_bg"),
            class_index_map=kwargs.get("class_index_map"),
        )
        self.metric.update(prepared.pred_tensor, prepared.target_tensor)

    def compute(self) -> dict[str, float]:
        """Compute final Dice coefficient metrics.

        Returns
        -------
        dict[str, float]
            Computed Dice coefficient results.
        """
        return {"Dice Score": float(self.metric.compute())}
