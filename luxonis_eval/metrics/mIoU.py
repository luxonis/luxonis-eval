from typing import Any, Literal

import depthai as dai
import numpy as np
from torchmetrics.segmentation import MeanIoU

from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.metrics.utils import prepare_segmentation_metric_inputs


class MIoU(BaseMetric):
    """Mean IoU metric."""

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        per_class: bool = False,
        input_format: Literal["one-hot", "index", "mixed"] = "index",
        **kwargs: Any,
    ) -> None:
        """Initialize the Mean IoU metric.

        Parameters
        ----------
        num_classes : int
            Number of classes in the segmentation task.
        include_background : bool, default=True
            Whether to score class 0. Foreground errors on background pixels
            still count when False, matching LuxonisTrain.
        per_class : bool, default=False
            Whether to compute IoU per class.
        input_format : Literal["one-hot", "index", "mixed"], default="index"
            Format of the input data.
        **kwargs : Any
            Additional metric configuration.
        """
        self.metric = MeanIoU(
            num_classes=num_classes,
            include_background=include_background,
            per_class=per_class,
            input_format=input_format,
        )
        self.per_class = per_class
        self.num_classes = num_classes
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
    ) -> None:
        """Update internal metric state.

        Parameters
        ----------
        predictions : SegmentationMask
            Model predictions (logits or probabilities).
        target : dict[str, np.ndarray]
            Ground-truth labels.
        """
        context = self.context
        if self.target_class_map is None:
            self.target_class_map = context.target_class_map
        prepared = prepare_segmentation_metric_inputs(
            predictions,
            target,
            target_bg=context.target_background_index,
            class_index_map=context.class_index_map,
        )
        self.metric.update(
            *prepared.formatted_tensors(
                num_classes=self.num_classes, input_format=self.input_format
            )
        )

    def compute(self) -> dict[str, float]:
        """Compute final mIoU metrics.

        Returns
        -------
        dict[str, float]
            Computed mIoU results.
        """
        results = self.metric.compute()

        if not self.per_class or results.ndim == 0 or results.numel() == 1:
            return {"mIoU": float(results)}

        first_class = 0 if self.include_background else 1
        class_names = [
            self.target_class_map.get(i, f"class_{i}")
            if self.target_class_map is not None
            else f"class_{i}"
            for i in range(first_class, first_class + results.numel())
        ]

        per_class = {
            f"mIoU ({name})": float(r)
            for name, r in zip(class_names, results, strict=True)
        }
        return {"mIoU": float(results.mean()), **per_class}
