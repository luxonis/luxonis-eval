from typing import Any, Literal

import depthai as dai
import numpy as np
import torch
import torchmetrics

from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.metrics.metrics_utils import (
    mask_ignore_pixels,
    normalize_prediction_segmentation_mask,
    remap_prediction_mask,
    target_segmentation_to_index_mask,
)
from luxonis_eval.utils.utils import extract_segmentation_mask


class F1Score(BaseMetric):
    """F1 score for semantic segmentation."""

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
        self.metric: torchmetrics.Metric | None = None
        super().__init__(**kwargs)

    def required_target_keys(self) -> list[str]:
        return ["/segmentation"]

    def reset(self) -> None:
        if self.metric is not None:
            self.metric.reset()

    def update(
        self,
        predictions: dai.SegmentationMask,
        target: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> None:
        if self.target_class_map is None:
            self.target_class_map = kwargs.get("target_class_map", {})
        target_bg = kwargs.get("target_bg")
        class_index_map = kwargs.get("class_index_map")

        target_mask, binary_target = target_segmentation_to_index_mask(
            target[self.required_target_keys()[0]]
        )
        pred_mask = normalize_prediction_segmentation_mask(
            extract_segmentation_mask(predictions),
            binary_target=binary_target,
        )

        if class_index_map is not None and not binary_target:
            pred_mask = remap_prediction_mask(pred_mask, class_index_map)

        if (
            not binary_target
            and not self.include_background
            and target_bg is not None
        ):
            pred_mask, target_mask = mask_ignore_pixels(
                pred_mask, target_mask, ignore_index=target_bg
            )

        pred_tensor = torch.from_numpy(pred_mask.astype(np.int64))
        target_tensor = torch.from_numpy(target_mask.astype(np.int64))

        if self.metric is None:
            self.metric = self._create_metric(
                target_mask=target_tensor,
                binary_target=binary_target,
            )
        self.metric.update(pred_tensor, target_tensor)

    def compute(self) -> dict[str, float]:
        if self.metric is None:
            return {"F1Score": 0.0}

        result = self.metric.compute()
        if result.ndim == 0 or result.numel() == 1:
            return {"F1Score": float(result)}

        class_names = [
            self.target_class_map.get(i, f"class_{i}")
            if self.target_class_map is not None
            else f"class_{i}"
            for i in range(result.numel())
        ]
        return {
            f"{type(self.metric).__name__}_{class_name}": float(value)
            for class_name, value in zip(class_names, result, strict=True)
        }

    def _create_metric(
        self,
        target_mask: torch.Tensor,
        binary_target: bool,
    ) -> torchmetrics.Metric:
        average = None if self.average == "none" else self.average
        if binary_target:
            return torchmetrics.F1Score(task="binary", average=average)

        num_classes = self.num_classes
        if num_classes is None:
            if self.target_class_map:
                num_classes = len(self.target_class_map)
            else:
                num_classes = int(target_mask.max().item()) + 1

        return torchmetrics.F1Score(
            task="multiclass",
            num_classes=num_classes,
            average=average,
        )
