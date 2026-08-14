from typing import Any, Literal

import numpy as np
import torch
import torchmetrics

from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.metrics.utils import (
    format_torchmetric_result,
    infer_num_classes,
    prepare_segmentation_metric_inputs,
)
from luxonis_eval.parsers.predictions import Prediction


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
        predictions: Prediction,
        target: dict[str, np.ndarray],
    ) -> None:
        context = self.require_context()
        if self.target_class_map is None:
            self.target_class_map = context.target_class_map
        prepared = prepare_segmentation_metric_inputs(
            predictions,
            target,
            include_background=self.include_background,
            target_bg=context.target_background_index,
            class_index_map=context.class_index_map,
        )

        if self.metric is None:
            self.metric = self._create_metric(
                target_mask=prepared.target_tensor,
                binary_target=prepared.binary_target,
            )
        self.metric.update(prepared.pred_tensor, prepared.target_tensor)

    def compute(self) -> dict[str, float]:
        if self.metric is None:
            return {"F1Score": 0.0}

        return format_torchmetric_result(
            self.metric.compute(),
            scalar_name="F1Score",
            per_class_prefix=type(self.metric).__name__,
            target_class_map=self.target_class_map,
        )

    def _create_metric(
        self,
        target_mask: torch.Tensor,
        binary_target: bool,
    ) -> torchmetrics.Metric:
        average = None if self.average == "none" else self.average
        if binary_target:
            return torchmetrics.F1Score(task="binary", average=average)

        return torchmetrics.F1Score(
            task="multiclass",
            num_classes=infer_num_classes(
                target_mask,
                configured_num_classes=self.num_classes,
                target_class_map=self.target_class_map,
            ),
            average=average,
        )
