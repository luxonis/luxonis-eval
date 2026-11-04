from typing import Any, Literal

import depthai as dai
import numpy as np
import torch
import torchmetrics

from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.metrics.utils import (
    format_torchmetric_result,
    infer_num_classes,
    prepare_segmentation_metric_inputs,
)


class JaccardIndex(BaseMetric):
    """Jaccard index for semantic segmentation.

    ``ignore_index`` excludes pixels with that ground-truth ID, matching
    LuxonisTrain's JaccardIndex wrapper.
    """

    def __init__(
        self,
        num_classes: int | None = None,
        ignore_index: int | None = None,
        per_class: bool = False,
        input_format: Literal["one-hot", "index", "mixed"] = "index",
        **kwargs: Any,
    ) -> None:
        if "include_background" in kwargs:
            raise ValueError(
                "JaccardIndex uses ignore_index instead of include_background. "
                "Set ignore_index to the dataset background ID to ignore "
                "background target pixels, or omit it to score all pixels."
            )
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.per_class = per_class
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
    ) -> None:
        context = self.context
        if self.target_class_map is None:
            self.target_class_map = context.target_class_map
        prepared = prepare_segmentation_metric_inputs(
            predictions,
            target,
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
            return {"JaccardIndex": 0.0}

        return format_torchmetric_result(
            self.metric.compute(),
            scalar_name="JaccardIndex",
            per_class_prefix=type(self.metric).__name__,
            target_class_map=self.target_class_map,
        )

    def _create_metric(
        self,
        target_mask: torch.Tensor,
        binary_target: bool,
    ) -> torchmetrics.Metric:
        average = None if self.per_class else "macro"
        if binary_target:
            return torchmetrics.JaccardIndex(
                task="binary", ignore_index=self.ignore_index
            )

        return torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=infer_num_classes(
                target_mask,
                configured_num_classes=self.num_classes,
                target_class_map=self.target_class_map,
            ),
            average=average,
            ignore_index=self.ignore_index,
        )
