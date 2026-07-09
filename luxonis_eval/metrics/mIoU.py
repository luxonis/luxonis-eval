from typing import Any, Literal

import depthai as dai
import numpy as np
import torch
from torchmetrics.segmentation import MeanIoU

from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.metrics.metrics_utils import (
    binary_segmentation_confusion,
    mask_ignore_pixels,
    normalize_prediction_segmentation_mask,
    remap_prediction_mask,
    target_segmentation_to_index_mask,
)
from luxonis_eval.utils.depthai_nodes import extract_segmentation_mask


class MIoU(BaseMetric):
    """Mean IoU metric."""

    @property
    def report_name(self) -> str:
        return "MIoU"

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        per_class: bool = True,
        input_format: Literal["one-hot", "index", "mixed"] = "index",
        **kwargs: Any,
    ) -> None:
        """Initialize the Mean IoU metric.

        Parameters
        ----------
        num_classes : int
            Number of classes in the segmentation task.
        include_background : bool, default=False
            Whether to include the background class in the metric calculation.
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
        # Retrieve additional metric-specific options
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

        self.metric.update(
            torch.from_numpy(pred_mask.astype(np.int64)),
            torch.from_numpy(target_mask.astype(np.int64)),
        )

    def compute(self) -> dict[str, float]:
        """Compute final mIoU metrics.

        Returns
        -------
        dict[str, float]
            Computed mIoU results.
        """
        results = self.metric.compute()

        if not self.per_class:
            return {"mIoU": float(results)}

        class_names = [
            self.target_class_map.get(i, f"class_{i}")
            if self.target_class_map is not None
            else f"class_{i}"
            for i in range(len(results))
        ]

        return {
            f"mIoU ({name})": float(r)
            for name, r in zip(class_names, results, strict=True)
        }


class JaccardIndex(BaseMetric):
    """LuxonisTrain-compatible Jaccard index for semantic segmentation."""

    def __init__(
        self,
        num_classes: int | None = None,
        include_background: bool = False,
        per_class: bool = False,
        input_format: Literal["one-hot", "index", "mixed"] = "index",
        **kwargs: Any,
    ) -> None:
        self.num_classes = num_classes
        self.include_background = include_background
        self.per_class = per_class
        self.input_format = input_format
        self.target_class_map = None
        super().__init__(**kwargs)

    @property
    def report_name(self) -> str:
        return "JaccardIndex"

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
            "luxonis-eval `JaccardIndex` currently mirrors LuxonisTrain "
            "behavior only for binary semantic segmentation. Use `MIoU` "
            "for non-binary semantic segmentation."
        )

    def compute(self) -> dict[str, float]:
        denom = (
            self.true_positives
            + self.false_positives
            + self.false_negatives
        )
        score = 0.0 if denom == 0 else self.true_positives / denom
        return {"JaccardIndex": float(score)}
