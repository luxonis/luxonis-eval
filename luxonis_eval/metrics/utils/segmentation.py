from dataclasses import dataclass
from typing import Mapping

import depthai as dai
import numpy as np
import torch

from luxonis_eval.metrics.metrics_utils import (
    mask_ignore_pixels,
    normalize_prediction_segmentation_mask,
    remap_prediction_mask,
    target_segmentation_to_index_mask,
)
from luxonis_eval.parsers.predictions import Prediction


@dataclass(slots=True)
class PreparedSegmentationData:
    """Normalized segmentation data ready for metric updates."""

    pred_mask: np.ndarray
    target_mask: np.ndarray
    pred_tensor: torch.Tensor
    target_tensor: torch.Tensor
    binary_target: bool


def extract_segmentation_mask(
    predictions: dai.SegmentationMask,
) -> np.ndarray:
    """Extract a semantic-segmentation mask from a prediction payload."""
    if hasattr(predictions, "getCvMask"):
        mask = predictions.getCvMask()
    elif hasattr(predictions, "getCvSegmentationMask"):
        mask = predictions.getCvSegmentationMask()
    else:
        raise TypeError(
            "Unsupported segmentation prediction type "
            f"{type(predictions)!r}: expected a DepthAI SegmentationMask "
            "message."
        )

    if mask is None:
        raise ValueError("Segmentation prediction does not contain a mask.")

    return np.asarray(mask)


def prepare_segmentation_metric_inputs(
    predictions: Prediction,
    target: dict[str, np.ndarray],
    *,
    include_background: bool,
    target_bg: int | None = None,
    class_index_map: dict[int, int] | None = None,
) -> PreparedSegmentationData:
    """Normalize segmentation predictions and targets for metric updates."""
    target_mask, binary_target = target_segmentation_to_index_mask(
        target["/segmentation"]
    )
    pred_mask = normalize_prediction_segmentation_mask(
        extract_segmentation_mask(predictions.require_segmentation_mask()),
        binary_target=binary_target,
    )

    if class_index_map is not None and not binary_target:
        pred_mask = remap_prediction_mask(pred_mask, class_index_map)

    if not binary_target and not include_background and target_bg is not None:
        pred_mask, target_mask = mask_ignore_pixels(
            pred_mask,
            target_mask,
            ignore_index=target_bg,
        )

    pred_tensor = torch.from_numpy(pred_mask.astype(np.int64))
    target_tensor = torch.from_numpy(target_mask.astype(np.int64))
    return PreparedSegmentationData(
        pred_mask=pred_mask,
        target_mask=target_mask,
        pred_tensor=pred_tensor,
        target_tensor=target_tensor,
        binary_target=binary_target,
    )


def infer_num_classes(
    target_tensor: torch.Tensor,
    *,
    configured_num_classes: int | None,
    target_class_map: Mapping[int, str] | None,
) -> int:
    """Resolve the effective class count for multiclass metrics."""
    if configured_num_classes is not None:
        return configured_num_classes
    if target_class_map:
        return len(target_class_map)

    unique_ids = torch.unique(target_tensor)
    if unique_ids.numel() == 0:
        raise ValueError("Cannot infer num_classes from an empty target.")
    return int(unique_ids.max().item()) + 1


def format_torchmetric_result(
    result: torch.Tensor,
    *,
    scalar_name: str,
    per_class_prefix: str,
    target_class_map: Mapping[int, str] | None = None,
) -> dict[str, float]:
    """Format scalar or per-class torchmetrics output consistently."""
    if result.ndim == 0 or result.numel() == 1:
        return {scalar_name: float(result)}

    class_names = [
        target_class_map.get(index, f"class_{index}")
        if target_class_map is not None
        else f"class_{index}"
        for index in range(result.numel())
    ]
    return {
        f"{per_class_prefix}_{class_name}": float(value)
        for class_name, value in zip(class_names, result, strict=True)
    }
