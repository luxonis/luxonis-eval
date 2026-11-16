from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import depthai as dai
import numpy as np
import torch

from luxonis_eval.metrics.metrics_utils import (
    normalize_prediction_segmentation_mask,
    remap_prediction_mask,
    target_segmentation_to_index_mask,
)


@dataclass(slots=True)
class PreparedSegmentationData:
    """Normalized segmentation data ready for metric updates."""

    pred_mask: np.ndarray
    target_mask: np.ndarray
    pred_tensor: torch.Tensor
    target_tensor: torch.Tensor
    binary_target: bool
    target_channels: torch.Tensor | None = None

    def formatted_tensors(
        self,
        *,
        num_classes: int,
        input_format: Literal["one-hot", "index", "mixed"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Represent one image in the requested TorchMetrics format."""
        if input_format == "index":
            return self.pred_tensor, self.target_tensor

        pred = torch.nn.functional.one_hot(
            self.pred_tensor, num_classes=num_classes
        ).movedim(-1, 1)
        if input_format == "mixed":
            return pred, self.target_tensor

        target = self.target_channels
        if target is None:
            target = torch.nn.functional.one_hot(
                self.target_tensor, num_classes=num_classes
            ).movedim(-1, 1)
        return pred, target


def extract_segmentation_mask(
    predictions: dai.SegmentationMask,
) -> np.ndarray:
    """Extract a semantic-segmentation mask from a prediction
    payload."""
    mask = predictions.getCvMask()

    if mask is None:
        raise ValueError("Segmentation prediction does not contain a mask.")

    return np.asarray(mask)


def prepare_segmentation_metric_inputs(
    predictions: dai.SegmentationMask,
    target: dict[str, np.ndarray],
    *,
    target_bg: int | None = None,
    class_index_map: dict[int, int] | None = None,
) -> PreparedSegmentationData:
    """Normalize segmentation predictions and targets for metric
    updates."""
    target_mask, binary_target = target_segmentation_to_index_mask(
        target["/segmentation"]
    )
    pred_mask = normalize_prediction_segmentation_mask(
        extract_segmentation_mask(predictions),
        binary_target=binary_target,
    )

    if pred_mask.shape != target_mask.shape:
        raise ValueError(
            "Segmentation prediction and target shapes must match, got "
            f"{pred_mask.shape} and {target_mask.shape}."
        )

    if not binary_target:
        # Save native unassigned pixels before remapping ordinary class IDs.
        unassigned = pred_mask == 255
        if np.any(unassigned) and target_bg is None:
            raise ValueError(
                "Segmentation predictions contain unassigned label 255, "
                "but the dataset has no background class to map it to."
            )
        if class_index_map is not None:
            pred_mask = remap_prediction_mask(pred_mask, class_index_map)
        if np.any(unassigned):
            pred_mask[unassigned] = target_bg

    # TorchMetrics expects BHW index masks; one update contains one image.
    pred_tensor = torch.from_numpy(pred_mask.astype(np.int64)).unsqueeze(0)
    target_tensor = torch.from_numpy(target_mask.astype(np.int64)).unsqueeze(0)
    return PreparedSegmentationData(
        pred_mask=pred_mask,
        target_mask=target_mask,
        pred_tensor=pred_tensor,
        target_tensor=target_tensor,
        binary_target=binary_target,
        target_channels=(
            torch.from_numpy(
                np.asarray(target["/segmentation"], dtype=bool)
            ).unsqueeze(0)
            if np.asarray(target["/segmentation"]).ndim == 3
            and not binary_target
            else None
        ),
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
    class_offset: int = 0,
) -> dict[str, float]:
    """Format scalar or per-class torchmetrics output consistently."""
    if result.ndim == 0 or result.numel() == 1:
        return {scalar_name: float(result)}

    class_names = [
        target_class_map.get(index, f"class_{index}")
        if target_class_map is not None
        else f"class_{index}"
        for index in range(class_offset, class_offset + result.numel())
    ]
    return {
        f"{per_class_prefix}_{class_name}": float(value)
        for class_name, value in zip(class_names, result, strict=True)
    }
