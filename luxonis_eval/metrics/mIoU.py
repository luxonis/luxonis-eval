from typing import Any, Literal

import numpy as np
import torch
from depthai_nodes import SegmentationMask
from torchmetrics.segmentation import MeanIoU

from luxonis_eval.metrics.base_metric import BaseMetric


class MIoU(BaseMetric):
    """Mean IoU metric."""

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

    def metric_keys(self) -> list[str]:
        """Return the ground-truth keys required by the metric.

        Returns
        -------
        list[str]
            Ground-truth key names.
        """
        return ["/segmentation"]

    @staticmethod
    def _remap_prediction_mask(
        pred_mask: np.ndarray,
        class_index_map: dict[int, int],
    ) -> np.ndarray:
        """Remap prediction mask indices to align with target mask indices.

        Parameters
        ----------
        pred_mask : np.ndarray
            The predicted segmentation mask with class indices.
        class_index_map : dict[int, int]
            Mapping from target class index to prediction class index.

        Returns
        -------
        np.ndarray
            The remapped prediction mask with indices aligned to the target mask."""
        remapped = np.full(pred_mask.shape, fill_value=255, dtype=np.int64)

        for target_idx, pred_idx in class_index_map.items():
            remapped[pred_mask == pred_idx] = target_idx

        return remapped

    @staticmethod
    def _mask_ignore_pixels(
        pred_mask: np.ndarray, target_mask: np.ndarray, ignore_index: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Replace ignore_index pixels with 0 in both masks so they cancel out and don't affect IoU for any real class.

        Parameters
        ----------
        pred_mask : np.ndarray
            The predicted segmentation mask with class indices.
        target_mask : np.ndarray
            The ground-truth segmentation mask with class indices.
        ignore_index : int, default=0
            The class index to ignore (e.g., background).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The modified prediction and target masks with ignore_index pixels set to 0.
        """
        ignore_mask = target_mask == ignore_index
        target_mask = target_mask.copy()
        pred_mask = pred_mask.copy()

        target_mask[ignore_mask] = 0
        pred_mask[ignore_mask] = 0

        return pred_mask, target_mask

    def _reset_impl(self) -> None:
        """Reset internal metric state."""
        self.metric.reset()

    def _update_impl(
        self, predictions: SegmentationMask, target: Any, **kwargs: Any
    ) -> None:
        """Update internal metric state.

        Parameters
        ----------
        predictions : SegmentationMask
            Model predictions (logits or probabilities).
        target : Any
            Ground-truth labels.
        **kwargs : Any
            Additional context.
        """
        # Retrieve additional metric-specific options
        if self.target_class_map is None:
            self.target_class_map = kwargs.get("target_class_map", {})
        target_bg = kwargs.get("target_bg")
        class_index_map = kwargs.get("class_index_map", {})

        pred_mask: np.ndarray = predictions.mask
        target_mask = np.argmax(target[self.metric_keys()[0]], axis=0)

        pred_mask = self._remap_prediction_mask(pred_mask, class_index_map)

        if not self.include_background and target_bg is not None:
            pred_mask, target_mask = self._mask_ignore_pixels(
                pred_mask, target_mask, ignore_index=target_bg
            )

        self.metric.update(
            torch.from_numpy(pred_mask.astype(np.int64)),
            torch.from_numpy(target_mask.astype(np.int64)),
        )

    def _compute_impl(self) -> dict[str, float]:
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
