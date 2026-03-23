from collections.abc import Sequence
from typing import Any

import depthai as dai
import numpy as np

from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.metrics.metrics_utils import (
    area_from_rle,
    bbox_from_rle,
    binary_mask_to_rle,
)
from luxonis_eval.utils.coco_utils import COCOStore


class MaskMeanAveragePrecision(BaseMetric):
    """Mask Mean Average Precision (mAP) metric.
    Uses COCO evaluation metrics for instance segmentation.
    """

    def __init__(self, iou_type: str = "segm", **kwargs: Any) -> None:
        """Initialize the mask mAP metric.

        Parameters
        ----------
        iou_type : str, optional
            Type of IoU to use for evaluation.
        **kwargs : Any
            Additional metric configuration.
        """
        self._store = COCOStore(iou_type=iou_type)
        super().__init__(**kwargs)

    def metric_keys(self) -> list[str]:
        """Return the ground-truth keys required by the metric.

        Returns
        -------
        list[str]
            Ground-truth key names.
        """
        return ["/boundingbox", "/instance_segmentation"]

    def _reset_impl(self) -> None:
        """Reset internal metric state."""
        self._store.reset()

    def _update_impl(
        self,
        predictions: dai.ImgDetections,
        target: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> None:
        """Update internal metric state.

        Parameters
        ----------
        predictions : dai.ImgDetections
            Model predictions.
        target : dict[str, np.ndarray]
            Ground-truth data.
        **kwargs : Any
            Additional context.
        """
        target_boxes = target[self.metric_keys()[0]]
        target_masks = target[self.metric_keys()[1]]

        width = int(kwargs["width"])
        height = int(kwargs["height"])

        class_map: dict[int, str] = kwargs.get("class_map", {})
        category_ids: Sequence[int] | None = kwargs.get("category_ids")
        class_index_map = kwargs.get("class_index_map")

        self._store.init_categories_once(
            class_map=class_map, category_ids=category_ids
        )
        img_id = self._store.new_image(width=width, height=height)

        # --- GT ---
        target_classes = target_boxes[:, 0].astype(np.int64)

        for mask, cls in zip(target_masks, target_classes, strict=True):
            cls = int(cls)
            if class_index_map is not None:
                cls = int(class_index_map[cls])
            if (
                self._store.category_ids_set is not None
                and cls not in self._store.category_ids_set
            ):
                continue

            rle = binary_mask_to_rle(mask.astype(bool))
            area = area_from_rle(rle)
            coco_bbox = bbox_from_rle(rle)

            self._store.add_gt(
                {
                    "image_id": img_id,
                    "category_id": cls,
                    "segmentation": rle,
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )

        # --- Predictions ---
        detections = predictions.detections
        scores = [det.confidence for det in detections]
        classes = [det.label for det in detections]
        masks: np.ndarray = predictions.getCvSegmentationMask()  # type: ignore
        # Unflatten (N*H, W) → (N, H, W)
        masks = (
            masks.reshape(len(detections), -1, masks.shape[-1])
            if detections
            else np.zeros((0, 0, 0), dtype=np.uint8)
        )
        if masks.size > 0 and masks.shape[1:] != (height, width):
            raise ValueError(
                f"Mask dimensions {masks.shape[1:]} do not match image dimensions ({height}, {width}). "
                f"Ensure masks in 'dai.ImgDetections' are stored as (N*H, W)."
            )

        for mask, score, cls in zip(masks, scores, classes, strict=True):
            cls = int(cls)
            if (
                self._store.category_ids_set is not None
                and cls not in self._store.category_ids_set
            ):
                continue

            if mask.sum() == 0:
                continue

            rle = binary_mask_to_rle(mask.astype(bool))
            coco_bbox = bbox_from_rle(rle)

            self._store.add_pred(
                {
                    "image_id": img_id,
                    "category_id": cls,
                    "segmentation": rle,
                    "bbox": coco_bbox,
                    "score": float(score),
                }
            )

    def _compute_impl(self) -> dict[str, float]:
        """Compute final mAP metrics.

        Returns
        -------
        dict[str, float]
            Computed mAP results.
        """
        return self._store.evaluate()
