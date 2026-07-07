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

    def required_target_keys(self) -> list[str]:
        """Return the ground-truth keys required by the metric.

        Returns
        -------
        list[str]
            Ground-truth key names.
        """
        return ["/boundingbox", "/instance_segmentation"]

    def reset(self) -> None:
        """Reset internal metric state."""
        self._store.reset()

    def update(
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
        target_boxes = target[self.required_target_keys()[0]]
        target_masks = target[self.required_target_keys()[1]]

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
        masks = self._resolve_prediction_masks(
            predictions.getCvSegmentationMask(),  # type: ignore[arg-type]
            n_detections=len(detections),
            height=height,
            width=width,
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

    def compute(self) -> dict[str, float]:
        """Compute final mAP metrics.

        Returns
        -------
        dict[str, float]
            Computed mAP results.
        """
        return self._store.evaluate()

    def _resolve_prediction_masks(
        self,
        raw_masks: np.ndarray | None,
        *,
        n_detections: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        if raw_masks is None:
            return np.zeros((0, height, width), dtype=np.uint8)

        masks = np.asarray(raw_masks)
        if masks.size == 0:
            return np.zeros((0, height, width), dtype=np.uint8)

        if masks.ndim == 3:
            if masks.shape == (n_detections, height, width):
                return masks.astype(np.uint8, copy=False)
            raise ValueError(
                "Unsupported 3D segmentation mask shape "
                f"{masks.shape}. Expected ({n_detections}, {height}, {width})."
            )

        if masks.ndim != 2:
            raise ValueError(
                f"Unsupported segmentation mask rank {masks.ndim}. "
                "Expected a 2D instance-id mask of shape (H, W) or a "
                "flattened stack of shape (N*H, W)."
            )

        if masks.shape == (height, width):
            if n_detections == 0:
                return np.zeros((0, height, width), dtype=np.uint8)
            return np.stack(
                [(masks == idx).astype(np.uint8) for idx in range(n_detections)],
                axis=0,
            )

        if n_detections > 0 and masks.shape == (n_detections * height, width):
            return masks.reshape(n_detections, height, width).astype(
                np.uint8, copy=False
            )

        raise ValueError(
            f"Mask dimensions {masks.shape} do not match the expected image "
            f"dimensions ({height}, {width}). Supported formats are (H, W) "
            "for an instance-id mask and (N*H, W) for a flattened per-instance stack."
        )
