from collections.abc import Sequence
from typing import Any

import depthai as dai
import numpy as np

from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.metrics.metrics_utils import detection_to_coco_xywh
from luxonis_eval.utils.coco_utils import COCOStore


class BboxMeanAveragePrecision(BaseMetric):
    """Bounding Box Mean Average Precision (mAP) metric.

    Uses COCO evaluation metrics for bounding box detection.
    """

    def __init__(self, iou_type: str = "bbox", **kwargs: Any) -> None:
        """Initialize the bounding box mAP metric.

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
        return ["/boundingbox"]

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
        target_boxes = target[self.required_target_keys()[0]]
        width = int(kwargs["width"])
        height = int(kwargs["height"])

        class_map: dict[int, str] = kwargs.get("class_map", {})
        category_ids: Sequence[int] | None = kwargs.get("category_ids")
        class_index_map = kwargs.get("class_index_map")
        target_converter = kwargs.get("target_converter")
        if target_converter is None:
            raise ValueError(
                "BboxMeanAveragePrecision requires target_converter in ctx."
            )

        self._store.init_categories_once(
            class_map=class_map, category_ids=category_ids
        )
        img_id = self._store.new_image(width=width, height=height)

        # --- GT ---
        target_classes, target_boxes_xywh = target_converter(
            target_boxes, width, height
        )
        for box_xywh, cls in zip(
            target_boxes_xywh, target_classes, strict=True
        ):
            cls = int(cls)
            if class_index_map is not None:
                cls = int(class_index_map[cls])
            if (
                self._store.category_ids_set is not None
                and cls not in self._store.category_ids_set
            ):
                continue
            x, y, w, h = map(float, box_xywh)
            self._store.add_gt(
                {
                    "image_id": img_id,
                    "category_id": cls,
                    "bbox": [x, y, w, h],
                    "area": float(max(w, 0.0) * max(h, 0.0)),
                    "iscrowd": 0,
                }
            )

        # --- Predictions ---
        for pred in predictions.detections:
            cls = int(pred.label)
            if (
                self._store.category_ids_set is not None
                and cls not in self._store.category_ids_set
            ):
                continue
            box = detection_to_coco_xywh(pred, width, height)
            if box[2] <= 0 or box[3] <= 0:
                continue
            self._store.add_pred(
                {
                    "image_id": img_id,
                    "category_id": cls,
                    "bbox": box,
                    "score": float(pred.confidence),
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
