from collections.abc import Sequence
from typing import Any

import numpy as np

from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.metrics.primitives.coco_utils import COCOStore


class BboxMeanAveragePrecision(BaseMetric):
    """Bounding Box Mean Average Precision (mAP) metric.
    Uses COCO evaluation metrics for bounding box detection.
    """

    def __init__(self) -> None:
        """Initialize the bounding box mAP metric."""
        self._store = COCOStore(iou_type="bbox")
        super().__init__()

    def _reset_impl(self) -> None:
        """Reset internal metric state."""
        self._store.reset()

    def _update_impl(
        self, predictions: Any, target: Any, **kwargs: Any
    ) -> None:
        """Update internal metric state.

        Parameters
        ----------
        predictions : Any
            Model predictions.
        target : Any
            Ground-truth data.
        **kwargs : Any
            Additional context.
        """
        width = int(kwargs["width"])
        height = int(kwargs["height"])

        native_class_map: dict[int, str] = kwargs.get("native_class_map", {})
        category_ids: Sequence[int] | None = kwargs.get("category_ids")
        class_index_map = kwargs.get("class_index_map")
        target_converter = kwargs.get("target_converter")
        if target_converter is None:
            raise ValueError(
                "BboxMeanAveragePrecision requires target_converter in ctx."
            )

        self._store.init_categories_once(
            native_class_map=native_class_map, category_ids=category_ids
        )
        img_id = self._store.new_image(width=width, height=height)

        # --- GT ---
        target_classes, target_boxes_xywh = target_converter(
            target, width, height
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

        # --- DT ---
        bboxes = predictions["bboxes"]
        scores = predictions["scores"]
        classes = predictions["classes"]

        bboxes = (
            bboxes.detach().cpu().numpy()
            if hasattr(bboxes, "detach")
            else np.asarray(bboxes)
        )
        scores = (
            scores.detach().cpu().numpy()
            if hasattr(scores, "detach")
            else np.asarray(scores)
        )
        classes = (
            classes.detach().cpu().numpy()
            if hasattr(classes, "detach")
            else np.asarray(classes)
        )

        for box, score, cls in zip(bboxes, scores, classes, strict=True):
            cls = int(cls)
            if (
                self._store.category_ids_set is not None
                and cls not in self._store.category_ids_set
            ):
                continue

            x1, y1, x2, y2 = map(float, box)
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            self._store.add_dt(
                {
                    "image_id": img_id,
                    "category_id": cls,
                    "bbox": [x1, y1, w, h],
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
