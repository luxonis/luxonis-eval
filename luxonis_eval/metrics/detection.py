from collections.abc import Sequence
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from luxonis_eval import __version__
from luxonis_eval.utils import suppress_stdout

from .base_metric import BaseMetric


class DetectionMetric(BaseMetric):
    """COCO-style object detection metric."""

    def __init__(self, *, iou_type: str = "bbox") -> None:
        """Initialize the detection metric.

        Parameters
        ----------
        iou_type : str, default="bbox"
            IoU evaluation type.
        """
        self.iou_type = iou_type
        super().__init__()

    def _reset_impl(self) -> None:
        """Reset internal metric state."""
        self._images: list[dict[str, Any]] = []
        self._target: list[dict[str, Any]] = []
        self._predictions: list[dict[str, Any]] = []
        self._target_id = 1
        self._img_id = 1

        self._categories: list[dict[str, Any]] | None = None
        self._category_ids_set: set[int] | None = None

    def _init_categories_once(self, **kwargs: Any) -> None:
        """Initialize COCO categories once.

        Parameters
        ----------
        **kwargs : Any
            Context containing native class map and category IDs.
        """
        # Retrieve additional task-specific options
        native_class_map: dict[int, str] = kwargs.get("native_class_map", {})
        category_ids: Sequence[int] | None = kwargs.get("category_ids")

        if self._categories is not None:
            return

        if category_ids is None:
            category_ids = sorted(native_class_map.keys())

        self._category_ids_set = {int(x) for x in category_ids}
        self._categories = [
            {
                "id": int(cid),
                "name": str(native_class_map.get(int(cid), str(cid))),
            }
            for cid in sorted(self._category_ids_set)
        ]

    def _update_impl(
        self, predictions: Any, target: Any, **kwargs: Any
    ) -> None:
        """Accumulate predictions and ground truths.

        Parameters
        ----------
        predictions : Any
            Detection predictions.
        target : Any
            Ground-truth data.
        **kwargs : Any
            Context including image size and class mappings.
        """
        # Retrieve additional task-specific options
        width = int(kwargs["width"])
        height = int(kwargs["height"])
        class_index_map = kwargs.get("class_index_map")
        target_converter = kwargs.get("target_converter")

        self._init_categories_once(**kwargs)

        if target_converter is None:
            raise ValueError(
                "COCOBBoxMetric needs target_converter in ctx to convert loader GT "
                "to (target_classes, target_boxes_xywh_px)."
            )

        img_id = self._img_id
        self._images.append({"id": img_id, "width": width, "height": height})

        target_classes, target_boxes_xywh = target_converter(
            target, width, height
        )

        for target_box, target_class in zip(
            target_boxes_xywh, target_classes, strict=True
        ):
            target_class = int(target_class)
            if class_index_map is not None:
                target_class = int(class_index_map[target_class])

            if (
                self._category_ids_set is not None
                and target_class not in self._category_ids_set
            ):
                continue

            x, y, w, h = map(float, target_box)
            self._target.append(
                {
                    "id": int(self._target_id),
                    "image_id": int(img_id),
                    "category_id": int(target_class),
                    "bbox": [x, y, w, h],
                    "area": float(max(w, 0.0) * max(h, 0.0)),
                    "iscrowd": 0,
                }
            )
            self._target_id += 1

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

        for box, score, cls_id in zip(bboxes, scores, classes, strict=True):
            cls_id = int(cls_id)
            if (
                self._category_ids_set is not None
                and cls_id not in self._category_ids_set
            ):
                continue

            x1, y1, x2, y2 = map(float, box)
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

            self._predictions.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(cls_id),
                    "bbox": [x1, y1, w, h],
                    "score": float(score),
                }
            )

        self._img_id += 1

    def _compute_impl(self) -> dict[str, float]:
        """Compute COCO detection metrics.

        Returns
        -------
        dict[str, float]
            COCO AP metrics.
        """
        coco_target_dict = {
            "info": {"description": "luxonis-eval", "version": __version__},
            "images": self._images,
            "annotations": self._target,
            "categories": self._categories or [],
        }

        with suppress_stdout():
            coco_target = COCO()
            coco_target.dataset = coco_target_dict  # type: ignore
            coco_target.createIndex()

            if len(self._predictions) == 0:
                return {
                    "AP": 0.0,
                    "AP50": 0.0,
                }

            coco_dt = coco_target.loadRes(self._predictions)  # type: ignore

            coco_eval = COCOeval(coco_target, coco_dt, iouType=self.iou_type)  # type: ignore
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        s = coco_eval.stats

        return {
            "AP": float(s[0]),
            "AP50": float(s[1]),
        }
