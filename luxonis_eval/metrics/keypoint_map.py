from collections.abc import Sequence
from typing import Any

import depthai as dai
import numpy as np

from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.metrics.metrics_utils import (
    bbox_area_from_keypoints,
    to_coco_kpts_flat,
)
from luxonis_eval.utils.coco_utils import COCOStore


class KeypointMeanAveragePrecision(BaseMetric):
    """Keypoint Mean Average Precision (mAP) metric.
    Uses COCO evaluation metrics for keypoint detection (OKS-based).
    """

    def __init__(
        self,
        iou_type: str = "keypoints",
        kpt_oks_sigmas: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the keypoint mAP metric.

        Parameters
        ----------
        iou_type : str, optional
            Type of IoU to use for evaluation.
        kpt_oks_sigmas : Sequence[float] | None, optional
            OKS sigma values used by COCO keypoint evaluation. Provide one
            value per keypoint for non-COCO keypoint schemas.
        **kwargs : Any
            Additional metric configuration.
        """
        self._store = COCOStore(
            iou_type=iou_type, kpt_oks_sigmas=kpt_oks_sigmas
        )
        super().__init__(**kwargs)

    def metric_keys(self) -> list[str]:
        """Return the ground-truth keys required by the metric.

        Returns
        -------
        list[str]
            Ground-truth key names.
        """
        return ["/boundingbox", "/keypoints"]

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
        target_kpts = target[self.metric_keys()[1]]
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

        for kpts, cls in zip(target_kpts, target_classes, strict=True):
            cls = int(cls)
            if class_index_map is not None:
                cls = int(class_index_map[cls])

            if (
                self._store.category_ids_set is not None
                and cls not in self._store.category_ids_set
            ):
                continue

            kpts_flat = to_coco_kpts_flat(kpts)
            bbox, area, num_kpts = bbox_area_from_keypoints(kpts_flat)

            self._store.add_gt(
                {
                    "image_id": img_id,
                    "category_id": cls,
                    "keypoints": kpts_flat,
                    "num_keypoints": int(num_kpts),
                    "bbox": bbox,
                    "area": float(area),
                    "iscrowd": 0,
                }
            )

        # --- Predictions ---
        detections = predictions.detections
        scores = [det.confidence for det in detections]
        classes = [det.label for det in detections]
        keypoints = np.array(
            [
                [
                    [
                        kp.imageCoordinates.x,
                        kp.imageCoordinates.y,
                        kp.confidence,
                    ]
                    for kp in det.getKeypoints()
                ]
                for det in detections
            ]
        )

        for kpts, score, cls in zip(keypoints, scores, classes, strict=True):
            cls = int(cls)
            if (
                self._store.category_ids_set is not None
                and cls not in self._store.category_ids_set
            ):
                continue

            kpts_flat = to_coco_kpts_flat(kpts)
            if not any(kpts_flat):
                continue

            self._store.add_pred(
                {
                    "image_id": img_id,
                    "category_id": cls,
                    "keypoints": kpts_flat,
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
