from collections.abc import Sequence
from typing import Any

import numpy as np

from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.utils.coco_utils import COCOStore


class KeypointMeanAveragePrecision(BaseMetric):
    """Keypoint Mean Average Precision (mAP) metric.
    Uses COCO evaluation metrics for keypoint detection (OKS-based).
    """

    def __init__(self, iou_type: str = "keypoints", **kwargs: Any) -> None:
        """Initialize the keypoint mAP metric.

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
        return ["/boundingbox", "/keypoints"]

    def _reset_impl(self) -> None:
        """Reset internal metric state."""
        self._store.reset()

    @staticmethod
    def _to_coco_kpts_flat(kpts: np.ndarray) -> list[float]:
        """Convert keypoints to COCO flat list [x,y,v,...].

        Parameters
        ----------
        kpts : np.ndarray
            Keypoints array, either flat (3K,) or shaped (K,3).

        Returns
        -------
        list[float]
            Flattened keypoints list in COCO format [x,y,v,...].
        """
        kpts = np.asarray(kpts, dtype=float)

        if kpts.ndim == 1:
            if kpts.size % 3 != 0:
                raise ValueError(
                    f"Keypoints flat array must have length multiple of 3; got {kpts.size}."
                )
            flat = kpts
        elif kpts.ndim == 2 and kpts.shape[1] == 3:
            flat = kpts.reshape(-1)
        else:
            raise ValueError(
                f"Unsupported keypoints shape {kpts.shape}; expected (K,3) or (3K,)."
            )

        return [float(x) for x in flat.tolist()]

    @staticmethod
    def _bbox_area_from_keypoints(
        kpts_flat: Sequence[float],
    ) -> tuple[list[float], float, int]:
        """Compute [x,y,w,h], area, num_keypoints from COCO flat keypoints.

        Parameters
        ----------
        kpts_flat : Sequence[float]
            Flattened keypoints list in COCO format [x,y,v,...].

        Returns
        -------
        tuple[list[float], float, int]
            Bounding box [x,y,w,h], area, and number of keypoints.
        """
        arr = np.asarray(kpts_flat, dtype=float)
        if arr.size % 3 != 0:
            raise ValueError("kpts_flat must be a multiple of 3.")

        xs = arr[0::3]
        ys = arr[1::3]
        vs = arr[2::3]

        labeled = vs > 0
        num_kpts = int(np.count_nonzero(labeled))

        if num_kpts == 0:
            return [0.0, 0.0, 0.0, 0.0], 0.0, 0

        x_min = float(np.min(xs[labeled]))
        y_min = float(np.min(ys[labeled]))
        x_max = float(np.max(xs[labeled]))
        y_max = float(np.max(ys[labeled]))

        w = max(0.0, x_max - x_min)
        h = max(0.0, y_max - y_min)
        area = float(w * h)

        return [x_min, y_min, float(w), float(h)], area, num_kpts

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
        target_boxes = target[self.metric_keys()[0]]
        target_kpts = target[self.metric_keys()[1]]
        width = int(kwargs["width"])
        height = int(kwargs["height"])

        native_class_map: dict[int, str] = kwargs.get("native_class_map", {})
        category_ids: Sequence[int] | None = kwargs.get("category_ids")
        class_index_map = kwargs.get("class_index_map")
        target_converter = kwargs.get("target_converter")
        if target_converter is None:
            raise ValueError(
                "KeypointMeanAveragePrecision requires target_converter in ctx."
            )

        self._store.init_categories_once(
            native_class_map=native_class_map, category_ids=category_ids
        )
        img_id = self._store.new_image(width=width, height=height)

        # --- GT ---
        target_classes, target_boxes_xywh = target_converter(
            target_boxes, width, height
        )

        for kpts, box, cls in zip(
            target_kpts, target_boxes_xywh, target_classes, strict=True
        ):
            cls = int(cls)
            if class_index_map is not None:
                cls = int(class_index_map[cls])

            if (
                self._store.category_ids_set is not None
                and cls not in self._store.category_ids_set
            ):
                continue

            kpts_flat = self._to_coco_kpts_flat(kpts)
            bbox, area, num_kpts = self._bbox_area_from_keypoints(kpts_flat)

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

        # --- DT ---
        keypoints = np.asarray(predictions["keypoints"], dtype=float)
        scores = np.asarray(predictions["scores"], dtype=float)
        classes = np.asarray(predictions["classes"], dtype=int)

        for kpts, score, cls in zip(keypoints, scores, classes, strict=True):
            cls = int(cls)
            if (
                self._store.category_ids_set is not None
                and cls not in self._store.category_ids_set
            ):
                continue

            kpts_flat = self._to_coco_kpts_flat(kpts)
            if not any(kpts_flat):
                continue

            self._store.add_dt(
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
