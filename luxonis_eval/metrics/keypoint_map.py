from collections.abc import Sequence
from typing import Any

import depthai as dai
import numpy as np
from loguru import logger

from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.metrics.metrics_utils import (
    bbox_area_from_keypoints,
    ldf_norm_kpts_to_coco_kpts_flat,
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
        debug_log_samples: int = 0,
        debug_max_instances: int = 5,
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
        debug_log_samples : int, optional
            Number of samples for which GT/prediction comparisons should be
            logged.
        debug_max_instances : int, optional
            Maximum number of GT/pred instances to print per debugged sample.
        **kwargs : Any
            Additional metric configuration.
        """
        self._store = COCOStore(
            iou_type=iou_type, kpt_oks_sigmas=kpt_oks_sigmas
        )
        self._debug_log_samples = debug_log_samples
        self._debug_max_instances = debug_max_instances
        self._debug_logged = 0
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
        self._debug_logged = 0

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
        target_class_map: dict[int, str] = kwargs.get("target_class_map", {})

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

            kpts_flat = ldf_norm_kpts_to_coco_kpts_flat(kpts, width, height)
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

            # Parser keypoints are normalized to [0, 1]; COCO expects pixels.
            kpts = kpts.copy()
            if kpts.size != 0:
                kpts[:, 0] *= width
                kpts[:, 1] *= height
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

        if self._debug_logged < self._debug_log_samples:
            self._log_debug_sample(
                detections=detections,
                target_boxes=target_boxes,
                target_kpts=target_kpts,
                width=width,
                height=height,
                class_map=class_map,
                target_class_map=target_class_map,
                class_index_map=class_index_map,
            )
            self._debug_logged += 1

    def _compute_impl(self) -> dict[str, float]:
        """Compute final mAP metrics.

        Returns
        -------
        dict[str, float]
            Computed mAP results.
        """
        return self._store.evaluate()

    def _log_debug_sample(
        self,
        *,
        detections: Sequence[Any],
        target_boxes: np.ndarray,
        target_kpts: np.ndarray,
        width: int,
        height: int,
        class_map: dict[int, str],
        target_class_map: dict[int, str],
        class_index_map: dict[int, int] | None,
    ) -> None:
        """Log GT and prediction details for a single sample."""
        sample_idx = self._debug_logged + 1
        logger.warning(
            "Keypoint debug sample {}: image={}x{}, gt_instances={}, pred_instances={}",
            sample_idx,
            width,
            height,
            len(target_boxes),
            len(detections),
        )

        max_items = self._debug_max_instances

        for i, (box, kpts) in enumerate(
            zip(target_boxes[:max_items], target_kpts[:max_items], strict=False)
        ):
            raw_cls = int(box[0])
            mapped_cls = (
                int(class_index_map[raw_cls])
                if class_index_map is not None
                else raw_cls
            )
            gt_bbox_px = np.array(
                [
                    box[1] * width,
                    box[2] * height,
                    box[3] * width,
                    box[4] * height,
                ],
                dtype=float,
            )
            gt_kpts_norm = np.asarray(kpts, dtype=float).reshape(-1, 3)
            gt_kpts_px = gt_kpts_norm.copy()
            gt_kpts_px[:, 0] *= width
            gt_kpts_px[:, 1] *= height
            logger.info(
                "GT[{}] ldf_cls={}({}) mapped_cls={}({}) bbox_xywh_px={} kpts_norm={} kpts_px={}",
                i,
                raw_cls,
                target_class_map.get(raw_cls, str(raw_cls)),
                mapped_cls,
                class_map.get(mapped_cls, str(mapped_cls)),
                np.round(gt_bbox_px, 2).tolist(),
                np.round(gt_kpts_norm, 4).tolist(),
                np.round(gt_kpts_px, 2).tolist(),
            )

        for i, det in enumerate(detections[:max_items]):
            pred_bbox = det.getBoundingBox().denormalize(
                width, height
            ).getOuterXYWH()
            pred_bbox_xywh_px = [
                pred_bbox[0].x,
                pred_bbox[0].y,
                pred_bbox[1].width,
                pred_bbox[1].height,
            ]
            pred_kpts_norm = np.array(
                [
                    [
                        kp.imageCoordinates.x,
                        kp.imageCoordinates.y,
                        kp.confidence,
                    ]
                    for kp in det.getKeypoints()
                ],
                dtype=float,
            )
            pred_kpts_px = pred_kpts_norm.copy()
            if pred_kpts_px.size != 0:
                pred_kpts_px[:, 0] *= width
                pred_kpts_px[:, 1] *= height
            logger.info(
                "PRED[{}] cls={}({}) score={:.4f} bbox_xywh_px={} kpts_norm={} kpts_px={}",
                i,
                int(det.label),
                class_map.get(int(det.label), str(int(det.label))),
                float(det.confidence),
                np.round(pred_bbox_xywh_px, 2).tolist(),
                np.round(pred_kpts_norm, 4).tolist(),
                np.round(pred_kpts_px, 2).tolist(),
            )
