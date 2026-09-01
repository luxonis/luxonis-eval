from collections.abc import Sequence
from typing import Any

import depthai as dai
import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.metrics.metrics_utils import (
    detection_to_coco_xywh,
)
from luxonis_eval.parsers.yolo import get_prediction_instance_masks


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
        if iou_type != "segm":
            raise ValueError(
                "MaskMeanAveragePrecision is fixed to instance-segmentation "
                "evaluation and only supports iou_type='segm'."
            )
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
        self.metric = MeanAveragePrecision(
            iou_type=("bbox", "segm"),
            backend="faster_coco_eval",
        )

    def update(
        self,
        predictions: dai.ImgDetections,
        target: dict[str, np.ndarray],
    ) -> None:
        """Update internal metric state.

        Parameters
        ----------
        predictions : dai.ImgDetections
            Model predictions.
        target : dict[str, np.ndarray]
            Ground-truth data.
        """
        context = self.require_context()
        target_boxes = target[self.required_target_keys()[0]]
        target_masks = target[self.required_target_keys()[1]]

        width = context.width
        height = context.height

        category_ids: Sequence[int] = context.category_ids
        class_index_map = context.class_index_map
        target_converter = context.target_converter

        target_classes, target_boxes_xywh = target_converter(
            target_boxes, width, height
        )
        if class_index_map is not None:
            target_classes = np.array(
                [class_index_map[int(cls)] for cls in target_classes],
                dtype=np.int64,
            )
        category_ids_set = (
            {int(category_id) for category_id in category_ids}
            if category_ids is not None
            else None
        )

        detections = predictions.detections
        masks = self._resolve_prediction_instance_masks(
            get_prediction_instance_masks(predictions),
            n_detections=len(detections),
            height=height,
            width=width,
        )

        pred_boxes_xyxy: list[list[float]] = []
        pred_scores: list[float] = []
        pred_classes: list[int] = []
        pred_masks: list[np.ndarray] = []
        for det, mask in zip(detections, masks, strict=True):
            cls = int(det.label)
            if category_ids_set is not None and cls not in category_ids_set:
                continue

            if mask.sum() == 0:
                continue

            x, y, w, h = detection_to_coco_xywh(det, width, height)
            pred_boxes_xyxy.append([x, y, x + w, y + h])
            pred_scores.append(float(det.confidence))
            pred_classes.append(cls)
            pred_masks.append(mask.astype(bool, copy=False))

        target_boxes_xyxy = self._xywh_to_xyxy(target_boxes_xywh)
        if category_ids_set is not None:
            keep = np.array(
                [int(cls) in category_ids_set for cls in target_classes],
                dtype=bool,
            )
            target_classes = target_classes[keep]
            target_boxes_xyxy = target_boxes_xyxy[keep]
            target_masks = target_masks[keep]

        preds = [
            {
                "boxes": self._as_boxes_tensor(pred_boxes_xyxy),
                "scores": self._as_scores_tensor(pred_scores),
                "labels": self._as_labels_tensor(pred_classes),
                "masks": self._as_masks_tensor(pred_masks, height, width),
            }
        ]
        targets = [
            {
                "boxes": self._as_boxes_tensor(target_boxes_xyxy.tolist()),
                "labels": self._as_labels_tensor(target_classes.tolist()),
                "masks": self._as_target_masks_tensor(
                    target_masks, height, width
                ),
            }
        ]
        self.metric.update(preds, targets)

    def compute(self) -> dict[str, float]:
        """Compute final mAP metrics.

        Returns
        -------
        dict[str, float]
            Computed mAP results.
        """
        metrics = self.metric.compute()
        return {
            "AP": float(metrics["segm_map"]),
            "AP50": float(metrics["segm_map_50"]),
        }

    @staticmethod
    def _resolve_prediction_instance_masks(
        raw_masks: np.ndarray | None,
        *,
        n_detections: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        if raw_masks is None:
            raise ValueError(
                "MaskMeanAveragePrecision requires raw per-instance masks "
                "for the prediction message."
            )

        masks = np.asarray(raw_masks)
        if masks.size == 0:
            if n_detections != 0:
                raise ValueError(
                    "MaskMeanAveragePrecision received no raw instance masks "
                    f"for {n_detections} detections."
                )
            return np.zeros((0, height, width), dtype=np.uint8)

        if masks.ndim != 3:
            raise ValueError(
                f"Unsupported raw instance mask rank {masks.ndim}. "
                "Expected shape (N, H, W)."
            )

        if masks.shape != (n_detections, height, width):
            raise ValueError(
                "Raw instance masks do not align with detections. Expected "
                f"({n_detections}, {height}, {width}), got {masks.shape}."
            )

        return masks.astype(np.uint8, copy=False)

    @staticmethod
    def _xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
        boxes_xywh = np.asarray(boxes_xywh, dtype=np.float32)
        if boxes_xywh.size == 0:
            return np.zeros((0, 4), dtype=np.float32)

        boxes_xyxy = boxes_xywh.copy()
        boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes_xyxy[:, 2]
        boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes_xyxy[:, 3]
        return boxes_xyxy

    @staticmethod
    def _as_boxes_tensor(boxes: list[list[float]]) -> torch.Tensor:
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

    @staticmethod
    def _as_scores_tensor(scores: list[float]) -> torch.Tensor:
        if not scores:
            return torch.zeros((0,), dtype=torch.float32)
        return torch.tensor(scores, dtype=torch.float32)

    @staticmethod
    def _as_labels_tensor(labels: list[int]) -> torch.Tensor:
        if not labels:
            return torch.zeros((0,), dtype=torch.int64)
        return torch.tensor(labels, dtype=torch.int64)

    @staticmethod
    def _as_masks_tensor(
        masks: list[np.ndarray], height: int, width: int
    ) -> torch.Tensor:
        if not masks:
            return torch.zeros((0, height, width), dtype=torch.bool)
        return torch.from_numpy(np.stack(masks, axis=0)).to(torch.bool)

    @staticmethod
    def _as_target_masks_tensor(
        masks: np.ndarray, height: int, width: int
    ) -> torch.Tensor:
        masks = np.asarray(masks)
        if masks.size == 0:
            return torch.zeros((0, height, width), dtype=torch.bool)
        return torch.from_numpy(masks.astype(bool, copy=False))
