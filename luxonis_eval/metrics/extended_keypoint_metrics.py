from collections.abc import Sequence
from typing import Any

import depthai as dai
import numpy as np
import torch
from faster_coco_eval.core import COCO, COCOeval_faster
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_eval.metrics.base_metric import BaseMetric


class ExtendedKeypointMetrics(BaseMetric):
    """Flexible keypoint mAP with train-compatible formulation.

    - accepts arbitrary class counts and arbitrary keypoint counts
    - uses Faster COCO Eval for OKS-based evaluation
    - applies ``sigmas`` and ``area_factor`` the same way as the train-side
      metric
    - expects parser predictions in normalized keypoint coordinates and
      converts them to pixels before evaluation
    """

    def __init__(
        self,
        sigmas: Sequence[float] | None = None,
        kpt_oks_sigmas: Sequence[float] | None = None,
        area_factor: float = 0.53,
        max_dets: int = 20,
        box_format: str = "xyxy",
        **kwargs: Any,
    ) -> None:
        self.sigmas = (
            torch.tensor(
                sigmas if sigmas is not None else kpt_oks_sigmas,
                dtype=torch.float32,
            )
            if sigmas is not None
            or kpt_oks_sigmas is not None
            else None
        )
        self.area_factor = area_factor
        self.max_dets = max_dets
        self.box_format = box_format
        super().__init__(**kwargs)

    def metric_keys(self) -> list[str]:
        return ["/boundingbox", "/keypoints"]

    def _reset_impl(self) -> None:
        self.pred_bboxes: list[Tensor] = []
        self.pred_scores: list[Tensor] = []
        self.pred_classes: list[Tensor] = []
        self.pred_keypoints: list[Tensor] = []

        self.target_bboxes: list[Tensor] = []
        self.target_classes: list[Tensor] = []
        self.target_keypoints: list[Tensor] = []

    def _update_impl(
        self,
        predictions: dai.ImgDetections,
        target: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> None:
        target_boxes = target[self.metric_keys()[0]]
        target_kpts = target[self.metric_keys()[1]]
        width = int(kwargs["width"])
        height = int(kwargs["height"])

        class_index_map = kwargs.get("class_index_map")
        target_converter = kwargs.get("target_converter")
        if target_converter is None:
            raise ValueError(
                "ExtendedKeypointMetrics requires target_converter in ctx."
            )

        pred_boxes_xyxy: list[list[float]] = []
        pred_scores: list[float] = []
        pred_classes: list[int] = []
        pred_keypoints: list[list[float]] = []

        for det in predictions.detections:
            box = det.getBoundingBox().denormalize(
                width, height
            ).getOuterXYWH()
            x = float(box[0].x)
            y = float(box[0].y)
            w = float(box[1].width)
            h = float(box[1].height)
            pred_boxes_xyxy.append([x, y, x + w, y + h])
            pred_scores.append(float(det.confidence))
            pred_classes.append(int(det.label))
            pred_keypoints.append(
                [
                    val
                    for kp in det.getKeypoints()
                    for val in (
                        float(kp.imageCoordinates.x),
                        float(kp.imageCoordinates.y),
                        float(kp.confidence),
                    )
                ]
            )

        pred_kpt_width = self._infer_num_keypoint_values(target_kpts)
        if pred_kpt_width == 0 and pred_keypoints:
            pred_kpt_width = len(pred_keypoints[0])

        self.pred_bboxes.append(
            self._convert_bboxes(self._as_2d_tensor(pred_boxes_xyxy, 4))
        )
        self.pred_scores.append(torch.tensor(pred_scores, dtype=torch.float32))
        self.pred_classes.append(torch.tensor(pred_classes, dtype=torch.int64))
        pred_keypoints_tensor = self._fix_empty_tensor(
            self._as_2d_tensor(
                pred_keypoints,
                pred_kpt_width,
            )
        )
        self.pred_keypoints.append(
            self._denormalize_keypoints(
                pred_keypoints_tensor, width=width, height=height
            )
        )

        target_classes, target_boxes_xywh = target_converter(
            target_boxes, width, height
        )
        target_classes_tensor = torch.tensor(
            target_classes, dtype=torch.int64
        )
        if class_index_map is not None and len(target_classes_tensor) > 0:
            target_classes_tensor = torch.tensor(
                [class_index_map[int(cls)] for cls in target_classes_tensor],
                dtype=torch.int64,
            )
        self.target_classes.append(target_classes_tensor)

        target_boxes_xywh_tensor = self._as_2d_tensor(target_boxes_xywh, 4)
        target_boxes_xyxy = box_convert(
            target_boxes_xywh_tensor, in_fmt="xywh", out_fmt="xyxy"
        )
        self.target_bboxes.append(
            self._convert_bboxes(target_boxes_xyxy.int())
        )

        target_kpts_tensor = torch.tensor(target_kpts, dtype=torch.float32)
        if target_kpts_tensor.ndim == 3:
            target_kpts_tensor = target_kpts_tensor.reshape(
                target_kpts_tensor.shape[0], -1
            )
        elif target_kpts_tensor.numel() == 0:
            target_kpts_tensor = self._as_2d_tensor(
                target_kpts,
                self._infer_num_keypoint_values(target_kpts),
            )
        target_kpts_tensor = self._fix_empty_tensor(target_kpts_tensor)
        if target_kpts_tensor.numel() > 0:
            target_kpts_tensor[:, 0::3] *= width
            target_kpts_tensor[:, 1::3] *= height
        self.target_keypoints.append(
            self._fix_empty_tensor(target_kpts_tensor.int())
        )

    def _compute_impl(self) -> dict[str, float]:
        """Compute final mAP metrics."""
        coco_target = self._get_coco(
            self.target_bboxes,
            self.target_keypoints,
            self.target_classes,
        )
        coco_preds = self._get_coco(
            self.pred_bboxes,
            self.pred_keypoints,
            self.pred_classes,
            self.pred_scores,
        )

        coco_eval_faster = COCOeval_faster(
            coco_target, coco_preds, iouType="keypoints"
        )
        sigmas = self.sigmas
        if sigmas is None:
            sigmas = self._get_default_sigmas()
        coco_eval_faster.params.kpt_oks_sigmas = sigmas.numpy()
        coco_eval_faster.params.maxDets = [self.max_dets]
        coco_eval_faster.run()

        stats = torch.tensor(coco_eval_faster.stats, dtype=torch.float32)
        metrics = {
            "MeanAveragePrecisionKeypoints": float(stats[0]),
            "kpt_map_50": float(stats[1]),
            "kpt_map_75": float(stats[2]),
            "kpt_map_medium": float(stats[3]),
            "kpt_map_large": float(stats[4]),
            "kpt_mar": float(stats[5]),
            "kpt_mar_50": float(stats[6]),
            "kpt_mar_75": float(stats[7]),
            "kpt_mar_medium": float(stats[8]),
            "kpt_mar_large": float(stats[9]),
            "MeanAveragePrecision": float(stats[0]),
        }
        return self._add_f1_metrics(metrics)

    @staticmethod
    def _fix_empty_tensor(tensor: Tensor) -> Tensor:
        if tensor.numel() == 0 and tensor.ndim == 1:
            return tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _as_2d_tensor(values: Any, width: int) -> Tensor:
        array = np.asarray(values, dtype=np.float32)
        if array.size == 0:
            return torch.zeros((0, width), dtype=torch.float32)
        return torch.tensor(array.reshape(-1, width), dtype=torch.float32)

    def _convert_bboxes(self, bboxes: Tensor) -> Tensor:
        bboxes = self._fix_empty_tensor(bboxes)
        if bboxes.numel() > 0:
            bboxes = box_convert(
                bboxes, in_fmt=self.box_format, out_fmt="xywh"
            )
        return bboxes

    @staticmethod
    def _denormalize_keypoints(keypoints: Tensor, width: int, height: int) -> Tensor:
        keypoints = keypoints.clone()
        if keypoints.numel() > 0:
            keypoints[:, 0::3] *= width
            keypoints[:, 1::3] *= height
        return keypoints

    def _get_coco(
        self,
        bboxes_list: list[Tensor],
        keypoints_list: list[Tensor],
        classes_list: list[Tensor],
        scores_list: list[Tensor] | None = None,
    ) -> Any:
        annotations = []

        for i, (bboxes, keypoints, classes) in enumerate(
            zip(bboxes_list, keypoints_list, classes_list, strict=True)
        ):
            for j, (bbox, kpts, class_id) in enumerate(
                zip(bboxes, keypoints, classes, strict=False)
            ):
                annotation: dict[str, Any] = {
                    "id": len(annotations) + 1,
                    "image_id": i,
                    "bbox": bbox.cpu().tolist(),
                    "area": (bbox[2] * bbox[3] * self.area_factor).item(),
                    "category_id": class_id.item(),
                    "keypoints": kpts.cpu().tolist(),
                    "num_keypoints": kpts[2::3].ne(0).sum().item(),
                    "iscrowd": 0,
                }

                if scores_list is not None:
                    annotation["score"] = scores_list[i][j].item()

                annotations.append(annotation)

        coco = COCO()
        coco.dataset = {
            "annotations": annotations,
            "images": [{"id": i} for i in range(len(bboxes_list))],
            "categories": self._get_classes(),
        }
        coco.createIndex()
        return coco

    def _get_classes(self) -> list[dict[str, Any]]:
        if not self.pred_classes and not self.target_classes:
            return []
        classes = torch.cat(self.pred_classes + self.target_classes).unique()
        return [{"id": int(i), "name": str(int(i))} for i in classes.tolist()]

    def _get_default_sigmas(self) -> Tensor:
        n_keypoints = self._get_num_keypoints()
        if n_keypoints == 17:
            return torch.tensor(
                [
                    0.026,
                    0.025,
                    0.025,
                    0.035,
                    0.035,
                    0.079,
                    0.079,
                    0.072,
                    0.072,
                    0.062,
                    0.062,
                    0.107,
                    0.107,
                    0.087,
                    0.087,
                    0.089,
                    0.089,
                ],
                dtype=torch.float32,
            )
        return torch.tensor([0.04] * n_keypoints, dtype=torch.float32)

    def _get_num_keypoints(self) -> int:
        for keypoints_list in (self.target_keypoints, self.pred_keypoints):
            for keypoints in keypoints_list:
                if keypoints.ndim == 2 and keypoints.shape[1] > 0:
                    return keypoints.shape[1] // 3
        return 0

    @staticmethod
    def _infer_num_keypoint_values(keypoints: Any) -> int:
        array = np.asarray(keypoints, dtype=np.float32)
        if array.ndim == 3:
            return int(array.shape[1] * array.shape[2])
        if array.ndim == 2:
            return int(array.shape[1])
        return 0

    @staticmethod
    def _add_f1_metrics(metrics: dict[str, float]) -> dict[str, float]:
        for key in list(metrics.keys()):
            if "map" not in key:
                continue
            mar_key = key.replace("map", "mar")
            if mar_key not in metrics:
                continue
            map_value = metrics[key]
            mar_value = metrics[mar_key]
            denom = map_value + mar_value
            metrics[key.replace("map", "f1")] = (
                0.0 if denom == 0 else 2 * (map_value * mar_value) / denom
            )
        return metrics
