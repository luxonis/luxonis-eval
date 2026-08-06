from typing import Any

import depthai as dai
import numpy as np
from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.yolo import (
    YOLOExtendedParser as DepthAINodesYOLOExtendedParser,
)

from luxonis_eval.engines.base_engine import ModelSpec
from luxonis_eval.engines.io import EngineOutput
from luxonis_eval.utils.depthai_nodes import (
    build_yolo_compute_inputs,
    build_yolo_instance_masks,
)

from .base_parser import BaseParser


prediction_instance_masks_by_id: dict[int, np.ndarray] = {}


def store_prediction_instance_masks(
    predictions: dai.ImgDetections, instance_masks: np.ndarray
) -> None:
    prediction_instance_masks_by_id[id(predictions)] = instance_masks


def get_prediction_instance_masks(
    predictions: dai.ImgDetections,
) -> np.ndarray | None:
    return prediction_instance_masks_by_id.get(id(predictions))


def clear_prediction_metadata(predictions: Any) -> None:
    prediction_instance_masks_by_id.pop(id(predictions), None)


class YOLOExtendedParser(BaseParser):
    """Parser for YOLO-based detection, segmentation, and pose outputs."""

    _DET_MODE = 0
    _KPTS_MODE = 1
    _SEG_MODE = 2

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def parse(
        self,
        output: EngineOutput,
        model_spec: ModelSpec,
        *,
        class_map: dict[int, str],
        subtype: str,
        n_classes: int | None = None,
        anchors: list[list[list[float]]] | None = None,
        strides: list[int] | tuple[int, ...] | None = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        n_keypoints: int | None = None,
        mask_conf: float = 0.5,
        max_det: int = 300,
        keypoint_label_names: list[str] | None = None,
        keypoint_edges: list[tuple[int, int]] | None = None,
        **kwargs: Any,
    ) -> dai.ImgDetections:
        del kwargs
        compute_inputs = build_yolo_compute_inputs(
            output,
            model_spec=model_spec,
            class_map=class_map,
            subtype=subtype,
            n_classes=n_classes,
            anchors=anchors,
            strides=list(strides) if strides is not None else None,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            n_keypoints=n_keypoints,
            mask_conf=mask_conf,
            keypoint_label_names=keypoint_label_names,
            keypoint_edges=keypoint_edges,
        )
        raw_outputs_values = None
        if any("mask" in name for name in compute_inputs.layer_names):
            raw_outputs_values = [
                value.copy() for value in compute_inputs.outputs_values
            ]
        payload = DepthAINodesYOLOExtendedParser.compute(compute_inputs)

        mode = self._resolve_mode(payload)
        if mode == self._KPTS_MODE:
            return create_detection_message(
                bboxes=payload["bboxes"],
                scores=payload["scores"],
                labels=payload["labels"],
                label_names=payload["label_names"],
                keypoints=payload["keypoints"],
                keypoints_scores=payload["keypoints_scores"],
                keypoint_label_names=payload.get(
                    "keypoint_label_names",
                    keypoint_label_names,
                ),
                keypoint_edges=payload.get(
                    "keypoint_edges",
                    keypoint_edges,
                ),
            )

        if mode == self._SEG_MODE:
            message = create_detection_message(
                bboxes=payload["bboxes"],
                scores=payload["scores"],
                labels=payload["labels"],
                label_names=payload["label_names"],
                masks=payload["masks"],
            )
            instance_masks = build_yolo_instance_masks(
                compute_inputs,
                outputs_values=raw_outputs_values,
            )
            if instance_masks.shape[0] != len(message.detections):
                raise ValueError(
                    "YOLOExtendedParser received mismatched segmentation outputs: "
                    f"{len(message.detections)} detections but "
                    f"{instance_masks.shape[0]} rebuilt instance masks."
                )
            store_prediction_instance_masks(message, instance_masks)
            return message

        return create_detection_message(
            bboxes=payload["bboxes"],
            scores=payload["scores"],
            labels=payload["labels"],
            label_names=payload["label_names"],
        )

    def _resolve_mode(self, payload: dict[str, Any]) -> int:
        mode = payload.get("mode")
        if mode is not None:
            return int(mode)
        keypoints = payload.get("keypoints")
        if keypoints is not None:
            keypoints_size = (
                int(keypoints.size)
                if hasattr(keypoints, "size")
                else len(keypoints)
            )
            if keypoints_size > 0:
                return self._KPTS_MODE
        if payload.get("masks") is not None:
            return self._SEG_MODE
        return self._DET_MODE
