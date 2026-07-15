from typing import Any

import depthai as dai
import numpy as np
from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.yolo import (
    YOLOComputeInputs,
)
from depthai_nodes.node.parsers.yolo import (
    YOLOExtendedParser as DepthAINodesYOLOExtendedParser,
)

from luxonis_eval.engines.base_engine import ModelSpec
from luxonis_eval.engines.io import EngineOutput
from luxonis_eval.utils.depthai_nodes import (
    build_luxonistrain_instance_masks,
    build_yolo_compute_kwargs,
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
    """Parser for YOLO-based detection, segmentation, and pose
    outputs."""

    _DET_MODE = 0
    _KPTS_MODE = 1
    _SEG_MODE = 2

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the YOLO parser."""
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
        strides: list[int] | None = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        mask_conf: float = 0.5,
        max_det: int = 300,
        keypoint_label_names: list[str] | None = None,
        keypoint_edges: list[tuple[int, int]] | None = None,
        **kwargs: Any,
    ) -> dai.ImgDetections:
        """Parse backend output into YOLO predictions."""
        del kwargs
        compute_kwargs = build_yolo_compute_kwargs(
            output,
            model_spec=model_spec,
            class_map=class_map,
            subtype=subtype,
            n_classes=n_classes,
            anchors=anchors,
            strides=strides,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            mask_conf=mask_conf,
            keypoint_label_names=keypoint_label_names,
            keypoint_edges=keypoint_edges,
        )
        payload = DepthAINodesYOLOExtendedParser.compute(
            YOLOComputeInputs(**compute_kwargs)
        )

        mode = int(payload["mode"])
        if mode == self._KPTS_MODE:
            return create_detection_message(
                bboxes=payload["bboxes"],
                scores=payload["scores"],
                labels=payload["labels"],
                label_names=payload["label_names"],
                keypoints=payload["keypoints"],
                keypoints_scores=payload["keypoints_scores"],
                keypoint_label_names=payload["keypoint_label_names"],
                keypoint_edges=payload["keypoint_edges"],
            )

        if mode == self._SEG_MODE:
            message = create_detection_message(
                bboxes=payload["bboxes"],
                scores=payload["scores"],
                labels=payload["labels"],
                label_names=payload["label_names"],
                masks=payload["masks"],
            )
            instance_masks = build_luxonistrain_instance_masks(
                compute_kwargs
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
