from typing import Any

import depthai as dai
import numpy as np
from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.yolo import (
    YOLOExtendedParser as DepthAINodesYOLOExtendedParser,
)

from luxonis_eval.engines.io import EngineOutput

from .base_parser import BaseParser
from .predictions import Prediction
from .utils.yolo import (
    build_yolo_compute_inputs,
    build_yolo_instance_masks,
)


class YOLOExtendedParser(BaseParser):
    """Parser for YOLO-based detection, segmentation, and pose outputs."""

    _DET_MODE = 0
    _KPTS_MODE = 1
    _SEG_MODE = 2

    def __init__(self, **kwargs: Any) -> None:
        self.subtype: str = kwargs.pop("subtype")
        self.n_classes: int | None = kwargs.pop("n_classes", None)
        self.anchors: list[list[list[float]]] | None = kwargs.pop(
            "anchors", None
        )
        self.strides: list[int] | None = kwargs.pop("strides", None)
        self.conf_threshold: float = kwargs.pop("conf_threshold", 0.5)
        self.iou_threshold: float = kwargs.pop("iou_threshold", 0.5)
        self.n_keypoints: int | None = kwargs.pop("n_keypoints", None)
        self.mask_conf: float = kwargs.pop("mask_conf", 0.5)
        self.max_det: int = kwargs.pop("max_det", 300)
        self.keypoint_label_names: list[str] | None = kwargs.pop(
            "keypoint_label_names", None
        )
        self.keypoint_edges: list[tuple[int, int]] | None = kwargs.pop(
            "keypoint_edges", None
        )
        super().__init__(**kwargs)

    def parse(self, output: EngineOutput) -> Prediction:
        context = self.require_context()
        compute_inputs = build_yolo_compute_inputs(
            output,
            model_spec=context.model_spec,
            class_map=context.class_map,
            subtype=self.subtype,
            n_classes=self.n_classes,
            anchors=self.anchors,
            strides=list(self.strides) if self.strides is not None else None,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_det=self.max_det,
            n_keypoints=self.n_keypoints,
            mask_conf=self.mask_conf,
            keypoint_label_names=self.keypoint_label_names,
            keypoint_edges=self.keypoint_edges,
        )
        raw_outputs_values = None
        if compute_inputs.masks_outputs_values is not None:
            raw_outputs_values = [
                value.copy() for value in compute_inputs.outputs_values
            ]
        payload = DepthAINodesYOLOExtendedParser.compute(compute_inputs)

        mode = int(payload["mode"])
        if mode == self._KPTS_MODE:
            return Prediction(
                detections=create_detection_message(
                    bboxes=payload["bboxes"],
                    scores=payload["scores"],
                    labels=payload["labels"],
                    label_names=payload["label_names"],
                    keypoints=payload["keypoints"],
                    keypoints_scores=payload["keypoints_scores"],
                    keypoint_label_names=payload.get(
                        "keypoint_label_names",
                        self.keypoint_label_names,
                    ),
                    keypoint_edges=payload.get(
                        "keypoint_edges",
                        self.keypoint_edges,
                    ),
                )
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
            return Prediction(
                detections=message,
                instance_masks=instance_masks,
            )

        return Prediction(
            detections=create_detection_message(
                bboxes=payload["bboxes"],
                scores=payload["scores"],
                labels=payload["labels"],
                label_names=payload["label_names"],
            )
        )
