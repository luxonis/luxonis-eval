from typing import Any

import depthai as dai
from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.yolo import (
    YOLOExtendedParser as DepthAINodesYOLOExtendedParser,
)

from luxonis_eval.engines.io import EngineOutput

from .base_parser import BaseParser
from .utils.yolo import build_yolo_compute_inputs


class YOLOExtendedParser(BaseParser):
    """Parser for YOLO-based detection, segmentation, and pose
    outputs."""

    _DET_MODE = 0
    _KPTS_MODE = 1
    _SEG_MODE = 2

    def __init__(
        self,
        subtype: str,
        n_classes: int | None = None,
        anchors: list[list[list[float]]] | None = None,
        strides: list[int] | None = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        n_keypoints: int | None = None,
        mask_conf: float = 0.5,
        max_det: int = 300,
        keypoint_label_names: list[str] | None = None,
        keypoint_edges: list[tuple[int, int]] | None = None,
        **kwargs: Any,
    ) -> None:
        self.subtype = subtype
        self.n_classes = n_classes
        self.anchors = anchors
        self.strides = strides
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.n_keypoints = n_keypoints
        self.mask_conf = mask_conf
        self.max_det = max_det
        self.keypoint_label_names = keypoint_label_names
        self.keypoint_edges = keypoint_edges
        super().__init__(**kwargs)

    def parse(self, output: EngineOutput) -> dai.ImgDetections:
        context = self.context
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
        payload = DepthAINodesYOLOExtendedParser.compute(compute_inputs)

        mode = int(payload["mode"])
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
                    self.keypoint_label_names,
                ),
                keypoint_edges=payload.get(
                    "keypoint_edges",
                    self.keypoint_edges,
                ),
            )

        if mode == self._SEG_MODE:
            return create_detection_message(
                bboxes=payload["bboxes"],
                scores=payload["scores"],
                labels=payload["labels"],
                label_names=payload["label_names"],
                masks=payload["masks"],
            )

        return create_detection_message(
            bboxes=payload["bboxes"],
            scores=payload["scores"],
            labels=payload["labels"],
            label_names=payload["label_names"],
        )
