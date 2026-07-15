from dataclasses import dataclass
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
from luxonis_eval.utils.depthai_nodes import build_yolo_compute_kwargs

from .base_parser import BaseParser


@dataclass
class ParsedImgDetections:
    message: dai.ImgDetections
    instance_masks: np.ndarray | None = None

    @property
    def detections(self) -> Any:
        return self.message.detections

    def getCvSegmentationMask(self) -> Any:
        return self.message.getCvSegmentationMask()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.message, name)


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
    ) -> ParsedImgDetections | dai.ImgDetections:
        """Parse backend output into YOLO predictions."""
        del kwargs
        payload = DepthAINodesYOLOExtendedParser.compute(
            YOLOComputeInputs(
                **build_yolo_compute_kwargs(
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
            )
        )

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
            return ParsedImgDetections(
                message=message,
                instance_masks=np.asarray(
                    payload.get("instance_masks", np.zeros((0, 0, 0)))
                ),
            )

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
