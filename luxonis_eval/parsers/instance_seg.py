from typing import Any

import depthai as dai
from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.yolo import (
    YOLOExtendedParser as DepthAINodesYOLOExtendedParser,
)

from luxonis_eval.engines.base_engine import ModelSpec
from luxonis_eval.engines.io import EngineOutput
from luxonis_eval.utils._depthai_nodes import build_yolo_compute_kwargs
from .base_parser import BaseParser


class YOLOInstanceSegmentationParser(BaseParser):
    """Parser for YOLO-based instance segmentation model outputs."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the YOLO instance segmentation parser."""
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
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        mask_conf: float = 0.5,
        max_det: int = 300,
        **kwargs: Any,
    ) -> dai.ImgDetections:
        """Parse backend output into detection predictions.

        Parameters
        ----------
        output : EngineOutput
            Engine-normalized inference output.
        model_spec : ModelSpec
            Resolved model IO metadata.
        class_map : dict[int, str]
            Mapping from class indices to class names.
        subtype : str
            YOLO model subtype.
        n_classes : int | None, optional
            Number of classes.
        anchors : list[list[list[float]]] | None, optional
            Anchor boxes.
        conf_threshold : float, default=0.5
            Confidence threshold.
        iou_threshold : float, default=0.5
            IoU threshold.
        mask_conf : float, default=0.5
            Mask confidence threshold.
        max_det : int, default=300
            Maximum detections.
        **kwargs : Any
            Additional parser arguments.

        Returns
        -------
        dai.ImgDetections
            Detection results including boxes, scores, classes, and metadata.
        """
        payload = DepthAINodesYOLOExtendedParser.compute(
            **build_yolo_compute_kwargs(
                output,
                model_spec=model_spec,
                class_map=class_map,
                subtype=subtype,
                n_classes=n_classes,
                anchors=anchors,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_det=max_det,
                mask_conf=mask_conf,
            )
        )

        return create_detection_message(
            bboxes=payload["bboxes"],
            scores=payload["scores"],
            labels=payload["labels"],
            label_names=payload["label_names"],
            masks=payload["masks"],
        )
