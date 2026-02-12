from typing import Any

import cv2
import depthai as dai
import numpy as np
from depthai_nodes.node.parsers.utils.bbox_format_converters import (
    normalize_bboxes,
    xyxy_to_xywh,
)
from depthai_nodes.node.parsers.utils.masks_utils import (
    get_segmentation_outputs,
    process_single_mask,
)
from depthai_nodes.node.parsers.utils.yolo import (
    YOLOSubtype,
    decode_yolo_output,
)
from loguru import logger

from .base_parser import BaseParser


class YOLOInstanceSegmentationParser(BaseParser):
    """Parser for YOLO-based instance segmentation model outputs."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the YOLO instance segmentation parser."""
        super().__init__(**kwargs)

    def parse(
        self,
        raw_output: dai.NNData | list[np.ndarray],
        *,
        class_map: dict[int, str],
        subtype: str,
        n_classes: int | None = None,
        anchors: list[list[list[float]]] | None = None,
        conf_thres: float = 0.001,
        iou_thres: float = 0.7,
        max_det: int = 300,
        **kwargs: Any,
    ) -> dict[str, np.ndarray | list]:
        """Parse backend output into detection predictions.

        Parameters
        ----------
        raw_output : dai.NNData | list[np.ndarray]
            Backend inference output.
        class_map : dict[int, str]
            Mapping from class indices to class names.
        subtype : str
            YOLO model subtype.
        n_classes : int | None, optional
            Number of classes.
        anchors : list[list[list[float]]] | None, optional
            Anchor boxes.
        conf_thres : float, default=0.001
            Confidence threshold.
        iou_thres : float, default=0.7
            IoU threshold.
        max_det : int, default=300
            Maximum detections.
        **kwargs : Any
            Additional parser arguments.

        Returns
        -------
        dict[str, np.ndarray | list]
            Detection results including boxes, scores, classes, and metadata.
        """
        try:
            subtype = YOLOSubtype(subtype.lower())
        except ValueError as err:
            raise ValueError(
                f"Invalid YOLO subtype {subtype}. Supported YOLO subtypes are {[e.value for e in YOLOSubtype][:-1]}."
            ) from err

        if isinstance(raw_output, dai.NNData):
            layer_names = raw_output.getAllLayerNames()
            logger.debug(f"Processing output with layers: {layer_names}")

            outputs_names = sorted(
                [n for n in layer_names if "_yolo" in n or "yolo-" in n]
            )
            outputs_values = [
                raw_output.getTensor(
                    o,
                    dequantize=True,
                    storageOrder=dai.TensorInfo.StorageOrder.NCHW,
                ).astype(np.float32)  # type: ignore
                for o in outputs_names
            ]
            (
                masks_outputs_values,
                protos_output,
                protos_len,
            ) = get_segmentation_outputs(raw_output)
        elif isinstance(raw_output, list):
            outputs_names = [f"output_{i}" for i in range(len(raw_output))]
            outputs_values = raw_output[:3]
            masks_outputs_values = raw_output[3:-1]
            protos_output = raw_output[-1]
            protos_len = protos_output.shape[1]
        else:
            raise TypeError(
                "raw_output must be dai.NNData or list[np.ndarray]"
            )

        strides = (
            [8, 16, 32]
            if subtype
            not in [YOLOSubtype.V3UT, YOLOSubtype.V3T, YOLOSubtype.V4T]
            else [16, 32]
        )
        input_shape = tuple(
            dim * strides[0] for dim in outputs_values[0].shape[2:4]
        )
        final_anchors: np.ndarray | None = (
            np.array(anchors).reshape(len(strides), -1) if anchors else None
        )
        inferred_n_classes = (
            outputs_values[0].shape[1] - 5
            if not final_anchors
            else (outputs_values[0].shape[1] // final_anchors.shape[0]) - 5
        )
        if n_classes and inferred_n_classes != n_classes:
            raise ValueError(
                f"The provided number of classes {n_classes} does not match the model's {inferred_n_classes}."
            )

        results = decode_yolo_output(
            yolo_outputs=outputs_values,
            strides=strides,
            anchors=final_anchors,
            kpts=None,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            num_classes=inferred_n_classes,
            det_mode=False,
            subtype=subtype,
            max_nms=max_det,
        )

        bboxes, labels, label_names, scores, additional_output = (
            [],
            [],
            [],
            [],
            [],
        )
        instance_masks: list[np.ndarray] = []
        for i in range(results.shape[0]):
            bbox, conf, label, other = (
                results[i, :4],
                results[i, 4],
                results[i, 5].astype(int),
                results[i, 6:],
            )
            bboxes.append(bbox)
            scores.append(float(conf))
            labels.append(int(label))
            label_names.append(class_map[int(label)])
            additional_output.append(other)

            bbox_xywh = xyxy_to_xywh(bbox.reshape(1, 4))
            bbox_xywh_norm = normalize_bboxes(
                bbox_xywh, height=input_shape[0], width=input_shape[1]
            )[0]

            seg_coeff = other.astype(int)
            hi, ai, xi, yi = seg_coeff
            mask_coeff = masks_outputs_values[hi][
                0, ai * protos_len : (ai + 1) * protos_len, yi, xi
            ]
            mask = process_single_mask(
                protos_output[0], mask_coeff, 0.4, bbox_xywh_norm
            )

            resized_mask = cv2.resize(
                mask,
                (input_shape[1], input_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            bin_mask = resized_mask > 0
            instance_masks.append(bin_mask)

        return {
            "masks": np.asarray(instance_masks),
            "bboxes": np.asarray(bboxes),
            "scores": np.asarray(scores, dtype=np.float32),
            "classes": np.asarray(labels, dtype=np.int64),
            "class_names": label_names,
            "extra": np.asarray(additional_output),
        }
