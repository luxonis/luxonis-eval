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

    def parse(
        self,
        raw_output: dai.NNData | list[np.ndarray],
        **kwargs: Any,
    ) -> dict[str, np.ndarray | list]:
        """Parse backend output into detection predictions.

        Parameters
        ----------
        raw_output : dai.NNData | list[np.ndarray]
            Backend inference output.
        **kwargs : Any
            Additional parser arguments.

        Returns
        -------
        dict[str, np.ndarray | list]
            Detection results including boxes, scores, classes, and metadata.
        """
        # Retrieve additional task-specific options
        class_map = kwargs.get("class_map", {})

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

        strides = [8, 16, 32]
        input_shape = tuple(
            dim * strides[0] for dim in outputs_values[0].shape[2:4]
        )
        n_classes = outputs_values[0].shape[1] - 5

        results = decode_yolo_output(
            yolo_outputs=outputs_values,
            strides=strides,
            anchors=None,
            kpts=None,
            conf_thres=0.4,
            iou_thres=0.45,
            num_classes=n_classes,
            det_mode=False,
            subtype=YOLOSubtype.V8,
            max_nms=300,
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
