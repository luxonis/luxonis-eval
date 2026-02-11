from typing import Any

import depthai as dai
import numpy as np
from depthai_nodes.node.parsers.utils.yolo import (
    YOLOSubtype,
    decode_yolo_output,
    parse_kpts,
)
from loguru import logger

from .base_parser import BaseParser


class YOLOKeypointDetectionParser(BaseParser):
    """Parser for YOLO-based keypoint detection model outputs."""

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
            kpts_output_names = sorted(
                [name for name in layer_names if "kpt_output" in name]
            )
            kpts_outputs = [
                raw_output.getTensor(
                    o,
                    dequantize=True,
                ).astype(np.float32)  # type: ignore
                for o in kpts_output_names
            ]
        elif isinstance(raw_output, list):
            outputs_names = [f"output_{i}" for i in range(len(raw_output))]
            outputs_values = raw_output[:3]
            kpts_outputs = raw_output[3:]
        else:
            raise TypeError(
                "raw_output must be dai.NNData or list[np.ndarray]"
            )

        strides = [8, 16, 32]
        input_shape = tuple(
            dim * strides[0] for dim in outputs_values[0].shape[2:4]
        )
        n_classes = outputs_values[0].shape[1] - 5
        num_keypoints = kpts_outputs[0].shape[1] // 3

        results = decode_yolo_output(
            yolo_outputs=outputs_values,
            strides=strides,
            anchors=None,
            kpts=kpts_outputs,
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

            kpts = parse_kpts(other, num_keypoints, input_shape)  # type: ignore
            additional_output.append(kpts)

        return {
            "bboxes": np.asarray(bboxes),
            "scores": np.asarray(scores, dtype=np.float32),
            "classes": np.asarray(labels, dtype=np.int64),
            "class_names": label_names,
            "keypoints": np.asarray(additional_output),
        }
