from typing import Any

import cv2
import depthai as dai
import numpy as np
from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.utils import (
    normalize_bboxes,
    sigmoid,
    xyxy_to_xywh,
)
from depthai_nodes.node.parsers.utils.masks_utils import process_single_mask
from depthai_nodes.node.parsers.utils.yolo import (
    YOLOSubtype,
    non_max_suppression,
    parse_kpts,
    parse_yolo_output,
)
from loguru import logger

from .base_parser import BaseParser


class YOLOUnifiedInstanceParser(BaseParser):
    """Parser for YOLO-based models that jointly output boxes, masks, and keypoints."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the unified YOLO parser."""
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
        mask_thres: float = 0.001,
        max_det: int = 300,
        keypoint_label_names: list[str] | None = None,
        keypoint_edges: list[tuple[int, int]] | None = None,
        **kwargs: Any,
    ) -> dai.ImgDetections:
        """Parse backend output into panoptic predictions.

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
            IoU threshold for NMS.
        mask_thres : float, default=0.001
            Mask binarisation threshold.
        max_det : int, default=300
            Maximum detections per image.
        keypoint_label_names : list[str] | None, optional
            Human-readable names for each keypoint.
        keypoint_edges : list[tuple[int, int]] | None, optional
            Skeleton edge pairs connecting keypoints.
        **kwargs : Any
            Additional parser arguments forwarded to the base parser.

        Returns
        -------
        dai.ImgDetections
            Detection results with bounding boxes, scores, class labels,
            instance masks, and keypoints.
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

            yolo_names = sorted(
                [name for name in layer_names if "_yolo" in name or "yolo-" in name]
            )
            outputs_values = [
                raw_output.getTensor(
                    name,
                    dequantize=True,
                    storageOrder=dai.TensorInfo.StorageOrder.NCHW,
                ).astype(np.float32)  # type: ignore
                for name in yolo_names
            ]
            aux_outputs = [
                raw_output.getTensor(
                    name,
                    dequantize=True,
                    storageOrder=dai.TensorInfo.StorageOrder.NCHW,
                ).astype(np.float32)  # type: ignore
                for name in layer_names
                if name not in yolo_names
            ]
        elif isinstance(raw_output, list):
            outputs_values = raw_output[:3]
            aux_outputs = raw_output[3:]
        else:
            raise TypeError(
                f"Unsupported raw_output type: {type(raw_output)}. "
                "Expected dai.NNData or list[np.ndarray]."
            )

        if len(outputs_values) == 0:
            raise ValueError("No YOLO detection heads were found in the output.")

        (
            kpts_outputs,
            masks_outputs,
            protos_output,
        ) = self._partition_export_aux_outputs(outputs_values, aux_outputs)

        strides = (
            [8, 16, 32]
            if subtype not in [YOLOSubtype.V3UT, YOLOSubtype.V3T, YOLOSubtype.V4T]
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

        num_keypoints = kpts_outputs[0].shape[1] // 3
        bbox_heads: list[np.ndarray] = []
        mask_heads: list[np.ndarray] = []
        kpt_heads: list[np.ndarray] = []

        for head_id, (bbox_head, kpt_head, mask_head, stride) in enumerate(
            zip(outputs_values, kpts_outputs, masks_outputs, strides, strict=True)
        ):
            anchors_head = (
                final_anchors[head_id] if final_anchors is not None else None
            )
            bbox_raw = parse_yolo_output(
                bbox_head.copy(),
                stride,
                inferred_n_classes + 5,
                anchors_head,
                head_id=head_id,
                kpts=None,
                det_mode=True,
                subtype=subtype,
            )
            num_locations = bbox_raw.shape[1]

            mask_flat = self._flatten_export_head(
                mask_head, num_locations, "mask coefficients"
            )
            kpt_flat = self._flatten_export_head(
                kpt_head, num_locations, "keypoints"
            )
            kpt_flat[..., 2::3] = sigmoid(kpt_flat[..., 2::3])

            bbox_heads.append(bbox_raw)
            mask_heads.append(mask_flat)
            kpt_heads.append(kpt_flat)

        bbox_output = np.concatenate(bbox_heads, axis=1)
        mask_output = np.concatenate(mask_heads, axis=1)
        keypoints_output = np.concatenate(kpt_heads, axis=1)
        protos_len = mask_output.shape[2]

        # filtering/top-k behavior: run a single NMS
        # over detections augmented with the aligned mask coefficients and kpts.
        preds_combined = np.concatenate(
            [bbox_output, mask_output, keypoints_output],
            axis=2,
        )
        results = non_max_suppression(
            preds_combined,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            num_classes=inferred_n_classes,
            max_det=max_det,
            max_nms=max_det,
        )[0]

        bboxes, labels, label_names, scores, instance_masks, keypoints_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for det_row in results:
            bbox_xyxy = det_row[:4]
            conf = float(det_row[4])
            label = int(det_row[5])
            mask_coeff = det_row[6 : 6 + protos_len]
            kpt_flat = det_row[6 + protos_len :]

            bbox_xywh = xyxy_to_xywh(bbox_xyxy.reshape(1, 4))
            bbox_norm = normalize_bboxes(
                bbox_xywh, height=input_shape[0], width=input_shape[1]
            )[0]

            bboxes.append(bbox_norm)
            scores.append(conf)
            labels.append(label)
            label_names.append(class_map[label])

            mask = process_single_mask(
                protos_output[0], mask_coeff, mask_thres, bbox_norm
            )
            resized_mask = cv2.resize(
                mask,
                (input_shape[1], input_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            instance_masks.append(resized_mask > 0)

            kps = parse_kpts(kpt_flat, num_keypoints, input_shape)  # type: ignore
            keypoints_list.append(kps)

        final_mask = np.asarray(instance_masks)
        if final_mask.size != 0:
            # Flatten (N, H, W) to (N*H, W)
            final_mask = final_mask.reshape(-1, final_mask.shape[-1])

        additional_output = np.array(keypoints_list)
        keypoints = (
            additional_output[:, :, :2]
            if additional_output.size > 0
            else np.array([])
        )
        keypoints_scores = (
            additional_output[:, :, 2]
            if additional_output.size > 0
            else np.array([])
        )

        return create_detection_message(
            bboxes=np.array(bboxes),
            scores=np.array(scores),
            labels=np.array(labels),
            label_names=label_names,
            masks=final_mask,
            keypoints=keypoints,
            keypoints_scores=keypoints_scores,
            keypoint_label_names=keypoint_label_names,
            keypoint_edges=keypoint_edges,
        )

    @staticmethod
    def _partition_export_aux_outputs(
        bbox_outputs: list[np.ndarray],
        aux_outputs: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """Partition outputs into kpts, masks, and protos."""
        num_heads = len(bbox_outputs)
        head_locations = {
            int(np.prod(output.shape[2:4])): idx
            for idx, output in enumerate(bbox_outputs)
        }
        kpts_outputs: list[np.ndarray | None] = [None] * num_heads
        masks_outputs: list[np.ndarray | None] = [None] * num_heads
        remaining: list[np.ndarray] = []

        for output in aux_outputs:
            locations = YOLOUnifiedInstanceParser._aux_locations(output)
            head_idx = head_locations.get(locations)
            if (
                head_idx is not None
                and output.shape[1] % 3 == 0
                and kpts_outputs[head_idx] is None
            ):
                kpts_outputs[head_idx] = output
            else:
                remaining.append(output)

        protos_output: np.ndarray | None = None
        for output in remaining:
            locations = YOLOUnifiedInstanceParser._aux_locations(output)
            head_idx = head_locations.get(locations)
            if head_idx is not None and masks_outputs[head_idx] is None:
                masks_outputs[head_idx] = output
            elif protos_output is None:
                protos_output = output
            else:
                raise ValueError(
                    "Unable to uniquely identify the pruned export outputs "
                    f"from shapes {[arr.shape for arr in aux_outputs]}."
                )

        if protos_output is None or any(x is None for x in kpts_outputs) or any(
            x is None for x in masks_outputs
        ):
            raise ValueError(
                "Incomplete pruned export outputs for unified parsing: "
                f"kpts={[x.shape if x is not None else None for x in kpts_outputs]}, "
                f"masks={[x.shape if x is not None else None for x in masks_outputs]}, "
                f"proto={None if protos_output is None else protos_output.shape}."
            )

        return (
            [x for x in kpts_outputs if x is not None],
            [x for x in masks_outputs if x is not None],
            protos_output,
        )

    @staticmethod
    def _aux_locations(output: np.ndarray) -> int:
        """Return the flattened spatial/location count of an aux head."""
        if output.ndim == 4:
            return int(np.prod(output.shape[2:4]))
        if output.ndim == 3:
            return int(output.shape[2])
        raise ValueError(f"Unexpected auxiliary output shape {output.shape}.")

    @staticmethod
    def _flatten_export_head(
        head_output: np.ndarray,
        num_locations: int,
        head_name: str,
    ) -> np.ndarray:
        """Flatten to (B, N, C)."""
        if head_output.ndim == 4:
            batch_size, channels, _, _ = head_output.shape
            flattened = head_output.reshape(batch_size, channels, -1)
        elif head_output.ndim == 3:
            flattened = head_output
        else:
            raise ValueError(
                f"Unexpected {head_name} output shape {head_output.shape}."
            )

        flattened = flattened.transpose(0, 2, 1).astype(np.float32, copy=False)
        if flattened.shape[1] != num_locations:
            raise ValueError(
                f"{head_name.capitalize()} output shape {head_output.shape} "
                f"does not align with {num_locations} decoded detections."
            )
        return flattened
