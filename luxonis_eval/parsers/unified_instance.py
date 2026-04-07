from typing import Any

import cv2
import depthai as dai
import numpy as np
from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.utils import normalize_bboxes, xyxy_to_xywh
from depthai_nodes.node.parsers.utils.masks_utils import (
    get_segmentation_outputs,
    process_single_mask,
)
from depthai_nodes.node.parsers.utils.yolo import (
    YOLOSubtype,
    decode_yolo_output,
    parse_kpts,
)
from loguru import logger

from .base_parser import BaseParser


class YOLOUnifiedInstanceParser(BaseParser):
    """Parser for YOLO-based panoptic models that jointly output object detections, instance segmentation masks, and keypoints."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the YOLO panoptic parser."""
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
        iou_match_thres: float = 0.5,
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
        iou_match_thres : float, default=0.5
            Minimum IoU required to match a keypoint detection to a
            segmentation detection during the merge step.
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
                [n for n in layer_names if "_yolo" in n or "yolo-" in n]
            )
            outputs_values = [
                raw_output.getTensor(
                    o,
                    dequantize=True,
                    storageOrder=dai.TensorInfo.StorageOrder.NCHW,
                ).astype(np.float32)  # type: ignore
                for o in yolo_names
            ]
            kpts_names = sorted([n for n in layer_names if "kpt_output" in n])
            kpts_outputs = [
                raw_output.getTensor(o, dequantize=True).astype(np.float32)  # type: ignore
                for o in kpts_names
            ]
            (
                masks_outputs,
                protos_output,
                protos_len,
            ) = get_segmentation_outputs(raw_output)

        elif isinstance(raw_output, list):
            # Expected order:
            #   [0:3]  → output*_yolov8   (main detection heads)
            #   [3:6]  → kpt_output*      (keypoint heads)
            #   [6:9]  → output*_masks    (mask coefficient heads)
            #   [9]    → protos_output
            outputs_values = raw_output[:3]
            kpts_outputs = raw_output[3:6]
            masks_outputs = raw_output[6:9]
            protos_output = raw_output[9]
            protos_len = protos_output.shape[1]
        else:
            raise TypeError(
                f"Unsupported raw_output type: {type(raw_output)}. "
                "Expected dai.NNData or list[np.ndarray]."
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
        num_keypoints = kpts_outputs[0].shape[1] // 3

        decode_common = {
            "strides": strides,
            "anchors": final_anchors,
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
            "num_classes": inferred_n_classes,
            "subtype": subtype,
            "max_nms": max_det,
        }

        # Pass A: keypoint mode
        kpt_results = decode_yolo_output(
            yolo_outputs=[o.copy() for o in outputs_values],
            kpts=[k.copy() for k in kpts_outputs],
            det_mode=False,
            **decode_common,
        )

        # Pass B: segmentation mode
        seg_results = decode_yolo_output(
            yolo_outputs=[o.copy() for o in outputs_values],
            kpts=None,
            det_mode=False,
            **decode_common,
        )

        seg_boxes = (
            seg_results[:, :4] if seg_results.shape[0] else np.zeros((0, 4))
        )
        kpt_boxes = (
            kpt_results[:, :4] if kpt_results.shape[0] else np.zeros((0, 4))
        )

        # For each seg detection, find the best-matching kpt detection
        kpt_match_idx: list[int | None] = [None] * seg_results.shape[0]
        if seg_results.shape[0] > 0 and kpt_results.shape[0] > 0:
            iou_mat = self._iou_matrix(seg_boxes, kpt_boxes)  # (N_seg, N_kpt)
            best_kpt = iou_mat.argmax(axis=1)
            best_iou = iou_mat[np.arange(len(best_kpt)), best_kpt]
            for seg_i, (ki, iou_val) in enumerate(
                zip(best_kpt, best_iou, strict=True)
            ):
                if iou_val >= iou_match_thres:
                    kpt_match_idx[seg_i] = int(ki)

        bboxes, labels, label_names, scores, instance_masks, keypoints_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for seg_i in range(seg_results.shape[0]):
            seg_row = seg_results[seg_i]
            bbox_xyxy = seg_row[:4]
            conf = float(seg_row[4])
            label = int(seg_row[5])
            seg_other = seg_row[6:]  # hi, ai, xi, yi

            bbox_xywh = xyxy_to_xywh(bbox_xyxy.reshape(1, 4))
            bbox_norm = normalize_bboxes(
                bbox_xywh, height=input_shape[0], width=input_shape[1]
            )[0]

            bboxes.append(bbox_norm)
            scores.append(conf)
            labels.append(label)
            label_names.append(class_map[label])

            # --- Instance mask ---
            seg_coeff = seg_other.astype(int)
            hi, ai, xi, yi = seg_coeff
            mask_coeff = masks_outputs[hi][
                0, ai * protos_len : (ai + 1) * protos_len, yi, xi
            ]
            mask = process_single_mask(
                protos_output[0], mask_coeff, mask_thres, bbox_norm
            )
            resized_mask = cv2.resize(
                mask,
                (input_shape[1], input_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            instance_masks.append(resized_mask > 0)

            # --- Keypoints (from matched kpt detection, or zeros) ---
            matched_ki = kpt_match_idx[seg_i]
            if matched_ki is not None:
                kpt_flat = kpt_results[matched_ki, 6:]  # *kpt_flat
                kps = parse_kpts(kpt_flat, num_keypoints, input_shape)  # type: ignore
            else:
                logger.warning(
                    f"Detection {seg_i} has no keypoint match "
                    f"(iou_match_thres={iou_match_thres}); using zero keypoints."
                )
                kps = [(0.0, 0.0, 0.0)] * num_keypoints
            keypoints_list.append(kps)

        final_mask = np.asarray(instance_masks)
        if final_mask.size != 0:
            # Flatten (N, H, W) to (N*H, W) since dai.ImgDetections expects a 2D mask.
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
    def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute pairwise IoU between two sets of xyxy boxes.

        Parameters
        ----------
        boxes_a : np.ndarray
            Shape (N, 4) in xyxy format.
        boxes_b : np.ndarray
            Shape (M, 4) in xyxy format.

        Returns
        -------
        np.ndarray
            IoU matrix of shape (N, M).
        """
        x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
        y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
        x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
        y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (
            boxes_a[:, 3] - boxes_a[:, 1]
        )
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (
            boxes_b[:, 3] - boxes_b[:, 1]
        )
        union = area_a[:, None] + area_b[None, :] - inter

        return np.where(union > 0, inter / union, 0.0)
