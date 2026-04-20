from typing import Any

import depthai as dai
import numpy as np
from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.utils import normalize_bboxes, xyxy_to_xywh
from depthai_nodes.node.parsers.utils.yolo import (
    YOLOSubtype,
    decode_yolo_output,
    parse_kpts,
)
from loguru import logger

from .base_parser import BaseParser


class YOLOKeypointDetectionParser(BaseParser):
    """Parser for YOLO-based keypoint detection model outputs."""

    def __init__(
        self,
        debug_raw_outputs_samples: int = 0,
        debug_raw_values: int = 8,
        debug_decoded_detections: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize the YOLO keypoint detection parser."""
        self._debug_raw_outputs_samples = debug_raw_outputs_samples
        self._debug_raw_values = debug_raw_values
        self._debug_decoded_detections = debug_decoded_detections
        self._debug_raw_logged = 0
        logger.warning(
            "YOLOKeypointDetectionParser init: debug_raw_outputs_samples={}, debug_raw_values={}, debug_decoded_detections={}",
            self._debug_raw_outputs_samples,
            self._debug_raw_values,
            self._debug_decoded_detections,
        )
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
        keypoint_label_names: list[str] | None = None,
        keypoint_edges: list[tuple[int, int]] | None = None,
        **kwargs: Any,
    ) -> dai.ImgDetections:
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
        keypoint_label_names : list[str] | None, optional
            Names of keypoint labels.
        keypoint_edges : list[tuple[int, int]] | None, optional
            Edges connecting keypoints.
        **kwargs : Any
            Additional parser arguments.

        Returns
        -------
        dai.ImgDetections
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
            native_outputs = None
            native_kpts_outputs = None
            if self._debug_raw_logged < self._debug_raw_outputs_samples:
                native_outputs = [
                    raw_output.getTensor(o, dequantize=True).astype(np.float32)  # type: ignore
                    for o in outputs_names
                ]
                native_kpts_outputs = [
                    raw_output.getTensor(o, dequantize=True).astype(np.float32)  # type: ignore
                    for o in kpts_output_names
                ]
        elif isinstance(raw_output, list):
            outputs_names = [f"output_{i}" for i in range(len(raw_output))]
            outputs_values = raw_output[:3]
            kpts_outputs = raw_output[3:]
            native_outputs = None
            native_kpts_outputs = None
        else:
            raise TypeError(
                f"Unsupported raw_output type: {type(raw_output)}. Expected dai.NNData or list[np.ndarray]."
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

        if self._debug_raw_logged < self._debug_raw_outputs_samples:
            self._log_raw_outputs(
                source_type=type(raw_output).__name__,
                outputs_names=outputs_names,
                outputs_values=outputs_values,
                kpts_outputs=kpts_outputs,
                native_outputs=native_outputs,
                native_kpts_outputs=native_kpts_outputs,
                input_shape=input_shape,
                inferred_n_classes=inferred_n_classes,
                num_keypoints=num_keypoints,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
            )

        results = decode_yolo_output(
            yolo_outputs=outputs_values,
            strides=strides,
            anchors=final_anchors,
            kpts=kpts_outputs,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            num_classes=inferred_n_classes,
            det_mode=False,
            subtype=subtype,
            max_nms=max_det,
        )

        if self._debug_raw_logged < self._debug_raw_outputs_samples:
            self._log_decoded_results(results)
            self._debug_raw_logged += 1

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
            bbox = xyxy_to_xywh(bbox.reshape(1, 4))
            bbox = normalize_bboxes(
                bbox, height=input_shape[0], width=input_shape[1]
            )[0]
            bboxes.append(bbox)
            scores.append(float(conf))
            labels.append(int(label))
            label_names.append(class_map[int(label)])

            kpts = parse_kpts(other, num_keypoints, input_shape)  # type: ignore
            additional_output.append(kpts)

        additional_output = np.array(additional_output)
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

        # logger.warning(f"Num Keypoints: {num_keypoints}, Results: {results.shape}, Keypoints: {keypoints.shape}, Scores: {keypoints_scores.shape}")

        return create_detection_message(
            bboxes=np.array(bboxes),
            scores=np.array(scores),
            labels=np.array(labels),
            label_names=label_names,
            keypoints=keypoints,
            keypoints_scores=keypoints_scores,
            keypoint_label_names=keypoint_label_names,
            keypoint_edges=keypoint_edges,
        )

    def _summarize_array(self, arr: np.ndarray) -> dict[str, Any]:
        flat = arr.reshape(-1)
        head = flat[: self._debug_raw_values]
        return {
            "shape": tuple(arr.shape),
            "min": float(arr.min()) if arr.size else None,
            "max": float(arr.max()) if arr.size else None,
            "mean": float(arr.mean()) if arr.size else None,
            "head": np.round(head, 5).tolist(),
        }

    def _compare_tensor_views(
        self, parsed: np.ndarray, native: np.ndarray
    ) -> dict[str, Any]:
        comparisons: dict[str, Any] = {
            "parsed_shape": tuple(parsed.shape),
            "native_shape": tuple(native.shape),
        }

        if parsed.shape == native.shape:
            comparisons["identity_max_abs_diff"] = float(
                np.max(np.abs(parsed - native))
            )

        if native.ndim == 4:
            nhwc_to_nchw = np.transpose(native, (0, 3, 1, 2))
            if parsed.shape == nhwc_to_nchw.shape:
                comparisons["nhwc_to_nchw_max_abs_diff"] = float(
                    np.max(np.abs(parsed - nhwc_to_nchw))
                )

            nchw_to_nhwc = np.transpose(native, (0, 2, 3, 1))
            if parsed.shape == nchw_to_nhwc.shape:
                comparisons["nchw_to_nhwc_max_abs_diff"] = float(
                    np.max(np.abs(parsed - nchw_to_nhwc))
                )

        return comparisons

    def _log_raw_outputs(
        self,
        *,
        source_type: str,
        outputs_names: list[str],
        outputs_values: list[np.ndarray],
        kpts_outputs: list[np.ndarray],
        native_outputs: list[np.ndarray] | None,
        native_kpts_outputs: list[np.ndarray] | None,
        input_shape: tuple[int, int],
        inferred_n_classes: int,
        num_keypoints: int,
        conf_thres: float,
        iou_thres: float,
    ) -> None:
        sample_idx = self._debug_raw_logged + 1
        logger.warning(
            "Parser raw debug sample {}: source={}, input_shape={}, inferred_n_classes={}, num_keypoints={}, conf_thres={}, iou_thres={}",
            sample_idx,
            source_type,
            input_shape,
            inferred_n_classes,
            num_keypoints,
            conf_thres,
            iou_thres,
        )

        for name, arr in zip(outputs_names, outputs_values, strict=True):
            logger.info(
                "Parsed det tensor {}: {}",
                name,
                self._summarize_array(arr),
            )
        for i, arr in enumerate(kpts_outputs, start=1):
            logger.info(
                "Parsed kpt tensor kpt_output{}: {}",
                i,
                self._summarize_array(arr),
            )

        if native_outputs is not None:
            for name, parsed_arr, native_arr in zip(
                outputs_names, outputs_values, native_outputs, strict=True
            ):
                logger.info(
                    "Native det tensor {}: {}",
                    name,
                    self._summarize_array(native_arr),
                )
                logger.info(
                    "Det tensor {} layout comparison: {}",
                    name,
                    self._compare_tensor_views(parsed_arr, native_arr),
                )
        if native_kpts_outputs is not None:
            for i, (parsed_arr, native_arr) in enumerate(
                zip(kpts_outputs, native_kpts_outputs, strict=True), start=1
            ):
                logger.info(
                    "Native kpt tensor kpt_output{}: {}",
                    i,
                    self._summarize_array(native_arr),
                )
                logger.info(
                    "Kpt tensor kpt_output{} layout comparison: {}",
                    i,
                    self._compare_tensor_views(parsed_arr, native_arr),
                )

    def _log_decoded_results(self, results: np.ndarray) -> None:
        logger.info(
            "Decoded results summary: shape={}, first_rows={}",
            tuple(results.shape),
            np.round(results[: self._debug_decoded_detections], 5).tolist(),
        )
