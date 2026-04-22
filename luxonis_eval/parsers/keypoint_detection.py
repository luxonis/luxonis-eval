from typing import Any

import depthai as dai
import numpy as np
import torch
from depthai_nodes.message.creators import create_detection_message
from depthai_nodes.node.parsers.utils import normalize_bboxes, xyxy_to_xywh
from depthai_nodes.node.parsers.utils.yolo import (
    YOLOSubtype,
    decode_yolo_output,
    parse_kpts,
)
from loguru import logger
from torchvision.ops import batched_nms

from .base_parser import BaseParser


class YOLOKeypointDetectionParser(BaseParser):
    """Parser for YOLO-based keypoint detection model outputs.

    The default ``depthai`` path uses the generic DepthAI YOLO decode. The
    ``luxonis_train`` path is intended for ONNX exports coming from
    Luxonis Train-style keypoint+bbox heads where ``self.export``
    changes the output tensors and the parser must reproduce the train-side
    postprocess before metrics are computed.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the YOLO keypoint detection parser."""
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
        postprocessing: str = "depthai",
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
        postprocessing : str, default="depthai"
            Postprocessing implementation. Use ``"luxonis_train"`` to
            match export-mode decoding and NMS from Luxonis Train custom
            keypoint heads.
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
        elif isinstance(raw_output, list):
            output_names = kwargs.get("output_names")
            if output_names is not None and len(output_names) == len(raw_output):
                named_outputs = {
                    str(name): value
                    for name, value in zip(
                        output_names, raw_output, strict=True
                    )
                }
                outputs_names = sorted(
                    [
                        name
                        for name in named_outputs
                        if "_yolo" in name or "yolo-" in name
                    ]
                )
                outputs_values = [
                    named_outputs[name].astype(np.float32, copy=False)
                    for name in outputs_names
                ]
                kpts_output_names = sorted(
                    [name for name in named_outputs if "kpt_output" in name]
                )
                kpts_outputs = [
                    named_outputs[name].astype(np.float32, copy=False)
                    for name in kpts_output_names
                ]
            else:
                outputs_names = [f"output_{i}" for i in range(len(raw_output))]
                outputs_values = raw_output[:3]
                kpts_outputs = raw_output[3:]
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

        if postprocessing == "luxonis_train":
            return self._parse_luxonis_train_export(
                outputs_values=outputs_values,
                kpts_outputs=kpts_outputs,
                input_shape=input_shape,
                class_map=class_map,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
                num_classes=inferred_n_classes,
                num_keypoints=num_keypoints,
                keypoint_label_names=keypoint_label_names,
                keypoint_edges=keypoint_edges,
            )
        if postprocessing != "depthai":
            raise ValueError(
                f"Unsupported postprocessing '{postprocessing}'. "
                "Supported values are 'depthai' and 'luxonis_train'."
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

    def _parse_luxonis_train_export(
        self,
        *,
        outputs_values: list[np.ndarray],
        kpts_outputs: list[np.ndarray],
        input_shape: tuple[int, int],
        class_map: dict[int, str],
        conf_thres: float,
        iou_thres: float,
        max_det: int,
        num_classes: int,
        num_keypoints: int,
        keypoint_label_names: list[str] | None,
        keypoint_edges: list[tuple[int, int]] | None,
    ) -> dai.ImgDetections:
        """Decode export-mode outputs with luxonis-train-style NMS."""
        strides = self._infer_strides(outputs_values)
        preds = self._prepare_luxonis_train_predictions(
            outputs_values=outputs_values,
            kpts_outputs=kpts_outputs,
            strides=strides,
            num_classes=num_classes,
        )
        nms_results = self._luxonis_train_nms(
            preds=preds,
            num_classes=num_classes,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
        )
        results = (
            nms_results[0].detach().cpu().numpy()
            if nms_results
            else np.zeros((0, 6 + num_keypoints * 3), dtype=np.float32)
        )
        return self._create_keypoint_message(
            results=results.astype(np.float32, copy=False),
            input_shape=input_shape,
            class_map=class_map,
            num_keypoints=num_keypoints,
            keypoint_label_names=keypoint_label_names,
            keypoint_edges=keypoint_edges,
        )

    @staticmethod
    def _infer_strides(outputs_values: list[np.ndarray]) -> list[int]:
        if len(outputs_values) == 3:
            return [8, 16, 32]
        return [2 ** (3 + i) for i in range(len(outputs_values))]

    def _prepare_luxonis_train_predictions(
        self,
        *,
        outputs_values: list[np.ndarray],
        kpts_outputs: list[np.ndarray],
        strides: list[int],
        num_classes: int,
    ) -> torch.Tensor:
        """Build prediction tensor matching luxonis-train NMS input."""
        pred_bboxes: list[torch.Tensor] = []
        pred_classes: list[torch.Tensor] = []
        pred_keypoints: list[torch.Tensor] = []

        for bbox_out, kpt_out, stride in zip(
            outputs_values, kpts_outputs, strides, strict=True
        ):
            bbox_tensor = torch.from_numpy(
                np.ascontiguousarray(bbox_out)
            ).float()
            kpt_tensor = torch.from_numpy(
                np.ascontiguousarray(kpt_out)
            ).float()

            bs, _, h, w = bbox_tensor.shape
            anchor_points = self._make_anchor_points(
                height=h,
                width=w,
                device=bbox_tensor.device,
                dtype=bbox_tensor.dtype,
            )

            bbox_dist = bbox_tensor[:, :4].reshape(bs, 4, -1)
            pred_bboxes.append(
                self._dist2bbox(
                    distances=bbox_dist,
                    anchor_points=anchor_points,
                )
                * float(stride)
            )
            pred_classes.append(
                bbox_tensor[:, 5 : 5 + num_classes]
                .reshape(bs, num_classes, -1)
                .permute(0, 2, 1)
            )

            kpt_tensor[:, 2::3, :] = torch.sigmoid(kpt_tensor[:, 2::3, :])
            pred_keypoints.append(kpt_tensor.permute(0, 2, 1))

        boxes = torch.cat(pred_bboxes, dim=1)
        class_probs = torch.cat(pred_classes, dim=1)
        keypoints = torch.cat(pred_keypoints, dim=1)
        objectness = torch.ones(
            (*boxes.shape[:2], 1), dtype=boxes.dtype, device=boxes.device
        )
        return torch.cat((boxes, objectness, class_probs, keypoints), dim=-1)

    @staticmethod
    def _make_anchor_points(
        *,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        shift_x = torch.arange(width, device=device, dtype=dtype) + 0.5
        shift_y = torch.arange(height, device=device, dtype=dtype) + 0.5
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        return torch.stack((shift_x, shift_y), dim=-1).reshape(-1, 2)

    @staticmethod
    def _dist2bbox(
        *,
        distances: torch.Tensor,
        anchor_points: torch.Tensor,
    ) -> torch.Tensor:
        left = anchor_points[:, 0].view(1, -1) - distances[:, 0]
        top = anchor_points[:, 1].view(1, -1) - distances[:, 1]
        right = anchor_points[:, 0].view(1, -1) + distances[:, 2]
        bottom = anchor_points[:, 1].view(1, -1) + distances[:, 3]
        return torch.stack((left, top, right, bottom), dim=-1)

    @staticmethod
    def _luxonis_train_nms(
        *,
        preds: torch.Tensor,
        num_classes: int,
        conf_thres: float,
        iou_thres: float,
        max_det: int,
    ) -> list[torch.Tensor]:
        """Torch port of luxonis-train NMS for predicts_objectness=False."""
        candidate_mask = torch.logical_and(
            preds[..., 4] > conf_thres,
            torch.max(preds[..., 5 : 5 + num_classes], dim=-1).values
            > conf_thres,
        )
        output = [
            torch.zeros((0, preds.size(-1)), device=preds.device)
        ] * preds.size(0)
        has_additional = preds.size(-1) > (5 + num_classes)

        for i, x in enumerate(preds):
            curr = x[candidate_mask[i]]
            if curr.size(0) == 0:
                continue

            curr[:, 5 : 5 + num_classes] *= curr[:, 4:5]
            conf, class_idx = curr[:, 5 : 5 + num_classes].max(
                1, keepdim=True
            )
            keep_mask = conf.view(-1) > conf_thres
            curr_out = torch.cat((curr[:, :4], conf, class_idx.float()), dim=1)[
                keep_mask
            ]

            if has_additional:
                curr_out = torch.hstack(
                    [curr_out, curr[keep_mask, 5 + num_classes :]]
                )

            if curr_out.size(0) == 0:
                continue

            keep_indices = batched_nms(
                boxes=curr_out[:, :4],
                scores=curr_out[:, 4],
                idxs=curr_out[:, 5].int(),
                iou_threshold=iou_thres,
            )[:max_det]
            output[i] = curr_out[keep_indices]

        return output

    @staticmethod
    def _create_keypoint_message(
        *,
        results: np.ndarray,
        input_shape: tuple[int, int],
        class_map: dict[int, str],
        num_keypoints: int,
        keypoint_label_names: list[str] | None,
        keypoint_edges: list[tuple[int, int]] | None,
    ) -> dai.ImgDetections:
        """Convert decoded results into a DepthAI detection message."""
        if results.size == 0:
            return create_detection_message(
                bboxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                labels=np.zeros((0,), dtype=np.int32),
                label_names=[],
                keypoints=np.zeros((0, num_keypoints, 2), dtype=np.float32),
                keypoints_scores=np.zeros(
                    (0, num_keypoints), dtype=np.float32
                ),
                keypoint_label_names=keypoint_label_names,
                keypoint_edges=keypoint_edges,
            )

        height, width = input_shape
        bboxes_xywh = xyxy_to_xywh(results[:, :4])
        bboxes = normalize_bboxes(
            bboxes_xywh, height=height, width=width
        ).astype(np.float32)
        scores = results[:, 4].astype(np.float32)
        labels = results[:, 5].astype(np.int32)
        label_names = [class_map[int(label)] for label in labels]

        keypoints = results[:, 6:].reshape(-1, num_keypoints, 3).copy()
        # ``luxonis_train`` postprocess returns pixel-space keypoints after
        # NMS. Keep that representation here to match checkpoint-side eval.

        return create_detection_message(
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            label_names=label_names,
            keypoints=keypoints[:, :, :2].astype(np.float32),
            keypoints_scores=keypoints[:, :, 2].astype(np.float32),
            keypoint_label_names=keypoint_label_names,
            keypoint_edges=keypoint_edges,
        )
