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
from depthai_nodes.node.parsers.utils.yolo import YOLOSubtype

from luxonis_eval.engines.base_engine import ModelSpec
from luxonis_eval.engines.io import EngineOutput
from luxonis_eval.utils.utils import ordered_class_names

from .base_parser import BaseParser


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
        strides: list[int] | tuple[int, ...] | None = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        n_keypoints: int | None = None,
        mask_conf: float = 0.5,
        max_det: int = 300,
        keypoint_label_names: list[str] | None = None,
        keypoint_edges: list[tuple[int, int]] | None = None,
        **kwargs: Any,
    ) -> dai.ImgDetections:
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
                    n_keypoints=n_keypoints,
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


def build_yolo_compute_kwargs(
    output: EngineOutput,
    model_spec: ModelSpec,
    *,
    class_map: dict[int, str],
    subtype: str,
    n_classes: int | None = None,
    anchors: list[list[list[float]]] | None = None,
    strides: list[int] | tuple[int, ...] | None = None,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    n_keypoints: int | None = None,
    mask_conf: float = 0.5,
    keypoint_label_names: list[str] | None = None,
    keypoint_edges: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Build the payload required by ``depthai_nodes`` YOLO compute."""
    try:
        subtype_enum = YOLOSubtype(subtype.lower())
    except ValueError as err:
        raise ValueError(
            f"Invalid YOLO subtype {subtype}. Supported YOLO subtypes are "
            f"{[e.value for e in YOLOSubtype][:-1]}."
        ) from err

    layer_names = list(output.names())
    outputs_values: list[np.ndarray]
    kpts_outputs: list[np.ndarray] | None = None
    masks_outputs_values: list[np.ndarray] | None = None
    protos_output: np.ndarray | None = None
    protos_len: int | None = None
    v26_mask_coeffs: np.ndarray | None = None
    v26_protos: np.ndarray | None = None
    v26_pose_kpts: np.ndarray | None = None

    if subtype_enum == YOLOSubtype.V26:
        if any("output_masks" in name for name in layer_names):
            outputs_values = [
                output.get("output_yolo26").astype(np.float32, copy=False)
            ]
            mask_name = next(
                name for name in layer_names if "output_masks" in name
            )
            protos_name = next(
                (name for name in layer_names if "protos" in name),
                "protos_output",
            )
            v26_mask_coeffs = output.get(mask_name).astype(
                np.float32, copy=False
            )
            v26_protos = output.get(protos_name, layout="NCHW").astype(
                np.float32, copy=False
            )[0]
        elif any("kpt_output" in name for name in layer_names):
            outputs_values = [
                output.get("output_yolo26").astype(np.float32, copy=False)
            ]
            kpt_name = next(
                name for name in layer_names if "kpt_output" in name
            )
            v26_pose_kpts = output.get(kpt_name).astype(
                np.float32, copy=False
            )
        else:
            outputs_values = [
                output.get(name).astype(np.float32, copy=False)
                for name in layer_names
            ]
        resolved_n_classes = n_classes or len(class_map)
    else:
        outputs_names = sorted(
            [
                name
                for name in layer_names
                if "_yolo" in name or "yolo-" in name
            ]
        ) or list(layer_names)
        outputs_values = [
            output.get(name, layout="NCHW").astype(np.float32, copy=False)
            for name in outputs_names
        ]

        if (
            any("kpt_output" in name for name in layer_names)
            and subtype_enum != YOLOSubtype.P
        ):
            kpts_output_names = (
                sorted([name for name in layer_names if "kpt_output" in name])
                or layer_names[len(outputs_names) :]
            )
            kpts_outputs = [
                output.get(name).astype(np.float32, copy=False)
                for name in kpts_output_names
            ]
        elif (
            any("mask" in name for name in layer_names)
            and subtype_enum != YOLOSubtype.P
        ):
            protos_name = next(
                (name for name in layer_names if "protos" in name),
                layer_names[-1],
            )
            mask_output_names = (
                sorted(
                    [
                        name
                        for name in layer_names
                        if "mask" in name and "proto" not in name
                    ]
                )
                or layer_names[len(outputs_names) : -1]
            )
            masks_outputs_values = [
                output.get(name, layout="NCHW").astype(np.float32, copy=False)
                for name in mask_output_names
            ]
            protos_output = output.get(protos_name, layout="NCHW").astype(
                np.float32, copy=False
            )
            protos_len = protos_output.shape[1]

        final_anchors: np.ndarray | None = (
            np.asarray(anchors, dtype=np.float32) if anchors else None
        )
        inferred_n_classes = (
            outputs_values[0].shape[1] - 5
            if final_anchors is None
            else (outputs_values[0].shape[1] // final_anchors.shape[1]) - 5
        )
        if n_classes is not None and inferred_n_classes != n_classes:
            raise ValueError(
                f"The provided number of classes {n_classes} does not match the "
                f"model's {inferred_n_classes}."
            )
        resolved_n_classes = inferred_n_classes

    inferred_n_keypoints = (
        kpts_outputs[0].shape[1] // 3 if kpts_outputs is not None else None
    )
    if (
        n_keypoints is not None
        and inferred_n_keypoints is not None
        and inferred_n_keypoints != n_keypoints
    ):
        raise ValueError(
            f"The provided number of keypoints {n_keypoints} does not match "
            f"the model's {inferred_n_keypoints}."
        )
    resolved_n_keypoints = inferred_n_keypoints or n_keypoints or 17

    return {
        "subtype": subtype_enum,
        "layer_names": layer_names,
        "outputs_values": outputs_values,
        "strides": list(strides) if strides is not None else None,
        "conf_threshold": conf_threshold,
        "n_classes": resolved_n_classes,
        "iou_threshold": iou_threshold,
        "max_det": max_det,
        "anchors": anchors,
        "n_keypoints": resolved_n_keypoints,
        "label_names": ordered_class_names(class_map),
        "keypoint_label_names": keypoint_label_names,
        "keypoint_edges": keypoint_edges,
        "input_shape": (model_spec.height, model_spec.width),
        "kpts_outputs": kpts_outputs,
        "masks_outputs_values": masks_outputs_values,
        "protos_output": protos_output,
        "protos_len": protos_len,
        "mask_conf": mask_conf,
        "v26_mask_coeffs": v26_mask_coeffs,
        "v26_protos": v26_protos,
        "v26_pose_kpts": v26_pose_kpts,
    }
