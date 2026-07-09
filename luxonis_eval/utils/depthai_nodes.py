from __future__ import annotations

from typing import Any

import numpy as np
from depthai_nodes.node.parsers.utils.yolo import YOLOSubtype

from luxonis_eval.engines.base_engine import ModelSpec
from luxonis_eval.engines.io import EngineOutput


def ordered_class_names(class_map: dict[int, str]) -> list[str]:
    """Return class names ordered by class index."""
    if not class_map:
        return []

    ordered_indices = sorted(class_map)
    expected_indices = list(range(len(ordered_indices)))
    if ordered_indices != expected_indices:
        raise ValueError(
            "class_map must contain contiguous zero-based indices, got "
            f"{ordered_indices}."
        )

    return [class_map[index] for index in ordered_indices]


def extract_segmentation_mask(predictions: Any) -> np.ndarray:
    """Extract a semantic-segmentation mask from DepthAI-style messages."""
    if hasattr(predictions, "getCvMask"):
        mask = predictions.getCvMask()
    elif hasattr(predictions, "getCvSegmentationMask"):
        mask = predictions.getCvSegmentationMask()
    elif hasattr(predictions, "mask"):
        mask = predictions.mask
    else:
        raise TypeError(
            "Unsupported segmentation prediction type "
            f"{type(predictions)!r}: expected a DepthAI SegmentationMask "
            "message or a compatible wrapper."
        )

    if mask is None:
        raise ValueError("Segmentation prediction does not contain a mask.")

    return np.asarray(mask)


def build_yolo_compute_kwargs(
    output: EngineOutput,
    model_spec: ModelSpec,
    *,
    class_map: dict[int, str],
    subtype: str,
    n_classes: int | None = None,
    anchors: list[list[list[float]]] | None = None,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    mask_conf: float = 0.5,
    keypoint_label_names: list[str] | None = None,
    keypoint_edges: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Adapter that converts EngineOutput + ModelSpec into the field mapping
    required to construct ``depthai_nodes`` ``YOLOComputeInputs``."""
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
                (
                    name
                    for name in layer_names
                    if "protos" in name
                ),
                "protos_output",
            )
            v26_mask_coeffs = output.get(mask_name).astype(
                np.float32, copy=False
            )
            v26_protos = output.get(
                protos_name, layout="NCHW"
            ).astype(np.float32, copy=False)[0]
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
            [name for name in layer_names if "_yolo" in name or "yolo-" in name]
        ) or list(layer_names)
        outputs_values = [
            output.get(name, layout="NCHW").astype(np.float32, copy=False)
            for name in outputs_names
        ]

        if (
            any("kpt_output" in name for name in layer_names)
            and subtype_enum != YOLOSubtype.P
        ):
            kpts_output_names = sorted(
                [name for name in layer_names if "kpt_output" in name]
            ) or layer_names[len(outputs_names) :]
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
            mask_output_names = sorted(
                [
                    name
                    for name in layer_names
                    if "mask" in name and "proto" not in name
                ]
            ) or layer_names[len(outputs_names) : -1]
            masks_outputs_values = [
                output.get(name, layout="NCHW").astype(np.float32, copy=False)
                for name in mask_output_names
            ]
            protos_output = output.get(protos_name, layout="NCHW").astype(
                np.float32, copy=False
            )
            protos_len = protos_output.shape[1]

        strides = (
            [8, 16, 32]
            if subtype_enum
            not in [YOLOSubtype.V3UT, YOLOSubtype.V3T, YOLOSubtype.V4T]
            else [16, 32]
        )
        final_anchors: np.ndarray | None = (
            np.array(anchors).reshape(len(strides), -1) if anchors else None
        )
        inferred_n_classes = (
            outputs_values[0].shape[1] - 5
            if final_anchors is None
            else (outputs_values[0].shape[1] // final_anchors.shape[0]) - 5
        )
        if n_classes is not None and inferred_n_classes != n_classes:
            raise ValueError(
                f"The provided number of classes {n_classes} does not match the "
                f"model's {inferred_n_classes}."
            )
        resolved_n_classes = inferred_n_classes

    return {
        "subtype": subtype_enum,
        "layer_names": layer_names,
        "outputs_values": outputs_values,
        "conf_threshold": conf_threshold,
        "n_classes": resolved_n_classes,
        "iou_threshold": iou_threshold,
        "max_det": max_det,
        "anchors": anchors,
        "n_keypoints": kpts_outputs[0].shape[1] // 3 if kpts_outputs else 17,
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
