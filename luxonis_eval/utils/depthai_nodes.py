from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from depthai_nodes.node.parsers.yolo import YOLOComputeInputs
from depthai_nodes.node.parsers.utils.yolo import (
    YOLOSubtype,
    decode_yolo26,
    decode_yolo_output,
    resolve_yolo_strides,
)

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
    """Extract a semantic-segmentation mask from a DepthAI message."""
    if hasattr(predictions, "getCvMask"):
        mask = predictions.getCvMask()
    elif hasattr(predictions, "getCvSegmentationMask"):
        mask = predictions.getCvSegmentationMask()
    else:
        raise TypeError(
            "Unsupported segmentation prediction type "
            f"{type(predictions)!r}: expected a DepthAI SegmentationMask "
            "message."
        )

    if mask is None:
        raise ValueError("Segmentation prediction does not contain a mask.")

    return np.asarray(mask)


def build_yolo_compute_inputs(
    output: EngineOutput,
    model_spec: ModelSpec,
    *,
    class_map: dict[int, str],
    subtype: str,
    n_classes: int | None = None,
    anchors: list[list[list[float]]] | None = None,
    strides: list[int] | None = None,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    n_keypoints: int | None = None,
    mask_conf: float = 0.5,
    keypoint_label_names: list[str] | None = None,
    keypoint_edges: list[tuple[int, int]] | None = None,
) -> YOLOComputeInputs:
    """Adapter that converts EngineOutput + ModelSpec into the field
    mapping required to construct ``depthai_nodes``
    ``YOLOComputeInputs``."""
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
            v26_pose_kpts = output.get(kpt_name).astype(np.float32, copy=False)
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

        resolved_strides = resolve_yolo_strides(
            strides,
            subtype_enum,
            num_outputs=len(outputs_values),
        )
        final_anchors: np.ndarray | None = (
            np.asarray(anchors, dtype=np.float32).reshape(
                len(resolved_strides), -1
            )
            if anchors
            else None
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

    return YOLOComputeInputs(
        subtype=subtype_enum,
        layer_names=layer_names,
        outputs_values=outputs_values,
        strides=resolved_strides if subtype_enum != YOLOSubtype.V26 else strides,
        conf_threshold=conf_threshold,
        n_classes=resolved_n_classes,
        iou_threshold=iou_threshold,
        max_det=max_det,
        anchors=anchors,
        n_keypoints=resolved_n_keypoints,
        label_names=ordered_class_names(class_map),
        keypoint_label_names=keypoint_label_names,
        keypoint_edges=keypoint_edges,
        input_shape=(model_spec.height, model_spec.width),
        kpts_outputs=kpts_outputs,
        masks_outputs_values=masks_outputs_values,
        protos_output=protos_output,
        protos_len=protos_len,
        mask_conf=mask_conf,
        v26_mask_coeffs=v26_mask_coeffs,
        v26_protos=v26_protos,
        v26_pose_kpts=v26_pose_kpts,
    )


def build_yolo_instance_masks(
    compute_inputs: YOLOComputeInputs,
    *,
    outputs_values: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Rebuild per-instance masks with refinement.

    Detection selection still comes from the DepthAI YOLO decode path.
    This helper only regenerates instance masks from the kept detections.

    Mirrored LuxonisTrain's
    ``luxonis_train.nodes.heads.precision_seg_bbox_head.refine_and_apply_masks()``:
    Eval uses the same prototype-combination, bbox-cropping, and
    upsampling behavior.
    """

    subtype = compute_inputs.subtype
    input_shape = compute_inputs.input_shape
    if input_shape is None:
        raise ValueError("YOLO mask rebuilding requires an input shape.")
    height, width = input_shape

    if subtype == YOLOSubtype.V26:
        results, mask_coeffs = decode_yolo26(
            compute_inputs.outputs_values[0],
            compute_inputs.conf_threshold,
            compute_inputs.max_det,
            extra_raw=compute_inputs.v26_mask_coeffs,
        )
        if mask_coeffs is None:
            raise ValueError(
                "YOLO26 instance segmentation requires mask coefficients."
            )
        mask_prototypes = compute_inputs.v26_protos
        if mask_prototypes is None:
            raise ValueError(
                "YOLO26 instance segmentation requires prototype masks."
            )
        return _refine_instance_masks(
            mask_prototypes=mask_prototypes,
            mask_coefficients=mask_coeffs,
            bounding_boxes=results[:, :4],
            height=height,
            width=width,
        )

    resolved_outputs_values = outputs_values or compute_inputs.outputs_values

    resolved_strides = resolve_yolo_strides(
        compute_inputs.strides,
        subtype,
        num_outputs=len(resolved_outputs_values),
    )

    anchors = compute_inputs.anchors
    anchors_array = (
        np.asarray(anchors, dtype=np.float32).reshape(
            len(resolved_strides), -1
        )
        if anchors is not None
        else None
    )

    results = decode_yolo_output(
        resolved_outputs_values,
        resolved_strides,
        anchors_array,
        conf_thres=compute_inputs.conf_threshold,
        iou_thres=compute_inputs.iou_threshold,
        num_classes=compute_inputs.n_classes,
        det_mode=False,
        subtype=subtype,
    )

    protos_output = compute_inputs.protos_output
    masks_outputs_values = compute_inputs.masks_outputs_values
    protos_len = compute_inputs.protos_len
    if protos_output is None or masks_outputs_values is None or protos_len is None:
        raise ValueError(
            "YOLO instance segmentation requires prototype and mask outputs."
        )

    mask_coefficients = []
    for other in results[:, 6:]:
        hi, ai, xi, yi = other.astype(int)
        mask_coefficients.append(
            masks_outputs_values[hi][0, ai * protos_len : (ai + 1) * protos_len, yi, xi]
        )

    if not mask_coefficients:
        return np.zeros((0, height, width), dtype=np.uint8)

    return _refine_instance_masks(
        mask_prototypes=protos_output[0],
        mask_coefficients=np.stack(mask_coefficients, axis=0),
        bounding_boxes=results[:, :4],
        height=height,
        width=width,
    )


def _refine_instance_masks(
    *,
    mask_prototypes: np.ndarray,
    mask_coefficients: np.ndarray,
    bounding_boxes: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """NumPy/Torch port of LuxonisTrain's mask refinement step.

    This mirrors
    ``luxonis_train.nodes.heads.precision_seg_bbox_head.refine_and_apply_masks()``
    because DepthAI returns a merged instance-id mask, while evaluation needs
    the per-instance masks before that merge.
    """
    if mask_coefficients.shape[0] == 0 or bounding_boxes.shape[0] == 0:
        return np.zeros((0, height, width), dtype=np.uint8)

    prototypes_tensor = torch.as_tensor(mask_prototypes, dtype=torch.float32)
    coefficients_tensor = torch.as_tensor(
        mask_coefficients, dtype=torch.float32
    )
    boxes_tensor = torch.as_tensor(bounding_boxes, dtype=torch.float32)

    channels, proto_h, proto_w = prototypes_tensor.shape
    masks_combined = (
        coefficients_tensor @ prototypes_tensor.view(channels, -1)
    ).view(-1, proto_h, proto_w)

    scaled_boxes = boxes_tensor.clone()
    scaled_boxes[:, [0, 2]] *= proto_w / width
    scaled_boxes[:, [1, 3]] *= proto_h / height

    cropped_masks = _apply_bounding_box_to_masks(masks_combined, scaled_boxes)
    upsampled_masks = F.interpolate(
        cropped_masks.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return (upsampled_masks > 0).to(torch.uint8).cpu().numpy()


def _apply_bounding_box_to_masks(
    masks: torch.Tensor,
    bounding_boxes: torch.Tensor,
) -> torch.Tensor:
    """Mirror LuxonisTrain's bbox mask cropping helper.

    This matches
    ``luxonis_train.utils.boundingbox.apply_bounding_box_to_masks()`` so the
    rebuilt masks follow the same crop semantics.
    """
    _, mask_height, mask_width = masks.shape
    left, top, right, bottom = torch.split(
        bounding_boxes[:, :, None], 1, dim=1
    )
    width_indices = torch.arange(
        mask_width, device=masks.device, dtype=left.dtype
    )[None, None, :]
    height_indices = torch.arange(
        mask_height, device=masks.device, dtype=left.dtype
    )[None, :, None]

    return masks * (
        (width_indices >= left)
        & (width_indices < right)
        & (height_indices >= top)
        & (height_indices < bottom)
    )
