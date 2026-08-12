from dataclasses import dataclass

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
from luxonis_eval.utils.utils import ordered_class_names


@dataclass(frozen=True, slots=True)
class YoloTensorBundle:
    layer_names: list[str]
    outputs_values: list[np.ndarray]
    kpts_outputs: list[np.ndarray] | None = None
    masks_outputs_values: list[np.ndarray] | None = None
    protos_output: np.ndarray | None = None
    protos_len: int | None = None
    v26_mask_coeffs: np.ndarray | None = None
    v26_protos: np.ndarray | None = None
    v26_pose_kpts: np.ndarray | None = None


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
    subtype_enum = parse_yolo_subtype(subtype)
    tensors = extract_yolo_tensors(output, subtype_enum)

    resolved_strides: list[int] | None = None
    if subtype_enum == YOLOSubtype.V26:
        resolved_n_classes = n_classes or len(class_map)
    else:
        resolved_strides = resolve_yolo_strides(
            strides,
            subtype_enum,
            num_outputs=len(tensors.outputs_values),
        )
        anchors_array = reshape_anchors(anchors, resolved_strides)
        resolved_n_classes = resolve_num_classes(
            tensors.outputs_values[0],
            anchors_array,
            n_classes,
        )

    resolved_n_keypoints = resolve_num_keypoints(
        tensors.kpts_outputs,
        n_keypoints,
    )

    return YOLOComputeInputs(
        subtype=subtype_enum,
        layer_names=tensors.layer_names,
        outputs_values=tensors.outputs_values,
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
        kpts_outputs=tensors.kpts_outputs,
        masks_outputs_values=tensors.masks_outputs_values,
        protos_output=tensors.protos_output,
        protos_len=tensors.protos_len,
        mask_conf=mask_conf,
        v26_mask_coeffs=tensors.v26_mask_coeffs,
        v26_protos=tensors.v26_protos,
        v26_pose_kpts=tensors.v26_pose_kpts,
    )


def build_yolo_instance_masks(
    compute_inputs: YOLOComputeInputs,
    *,
    outputs_values: list[np.ndarray] | None = None,
) -> np.ndarray:
    input_shape = compute_inputs.input_shape
    if input_shape is None:
        raise ValueError("YOLO mask rebuilding requires an input shape.")
    height, width = input_shape

    if compute_inputs.subtype == YOLOSubtype.V26:
        return _build_yolo26_instance_masks(
            compute_inputs,
            height=height,
            width=width,
        )

    return _build_standard_instance_masks(
        compute_inputs,
        outputs_values=outputs_values,
        height=height,
        width=width,
    )


def parse_yolo_subtype(subtype: str) -> YOLOSubtype:
    try:
        return YOLOSubtype(subtype.lower())
    except ValueError as err:
        raise ValueError(
            f"Invalid YOLO subtype {subtype}. Supported YOLO subtypes are "
            f"{[member.value for member in YOLOSubtype][:-1]}."
        ) from err


def extract_yolo_tensors(
    output: EngineOutput,
    subtype: YOLOSubtype,
) -> YoloTensorBundle:
    layer_names = list(output.names())
    if subtype == YOLOSubtype.V26:
        return _extract_yolo26_tensors(output, layer_names)
    return _extract_standard_yolo_tensors(output, layer_names, subtype)


def _extract_yolo26_tensors(
    output: EngineOutput,
    layer_names: list[str],
) -> YoloTensorBundle:
    if any("output_masks" in name for name in layer_names):
        mask_name = next(name for name in layer_names if "output_masks" in name)
        protos_name = next(
            (name for name in layer_names if "protos" in name),
            "protos_output",
        )
        return YoloTensorBundle(
            layer_names=layer_names,
            outputs_values=[
                output.get("output_yolo26").astype(np.float32, copy=False)
            ],
            v26_mask_coeffs=output.get(mask_name).astype(
                np.float32,
                copy=False,
            ),
            v26_protos=output.get(protos_name, layout="NCHW").astype(
                np.float32,
                copy=False,
            )[0],
        )

    if any("kpt_output" in name for name in layer_names):
        kpt_name = next(name for name in layer_names if "kpt_output" in name)
        return YoloTensorBundle(
            layer_names=layer_names,
            outputs_values=[
                output.get("output_yolo26").astype(np.float32, copy=False)
            ],
            v26_pose_kpts=output.get(kpt_name).astype(np.float32, copy=False),
        )

    return YoloTensorBundle(
        layer_names=layer_names,
        outputs_values=[
            output.get(name).astype(np.float32, copy=False)
            for name in layer_names
        ],
    )


def _extract_standard_yolo_tensors(
    output: EngineOutput,
    layer_names: list[str],
    subtype: YOLOSubtype,
) -> YoloTensorBundle:
    outputs_names = sorted(
        [name for name in layer_names if "_yolo" in name or "yolo-" in name]
    )
    outputs_values = [
        output.get(name, layout="NCHW").astype(np.float32, copy=False)
        for name in outputs_names
    ]

    if any("kpt_output" in name for name in layer_names) and subtype != YOLOSubtype.P:
        kpts_output_names = sorted(
            [name for name in layer_names if "kpt_output" in name]
        )
        return YoloTensorBundle(
            layer_names=layer_names,
            outputs_values=outputs_values,
            kpts_outputs=[
                output.get(name).astype(np.float32, copy=False)
                for name in kpts_output_names
            ],
        )

    if any("_masks" in name for name in layer_names) and subtype != YOLOSubtype.P:
        protos_name = next(
            (name for name in layer_names if "protos" in name),
            "protos_output",
        )
        mask_output_names = sorted(
            [name for name in layer_names if "_masks" in name]
        )
        protos_output = output.get(protos_name, layout="NCHW").astype(
            np.float32,
            copy=False,
        )
        return YoloTensorBundle(
            layer_names=layer_names,
            outputs_values=outputs_values,
            masks_outputs_values=[
                output.get(name, layout="NCHW").astype(np.float32, copy=False)
                for name in mask_output_names
            ],
            protos_output=protos_output,
            protos_len=protos_output.shape[1],
        )

    return YoloTensorBundle(
        layer_names=layer_names,
        outputs_values=outputs_values,
    )


def reshape_anchors(
    anchors: list[list[list[float]]] | None,
    strides: list[int],
) -> np.ndarray | None:
    if not anchors:
        return None
    return np.asarray(anchors, dtype=np.float32).reshape(len(strides), -1)


def resolve_num_classes(
    output_tensor: np.ndarray,
    anchors: np.ndarray | None,
    configured_n_classes: int | None,
) -> int:
    n_anchors_per_head = anchors.shape[1] // 2 if anchors is not None else 1
    inferred_n_classes = (
        output_tensor.shape[1] - 5
        if anchors is None
        else (output_tensor.shape[1] // n_anchors_per_head) - 5
    )
    if (
        configured_n_classes is not None
        and inferred_n_classes != configured_n_classes
    ):
        raise ValueError(
            f"The provided number of classes {configured_n_classes} does not match the "
            f"model's {inferred_n_classes}."
        )
    return inferred_n_classes


def resolve_num_keypoints(
    kpts_outputs: list[np.ndarray] | None,
    configured_n_keypoints: int | None,
) -> int:
    inferred_n_keypoints = (
        kpts_outputs[0].shape[1] // 3 if kpts_outputs is not None else None
    )
    if (
        configured_n_keypoints is not None
        and inferred_n_keypoints is not None
        and inferred_n_keypoints != configured_n_keypoints
    ):
        raise ValueError(
            f"The provided number of keypoints {configured_n_keypoints} does not match "
            f"the model's {inferred_n_keypoints}."
        )
    return inferred_n_keypoints or configured_n_keypoints or 17


def _build_yolo26_instance_masks(
    compute_inputs: YOLOComputeInputs,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    results, mask_coefficients = decode_yolo26(
        compute_inputs.outputs_values[0],
        compute_inputs.conf_threshold,
        compute_inputs.max_det,
        extra_raw=compute_inputs.v26_mask_coeffs,
    )
    if mask_coefficients is None:
        raise ValueError(
            "YOLO26 instance segmentation requires mask coefficients."
        )
    if compute_inputs.v26_protos is None:
        raise ValueError(
            "YOLO26 instance segmentation requires prototype masks."
        )
    return _refine_instance_masks(
        mask_prototypes=compute_inputs.v26_protos,
        mask_coefficients=mask_coefficients,
        bounding_boxes=results[:, :4],
        height=height,
        width=width,
    )


def _build_standard_instance_masks(
    compute_inputs: YOLOComputeInputs,
    *,
    outputs_values: list[np.ndarray] | None,
    height: int,
    width: int,
) -> np.ndarray:
    resolved_outputs_values = outputs_values or compute_inputs.outputs_values
    resolved_strides = resolve_yolo_strides(
        compute_inputs.strides,
        compute_inputs.subtype,
        num_outputs=len(resolved_outputs_values),
    )
    anchors_array = reshape_anchors(compute_inputs.anchors, resolved_strides)

    results = decode_yolo_output(
        resolved_outputs_values,
        resolved_strides,
        anchors_array,
        conf_thres=compute_inputs.conf_threshold,
        iou_thres=compute_inputs.iou_threshold,
        num_classes=compute_inputs.n_classes,
        det_mode=False,
        subtype=compute_inputs.subtype,
    )

    if (
        compute_inputs.protos_output is None
        or compute_inputs.masks_outputs_values is None
        or compute_inputs.protos_len is None
    ):
        raise ValueError(
            "YOLO instance segmentation requires prototype and mask outputs."
        )

    mask_coefficients = _collect_mask_coefficients(
        results[:, 6:],
        compute_inputs.masks_outputs_values,
        compute_inputs.protos_len,
    )
    if mask_coefficients.size == 0:
        return np.zeros((0, height, width), dtype=np.uint8)

    return _refine_instance_masks(
        mask_prototypes=compute_inputs.protos_output[0],
        mask_coefficients=mask_coefficients,
        bounding_boxes=results[:, :4],
        height=height,
        width=width,
    )


def _collect_mask_coefficients(
    detections_metadata: np.ndarray,
    masks_outputs_values: list[np.ndarray],
    protos_len: int,
) -> np.ndarray:
    if detections_metadata.size == 0:
        return np.zeros((0, protos_len), dtype=np.float32)

    mask_coefficients = [
        masks_outputs_values[hi][
            0,
            ai * protos_len : (ai + 1) * protos_len,
            yi,
            xi,
        ]
        for hi, ai, xi, yi in detections_metadata.astype(int)
    ]
    return np.stack(mask_coefficients, axis=0)


def _refine_instance_masks(
    *,
    mask_prototypes: np.ndarray,
    mask_coefficients: np.ndarray,
    bounding_boxes: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
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
    _, mask_height, mask_width = masks.shape
    left, top, right, bottom = torch.split(
        bounding_boxes[:, :, None],
        1,
        dim=1,
    )
    width_indices = torch.arange(
        mask_width,
        device=masks.device,
        dtype=left.dtype,
    )[None, None, :]
    height_indices = torch.arange(
        mask_height,
        device=masks.device,
        dtype=left.dtype,
    )[None, :, None]

    return masks * (
        (width_indices >= left)
        & (width_indices < right)
        & (height_indices >= top)
        & (height_indices < bottom)
    )
