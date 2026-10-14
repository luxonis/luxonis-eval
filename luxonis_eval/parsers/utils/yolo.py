from dataclasses import dataclass

import numpy as np
from depthai_nodes.node.parsers.utils.yolo import (
    YOLOSubtype,
    resolve_yolo_strides,
)
from depthai_nodes.node.parsers.yolo import YOLOComputeInputs

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
        strides=resolved_strides
        if subtype_enum != YOLOSubtype.V26
        else strides,
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
        mask_name = next(
            name for name in layer_names if "output_masks" in name
        )
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

    if (
        any("kpt_output" in name for name in layer_names)
        and subtype != YOLOSubtype.P
    ):
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

    if (
        any("_masks" in name for name in layer_names)
        and subtype != YOLOSubtype.P
    ):
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
