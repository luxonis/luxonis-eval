import colorsys
from collections.abc import Mapping, Sequence

import depthai as dai
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import box_convert
from torchvision.utils import (
    draw_bounding_boxes,
)
from torchvision.utils import (
    draw_segmentation_masks as torchvision_draw_segmentation_masks,
)

from luxonis_eval.core.context import EvalContext
from luxonis_eval.metrics.metrics_utils import detection_to_coco_xywh
from luxonis_eval.parsers.yolo import get_prediction_instance_masks

from .base_visualizer import VisualizationData

Color = str | tuple[int, int, int]


def get_color(seed: int) -> Color:
    """Generate a deterministic color for a class index."""
    hue = ((seed + 45) * 157) % 360
    red, green, blue = colorsys.hls_to_rgb(hue / 360, 0.5, 0.8)
    return int(red * 255), int(green * 255), int(blue * 255)


def prepare_visualization_frame(
    frame: np.ndarray,
    *,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> np.ndarray:
    """Convert an engine visualization frame to an RGB-like uint8 image.

    ``mean`` and ``std`` undo host-side normalization. DepthAI frames do not
    need these values because NNArchive preprocessing happens on-device.
    """
    image = np.asarray(frame)
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3 or image.shape[2] not in {1, 3, 4}:
        raise ValueError(
            "Visualization frame must have shape HW or HWC with one, three, "
            f"or four channels, got {image.shape}."
        )

    image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32, copy=True)
        if mean is not None or std is not None:
            if mean is None or std is None:
                raise ValueError(
                    "Both mean and std are required to denormalize a frame."
                )
            channels = image.shape[2]
            mean_array = _channel_values(mean, channels, "mean")
            std_array = _channel_values(std, channels, "std")
            image = image * std_array + mean_array
            image *= 255.0
        elif image.size == 0 or (
            float(image.min()) >= 0.0 and float(image.max()) <= 1.0
        ):
            image *= 255.0

    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def numpy_to_batched_canvas(image: np.ndarray) -> torch.Tensor:
    """Convert an ``HWC``/``HW`` uint8 image to a ``1CHW`` tensor."""
    image = prepare_visualization_frame(image)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)


def combine_visualizations(
    visualization: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Combine target and prediction canvases side by side."""
    targets, predictions = visualization
    if targets.ndim != 4 or predictions.ndim != 4:
        raise ValueError("Visualization canvases must use BCHW layout.")
    if targets.shape[0] != 1 or predictions.shape[0] != 1:
        raise ValueError("LuxonisEval visualizes one sample at a time.")
    if targets.shape[:3] != predictions.shape[:3]:
        raise ValueError(
            "Target and prediction canvases must have matching batch, "
            "channel, and height dimensions."
        )
    return torch.cat([targets[0], predictions[0]], dim=-1)


def convert_detection_predictions(
    detections: Sequence[dai.ImgDetection], context: EvalContext
) -> Tensor:
    """Convert DepthAI detections to pixel ``xyxy`` drawing rows."""
    rows: list[list[float]] = []
    for detection in detections:
        x, y, width, height = detection_to_coco_xywh(
            detection,
            context.width,
            context.height,
        )
        rows.append(
            [
                x,
                y,
                x + width,
                y + height,
                float(detection.confidence),
                float(detection.label),
            ]
        )
    if not rows:
        return torch.empty((0, 6), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def convert_bounding_box_targets(
    target_boxes: np.ndarray, context: EvalContext
) -> Tensor:
    """Convert normalized LDF bounding boxes to batched drawing rows."""
    boxes = np.asarray(target_boxes, dtype=np.float32)
    if boxes.size == 0:
        return torch.empty((0, 6), dtype=torch.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError(
            "Bounding-box targets must have shape (N, 5) as "
            "[class, x, y, width, height], got "
            f"{boxes.shape}."
        )

    boxes = boxes.copy()
    if context.class_index_map is not None:
        try:
            boxes[:, 0] = [
                context.class_index_map[int(class_id)]
                for class_id in boxes[:, 0]
            ]
        except KeyError as error:
            raise ValueError(
                f"Target class {error.args[0]} has no model class mapping."
            ) from error

    batch_indices = np.zeros((boxes.shape[0], 1), dtype=np.float32)
    return torch.from_numpy(np.concatenate([batch_indices, boxes], axis=1))


def convert_keypoint_predictions(
    detections: Sequence[dai.ImgDetection],
    context: EvalContext,
    *,
    expected_keypoints: int | None = None,
) -> Tensor:
    """Convert normalized DepthAI keypoints to pixel ``(x, y, visibility)``."""
    rows: list[list[list[float]]] = []
    n_keypoints = expected_keypoints
    for detection in detections:
        detection_keypoints = detection.getKeypoints()
        if n_keypoints is None:
            n_keypoints = len(detection_keypoints)
        if len(detection_keypoints) != n_keypoints:
            raise ValueError(
                "All detections and targets must have the same number of "
                f"keypoints; expected {n_keypoints}, got "
                f"{len(detection_keypoints)}."
            )
        rows.append(
            [
                [
                    float(keypoint.imageCoordinates.x) * context.width,
                    float(keypoint.imageCoordinates.y) * context.height,
                    float(keypoint.confidence),
                ]
                for keypoint in detection_keypoints
            ]
        )

    n_keypoints = n_keypoints or 0
    if not rows:
        return torch.empty((0, n_keypoints, 3), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def convert_keypoint_targets(
    target_keypoints: np.ndarray,
    *,
    expected_instances: int,
) -> Tensor:
    """Convert normalized keypoint targets to flattened batched rows."""
    keypoints = np.asarray(target_keypoints, dtype=np.float32)
    if keypoints.ndim == 3:
        if keypoints.shape[2] != 3:
            raise ValueError(
                "Keypoint targets must end in (x, y, visibility), got "
                f"shape {keypoints.shape}."
            )
        keypoints = keypoints.reshape(keypoints.shape[0], -1)
    elif keypoints.ndim != 2:
        raise ValueError(
            "Keypoint targets must have shape (N, K * 3) or (N, K, 3), "
            f"got {keypoints.shape}."
        )

    if keypoints.shape[0] != expected_instances:
        raise ValueError(
            "Bounding-box and keypoint target counts must match; got "
            f"{expected_instances} boxes and {keypoints.shape[0]} keypoint "
            "rows."
        )
    if keypoints.shape[1] % 3 != 0:
        raise ValueError(
            "Flattened keypoint targets must contain a multiple of three "
            f"values per instance, got {keypoints.shape[1]}."
        )

    batch_indices = np.zeros((keypoints.shape[0], 1), dtype=np.float32)
    return torch.from_numpy(
        np.concatenate([batch_indices, keypoints], axis=1)
    )


def convert_visualization_data(
    predictions: object,
    target: dict[str, np.ndarray],
    context: EvalContext,
    target_keys: Sequence[str],
) -> VisualizationData:
    """Convert supported runtime messages and targets to task-keyed tensors."""
    requested_targets = set(target_keys)
    missing_targets = requested_targets - set(target)
    if missing_targets:
        raise ValueError(
            f"Visualization targets are missing keys {sorted(missing_targets)}."
        )

    converted_predictions: dict[str, list[Tensor]] = {}
    converted_targets: dict[str, Tensor] = {}

    if isinstance(predictions, dai.ImgDetections):
        target_boxes: Tensor | None = None
        if "/boundingbox" in requested_targets:
            converted_predictions["boundingbox"] = [
                convert_detection_predictions(
                    predictions.detections, context
                )
            ]
            target_boxes = convert_bounding_box_targets(
                target["/boundingbox"], context
            )
            converted_targets["boundingbox"] = target_boxes

        if "/instance_segmentation" in requested_targets:
            if target_boxes is None:
                raise ValueError(
                    "Instance segmentation conversion requires bounding-box "
                    "targets."
                )
            converted_predictions["instance_segmentation"] = [
                _convert_instance_prediction_masks(predictions, context)
            ]
            converted_targets["instance_segmentation"] = _mask_tensor(
                target["/instance_segmentation"],
                expected_instances=target_boxes.shape[0],
                context=context,
                source="Target",
            )

        if "/keypoints" in requested_targets:
            if target_boxes is None:
                raise ValueError(
                    "Keypoint conversion requires bounding-box targets."
                )
            target_keypoints = convert_keypoint_targets(
                target["/keypoints"],
                expected_instances=target_boxes.shape[0],
            )
            expected_keypoints = (target_keypoints.shape[1] - 1) // 3
            converted_predictions["keypoints"] = [
                convert_keypoint_predictions(
                    predictions.detections,
                    context,
                    expected_keypoints=expected_keypoints or None,
                )
            ]
            converted_targets["keypoints"] = target_keypoints

    elif isinstance(predictions, dai.SegmentationMask):
        if "/segmentation" not in requested_targets:
            raise ValueError(
                "Semantic segmentation conversion requires a segmentation "
                "target."
            )
        prediction_mask = predictions.getCvMask()
        if prediction_mask is None:
            raise ValueError("Segmentation prediction does not contain a mask.")
        prediction_channels, target_channels = (
            _convert_semantic_segmentation(
                np.asarray(prediction_mask),
                np.asarray(target["/segmentation"]),
                context,
            )
        )
        converted_predictions["segmentation"] = [
            torch.from_numpy(prediction_channels)
        ]
        converted_targets["segmentation"] = torch.from_numpy(
            target_channels[None]
        )
    else:
        raise TypeError(
            "Visualization conversion supports dai.ImgDetections and "
            f"dai.SegmentationMask, got {type(predictions).__name__}."
        )

    return VisualizationData(
        predictions=converted_predictions,
        targets=converted_targets,
    )


def _convert_instance_prediction_masks(
    predictions: dai.ImgDetections,
    context: EvalContext,
) -> Tensor:
    n_detections = len(predictions.detections)
    raw_masks = get_prediction_instance_masks(predictions)
    if raw_masks is None:
        indexed_mask = predictions.getCvSegmentationMask()
        if indexed_mask is None or indexed_mask.size == 0:
            if n_detections == 0:
                return torch.zeros(
                    (0, context.height, context.width), dtype=torch.bool
                )
            raise ValueError(
                "Instance segmentation predictions do not contain masks."
            )
        indexed_mask = np.asarray(indexed_mask)
        if indexed_mask.shape != (context.height, context.width):
            raise ValueError(
                "Prediction mask shape must match the model input shape "
                f"({context.height}, {context.width}), got "
                f"{indexed_mask.shape}."
            )
        if n_detections == 0:
            return torch.zeros(
                (0, context.height, context.width), dtype=torch.bool
            )
        raw_masks = np.stack(
            [indexed_mask == index for index in range(n_detections)],
            axis=0,
        )

    return _mask_tensor(
        raw_masks,
        expected_instances=n_detections,
        context=context,
        source="Prediction",
    )


def _mask_tensor(
    masks: np.ndarray,
    *,
    expected_instances: int,
    context: EvalContext,
    source: str,
) -> Tensor:
    array = np.asarray(masks)
    expected_shape = (
        expected_instances,
        context.height,
        context.width,
    )
    if array.size == 0 and expected_instances == 0:
        return torch.zeros(expected_shape, dtype=torch.bool)
    if array.shape != expected_shape:
        raise ValueError(
            f"{source} masks must have shape {expected_shape}, got "
            f"{array.shape}."
        )
    return torch.from_numpy(array.astype(bool, copy=False))


def _convert_semantic_segmentation(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    context: EvalContext,
) -> tuple[np.ndarray, np.ndarray]:
    if prediction_mask.ndim != 2:
        raise ValueError(
            "Segmentation prediction must have shape (height, width), "
            f"got {prediction_mask.shape}."
        )
    _validate_mask_spatial_shape(prediction_mask, context, "Prediction")
    target_channels = _semantic_target_channels(target_mask, context)

    if target_mask.ndim == 3 and target_mask.shape[0] == 1:
        prediction_channels = (prediction_mask == 0)[None]
    else:
        prediction_channels = np.stack(
            [
                prediction_mask
                == _prediction_class_id(target_class_id, context)
                for target_class_id in range(target_channels.shape[0])
            ],
            axis=0,
        )
    return prediction_channels, target_channels


def _semantic_target_channels(
    target: np.ndarray,
    context: EvalContext,
) -> np.ndarray:
    if target.ndim not in {2, 3}:
        raise ValueError(
            "Segmentation targets must have shape (height, width) or "
            f"(classes, height, width), got {target.shape}."
        )
    _validate_mask_spatial_shape(target, context, "Target")

    if target.ndim == 3:
        if target.shape[0] == 0:
            raise ValueError(
                "Segmentation targets must contain at least one class channel."
            )
        return target.astype(bool, copy=False)

    assigned = target[target != 255]
    if assigned.size and np.any(assigned < 0):
        raise ValueError(
            "Segmentation target class indices must be non-negative."
        )
    class_indices = list(context.target_class_map)
    if assigned.size:
        class_indices.extend(int(index) for index in np.unique(assigned))
    if not class_indices:
        raise ValueError(
            "Cannot infer class channels from an unassigned segmentation "
            "target without a target class map."
        )
    n_classes = max(class_indices) + 1
    return np.stack(
        [target == class_index for class_index in range(n_classes)],
        axis=0,
    )


def _prediction_class_id(
    target_class_id: int,
    context: EvalContext,
) -> int:
    mapping = context.class_index_map
    if mapping is None or target_class_id in mapping:
        return (
            mapping.get(target_class_id, target_class_id)
            if mapping
            else target_class_id
        )

    target_name = context.target_class_map.get(target_class_id)
    if target_name is not None:
        for prediction_class_id, prediction_names in context.class_map.items():
            if target_name in prediction_names.split(", "):
                return prediction_class_id
    return target_class_id


def _validate_mask_spatial_shape(
    mask: np.ndarray,
    context: EvalContext,
    source: str,
) -> None:
    if mask.shape[-2:] != (context.height, context.width):
        raise ValueError(
            f"{source} segmentation mask must have spatial shape "
            f"({context.height}, {context.width}), got {mask.shape[-2:]}."
        )


def draw_bounding_box_targets(
    image: Tensor,
    boxes: Tensor,
    **kwargs: object,
) -> Tensor:
    """Draw normalized ``xywh`` target boxes on a CHW image."""
    if boxes.shape[0] == 0:
        return image
    height, width = image.shape[-2:]
    boxes_xyxy = box_convert(boxes.clone(), "xywh", "xyxy")
    boxes_xyxy[:, 0::2] *= width
    boxes_xyxy[:, 1::2] *= height
    return draw_bounding_boxes(image, boxes_xyxy, **kwargs)  # type: ignore[arg-type]


def draw_segmentation_masks(
    image: Tensor,
    masks: Tensor,
    *,
    alpha: float,
    colors: list[Color],
) -> Tensor:
    """Draw boolean masks, leaving the image unchanged if empty."""
    if masks.shape[0] == 0:
        return image
    return torchvision_draw_segmentation_masks(
        image,
        masks.to(torch.bool),
        alpha=alpha,
        colors=colors,
    )


def get_prediction_labels(
    prediction: Tensor,
    label_dict: Mapping[int, str],
    *,
    draw_labels: bool,
    draw_scores: bool,
) -> list[str] | None:
    """Format optional class names and confidence scores for boxes."""
    if not (draw_labels or draw_scores):
        return None

    output: list[str] = []
    for row in prediction:
        class_id = int(row[5])
        parts: list[str] = []
        if draw_labels:
            parts.append(label_dict.get(class_id, str(class_id)))
        if draw_scores:
            parts.append(f"{float(row[4]):.2f}")
        output.append(" ".join(parts))
    return output


def scale_masks(masks: Tensor, scale: float) -> Tensor:
    """Resize instance masks with nearest-neighbor interpolation."""
    if scale == 1.0 or masks.shape[0] == 0:
        return masks
    height = max(1, round(masks.shape[-2] * scale))
    width = max(1, round(masks.shape[-1] * scale))
    return F.interpolate(
        masks[:, None].float(),
        size=(height, width),
        mode="nearest",
    )[:, 0].to(torch.bool)


def _channel_values(
    values: Sequence[float], channels: int, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 1:
        array = np.repeat(array, channels)
    elif channels == 1 and array.size == 3:
        array = array[:1]
    elif array.size != channels:
        raise ValueError(
            f"Normalization {name} has {array.size} values for a "
            f"{channels}-channel frame."
        )
    return array.reshape(1, 1, channels)
