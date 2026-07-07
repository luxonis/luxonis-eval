from collections.abc import Sequence
from typing import Any

import depthai as dai
import numpy as np
from pycocotools import mask as mask_utils


def normalized_xywh_to_coco_xywh(
    target: np.ndarray, img_w: int, img_h: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized top-left ``xywh`` labels to COCO ``xywh``
    boxes.

    Parameters
    ----------
    target : np.ndarray
        Array of shape ``(N, 5)`` as ``(class_id, x_min, y_min, w, h)``.
    img_w : int
        Image width in pixels.
    img_h : int
        Image height in pixels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Class indices and COCO-format boxes in pixels.
    """
    target = np.asarray(target, dtype=np.float32)
    if target.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros(
            (0, 4), dtype=np.float32
        )

    cls_idc = target[:, 0].astype(np.int64)
    x = target[:, 1] * img_w
    y = target[:, 2] * img_h
    bw = target[:, 3] * img_w
    bh = target[:, 4] * img_h
    boxes_xywh = np.stack([x, y, bw, bh], axis=-1)
    return cls_idc, boxes_xywh


def detection_to_coco_xywh(
    detection: dai.ImgDetection, width: int, height: int
) -> list[float]:
    """Convert a detection object to COCO top-left ``xywh`` pixels.

    Parameters
    ----------
    detection : dai.ImgDetection
        DepthAI detection with a normalized bounding box.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    """
    box = detection.getBoundingBox().denormalize(width, height).getOuterXYWH()
    return [
        float(box[0].x),
        float(box[0].y),
        float(box[1].width),
        float(box[1].height),
    ]


def binary_mask_to_rle(binary_mask: np.ndarray) -> Any:
    """Convert a binary mask to COCO RLE format.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask of shape (H, W).

    Returns
    -------
    dict[str, Any]
        COCO RLE representation.
    """
    m = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(m)
    rle["counts"] = rle["counts"].decode("utf-8")  # type: ignore
    return rle


def area_from_rle(rle: Any) -> float:
    return float(mask_utils.area(rle))


def bbox_from_rle(rle: Any) -> list[float]:
    return [float(x) for x in mask_utils.toBbox(rle)]


def remap_prediction_mask(
    pred_mask: np.ndarray,
    class_index_map: dict[int, int],
) -> np.ndarray:
    """Remap prediction mask indices to align with target mask indices.

    Parameters
    ----------
    pred_mask : np.ndarray
        The predicted segmentation mask with class indices.
    class_index_map : dict[int, int]
        Mapping from target class index to prediction class index.

    Returns
    -------
    np.ndarray
        The remapped prediction mask with indices aligned to the target mask.
    """
    remapped = np.full(pred_mask.shape, fill_value=255, dtype=np.int64)

    for target_idx, pred_idx in class_index_map.items():
        remapped[pred_mask == pred_idx] = target_idx

    return remapped


def target_segmentation_to_index_mask(
    target_mask: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Convert a segmentation target tensor to an index mask.

    Parameters
    ----------
    target_mask : np.ndarray
        Target segmentation tensor, either shaped ``(C, H, W)`` or ``(H, W)``.

    Returns
    -------
    tuple[np.ndarray, bool]
        The index mask and whether the target represents a binary single-channel mask.
    """
    mask = np.asarray(target_mask)

    if mask.ndim == 2:
        return mask.astype(np.int64), False

    if mask.ndim != 3:
        raise ValueError(
            f"Expected segmentation target with 2 or 3 dimensions, got {mask.ndim}."
        )

    if mask.shape[0] == 1:
        return mask[0].astype(np.int64), True

    return np.argmax(mask, axis=0).astype(np.int64), False


def normalize_prediction_segmentation_mask(
    pred_mask: np.ndarray,
    *,
    binary_target: bool,
) -> np.ndarray:
    """Normalize a predicted segmentation mask to class indices.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted segmentation mask.
    binary_target : bool
        Whether the corresponding target is a binary single-channel mask.

    Returns
    -------
    np.ndarray
        Normalized predicted class-index mask.
    """
    mask = np.asarray(pred_mask).astype(np.int64)

    if binary_target and mask.min() < 0:
        # DepthAI semantic masks use -1 for background and 0 for the only
        # foreground class. Binary metrics expect 0 for background and 1 for
        # foreground, so shift the mask into that range.
        mask = mask + 1

    return mask


def mask_ignore_pixels(
    pred_mask: np.ndarray, target_mask: np.ndarray, ignore_index: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Replace ignore_index pixels with 0 in both masks so they cancel
    out and don't affect IoU for any real class.

    Parameters
    ----------
    pred_mask : np.ndarray
        The predicted segmentation mask with class indices.
    target_mask : np.ndarray
        The ground-truth segmentation mask with class indices.
    ignore_index : int, default=0
        The class index to ignore (e.g., background).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The modified prediction and target masks with ignore_index pixels set to 0.
    """
    ignore_mask = target_mask == ignore_index
    target_mask = target_mask.copy()
    pred_mask = pred_mask.copy()

    target_mask[ignore_mask] = 0
    pred_mask[ignore_mask] = 0

    return pred_mask, target_mask


def binary_segmentation_confusion(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[int, int, int]:
    """Compute TP, FP and FN for binary segmentation masks."""
    pred_fg = np.asarray(pred_mask) > 0
    target_fg = np.asarray(target_mask) > 0

    tp = int(np.logical_and(pred_fg, target_fg).sum())
    fp = int(np.logical_and(pred_fg, np.logical_not(target_fg)).sum())
    fn = int(np.logical_and(np.logical_not(pred_fg), target_fg).sum())
    return tp, fp, fn


def to_coco_kpts_flat(kpts: np.ndarray) -> list[float]:
    """Convert keypoints to COCO flat list [x,y,v,...].

    Parameters
    ----------
    kpts : np.ndarray
        Keypoints array, either flat (3K,) or shaped (K,3).

    Returns
    -------
    list[float]
        Flattened keypoints list in COCO format [x,y,v,...].
    """
    kpts = np.asarray(kpts, dtype=float)

    if kpts.ndim == 1:
        if kpts.size % 3 != 0:
            raise ValueError(
                f"Keypoints flat array must have length multiple of 3; got {kpts.size}."
            )
        flat = kpts
    elif kpts.ndim == 2 and kpts.shape[1] == 3:
        flat = kpts.reshape(-1)
    else:
        raise ValueError(
            f"Unsupported keypoints shape {kpts.shape}; expected (K,3) or (3K,)."
        )

    return [float(x) for x in flat.tolist()]


def bbox_area_from_keypoints(
    kpts_flat: Sequence[float],
) -> tuple[list[float], float, int]:
    """Compute [x,y,w,h], area, num_keypoints from COCO flat keypoints.

    Parameters
    ----------
    kpts_flat : Sequence[float]
        Flattened keypoints list in COCO format [x,y,v,...].

    Returns
    -------
    tuple[list[float], float, int]
        Bounding box [x,y,w,h], area, and number of keypoints.
    """
    arr = np.asarray(kpts_flat, dtype=float)
    if arr.size % 3 != 0:
        raise ValueError("kpts_flat must be a multiple of 3.")

    xs = arr[0::3]
    ys = arr[1::3]
    vs = arr[2::3]

    labeled = vs > 0
    num_kpts = int(np.count_nonzero(labeled))

    if num_kpts == 0:
        return [0.0, 0.0, 0.0, 0.0], 0.0, 0

    x_min = float(np.min(xs[labeled]))
    y_min = float(np.min(ys[labeled]))
    x_max = float(np.max(xs[labeled]))
    y_max = float(np.max(ys[labeled]))

    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)
    area = float(w * h)

    return [x_min, y_min, float(w), float(h)], area, num_kpts
