from collections.abc import Sequence
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


def yolo_norm_to_coco_xywh(
    target: np.ndarray, img_w: int, img_h: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert YOLO-normalized labels to COCO xywh boxes.

    Parameters
    ----------
    target : np.ndarray
        Array of shape ``(N, 5)`` as ``(class_id, x_center, y_center, w, h)``.
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
    xc = target[:, 1] * img_w
    yc = target[:, 2] * img_h
    bw = target[:, 3] * img_w
    bh = target[:, 4] * img_h
    boxes_xywh = np.stack([xc, yc, bw, bh], axis=-1)
    return cls_idc, boxes_xywh


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
        The remapped prediction mask with indices aligned to the target mask."""
    remapped = np.full(pred_mask.shape, fill_value=255, dtype=np.int64)

    for target_idx, pred_idx in class_index_map.items():
        remapped[pred_mask == pred_idx] = target_idx

    return remapped


def mask_ignore_pixels(
    pred_mask: np.ndarray, target_mask: np.ndarray, ignore_index: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace ignore_index pixels with 0 in both masks so they cancel out and don't affect IoU for any real class.

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
