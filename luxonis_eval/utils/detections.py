import depthai as dai
import numpy as np

INSTANCE_MASK_BACKGROUND = 255


def get_instance_masks(
    predictions: dai.ImgDetections,
    *,
    height: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    """Extract per-detection masks from an ``ImgDetections`` message.

    ``ImgDetections`` stores instance segmentation as a single indexed mask,
    where each foreground pixel contains the index of its detection and value
    ``255`` denotes background. This function converts that representation to
    a boolean array of shape ``(N, H, W)``.

    Parameters
    ----------
    predictions : dai.ImgDetections
        Detection message containing an indexed segmentation mask.
    height : int | None, optional
        Expected mask height. Also supplies the height for an empty message
        without a segmentation mask.
    width : int | None, optional
        Expected mask width. Also supplies the width for an empty message
        without a segmentation mask.

    Returns
    -------
    np.ndarray
        Boolean instance masks with shape ``(N, H, W)``.
    """
    if (height is None) != (width is None):
        raise ValueError("Both height and width must be provided together.")

    n_detections = len(predictions.detections)
    indexed_mask = predictions.getCvSegmentationMask()
    if indexed_mask is None:
        if n_detections:
            raise ValueError(
                "ImgDetections contains detections but no segmentation mask."
            )
        return np.zeros(
            (0, height or 0, width or 0),
            dtype=bool,
        )

    mask = np.asarray(indexed_mask)
    if mask.ndim != 2:
        raise ValueError(
            "ImgDetections segmentation mask must have shape (H, W), "
            f"got {mask.shape}."
        )

    if height is not None and mask.shape != (height, width):
        raise ValueError(
            "ImgDetections segmentation mask has an unexpected shape. "
            f"Expected ({height}, {width}), got {mask.shape}."
        )

    if n_detections == 0:
        return np.zeros((0, *mask.shape), dtype=bool)

    if n_detections > INSTANCE_MASK_BACKGROUND:
        raise ValueError(
            "ImgDetections indexed segmentation masks support at most 255 "
            "detections because index 255 is reserved for background."
        )

    instance_ids = np.arange(n_detections, dtype=mask.dtype)[:, None, None]
    foreground = mask != INSTANCE_MASK_BACKGROUND
    return foreground[None, :, :] & (mask[None, :, :] == instance_ids)
