import numpy as np

from luxonis_eval.metrics.metrics_utils import normalized_xywh_to_coco_xywh

BOUNDINGBOX_KEY = "/boundingbox"
PREPARED_BBOX_CLASSES_KEY = "/_bbox_classes"
PREPARED_BBOX_XYWH_KEY = "/_bbox_xywh"


def prepare_target(
    target: dict[str, np.ndarray], *, width: int, height: int
) -> dict[str, np.ndarray]:
    """Attach canonical per-sample bbox arrays used by box-based metrics."""
    if BOUNDINGBOX_KEY not in target:
        return target

    bbox_classes, bbox_xywh = normalized_xywh_to_coco_xywh(
        target[BOUNDINGBOX_KEY], width, height
    )
    prepared_target = dict(target)
    prepared_target[PREPARED_BBOX_CLASSES_KEY] = bbox_classes
    prepared_target[PREPARED_BBOX_XYWH_KEY] = bbox_xywh
    return prepared_target


def require_prepared_bboxes(
    target: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical bbox arrays prepared during sample normalization."""
    try:
        return (
            target[PREPARED_BBOX_CLASSES_KEY],
            target[PREPARED_BBOX_XYWH_KEY],
        )
    except KeyError as exc:
        raise RuntimeError(
            "Bounding box targets are missing prepared runtime fields. "
            "Call normalize_target(..., model_spec=...) before metric.update()."
        ) from exc
