import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from luxonis_eval import __version__


@contextmanager
def suppress_stdout() -> Iterator[None]:
    """Suppress stdout within a context."""
    fd = sys.stdout.fileno()
    saved_fd = os.dup(fd)

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), fd)
        yield
    finally:
        os.dup2(saved_fd, fd)
        os.close(saved_fd)


class COCOStore:
    """A simple COCO-style storage for ground-truth and detection results."""

    def __init__(
        self,
        *,
        iou_type: str,
        kpt_oks_sigmas: Sequence[float] | None = None,
    ) -> None:
        """Initialize the COCOStore.

        Parameters
        ----------
        iou_type : str
            Type of IoU to evaluate ('bbox', 'segm', etc.).
        kpt_oks_sigmas : Sequence[float] | None, optional
            Optional OKS sigma values to use for keypoint evaluation.
        """
        self.iou_type = iou_type
        self.kpt_oks_sigmas = (
            np.asarray(kpt_oks_sigmas, dtype=np.float64)
            if kpt_oks_sigmas is not None
            else None
        )
        self.images: list[dict[str, Any]] = []
        self.gt_annotations: list[dict[str, Any]] = []
        self.pred_results: list[dict[str, Any]] = []
        self.categories: list[dict[str, Any]] | None = None
        self.category_ids_set: set[int] | None = None
        self.next_ann_id: int = 1
        self.next_img_id: int = 1
        self.num_keypoints: int | None = None

    def reset(self) -> None:
        """Reset internal storage."""
        self.images.clear()
        self.gt_annotations.clear()
        self.pred_results.clear()
        self.categories = None
        self.category_ids_set = None
        self.next_ann_id = 1
        self.next_img_id = 1
        self.num_keypoints = None

    def _track_num_keypoints(self, ann_like: dict[str, Any]) -> None:
        """Track and validate the number of keypoints used by annotations."""
        if self.iou_type != "keypoints" or "keypoints" not in ann_like:
            return

        keypoints = ann_like["keypoints"]
        n_values = len(keypoints)
        if n_values % 3 != 0:
            raise ValueError(
                f"COCO keypoints must be flat [x, y, v, ...]; got {n_values} values."
            )

        current_num_keypoints = n_values // 3
        if self.num_keypoints is None:
            self.num_keypoints = current_num_keypoints
            return

        if current_num_keypoints != self.num_keypoints:
            raise ValueError(
                "All keypoint annotations must use the same number of keypoints "
                f"for COCO keypoint evaluation. Expected {self.num_keypoints}, "
                f"got {current_num_keypoints}."
            )

    def init_categories_once(
        self,
        *,
        class_map: dict[int, str],
        category_ids: Sequence[int] | None,
    ) -> None:
        """Initialize categories if not already initialized.

        Parameters
        ----------
        class_map : dict[int, str]
            Mapping from native class indices to class names.
        category_ids : Sequence[int] | None
            Sequence of category IDs to include.
        """
        if self.categories is not None:
            return
        if category_ids is None:
            category_ids = sorted(class_map.keys())
        self.category_ids_set = {int(x) for x in category_ids}
        self.categories = [
            {
                "id": int(cid),
                "name": str(class_map.get(int(cid), str(cid))),
            }
            for cid in sorted(self.category_ids_set)
        ]

    def new_image(self, *, width: int, height: int) -> int:
        """Add a new image to the store.

        Parameters
        ----------
        width : int
            Width of the image.
        height : int
            Height of the image.

        Returns
        -------
        int
            The ID of the newly added image.
        """
        img_id = self.next_img_id
        self.images.append(
            {"id": int(img_id), "width": int(width), "height": int(height)}
        )
        self.next_img_id += 1
        return int(img_id)

    def add_gt(self, ann: dict[str, Any]) -> None:
        """Add a ground-truth annotation.

        Parameters
        ----------
        ann : dict[str, Any]
            Ground-truth annotation in COCO format.
        """
        if "id" not in ann:
            ann = dict(ann)
            ann["id"] = int(self.next_ann_id)
        self._track_num_keypoints(ann)
        self.gt_annotations.append(ann)
        self.next_ann_id = max(self.next_ann_id, int(ann["id"]) + 1)

    def add_pred(self, res: dict[str, Any]) -> None:
        """Add a prediction result.

        Parameters
        ----------
        res : dict[str, Any]
            Prediction result in COCO format.
        """
        self._track_num_keypoints(res)
        self.pred_results.append(res)

    def evaluate(self) -> dict[str, float]:
        """Evaluate prediction results using COCO evaluation.

        Returns
        -------
        dict[str, float]
            Computed mAP results.
        """
        coco_target_dict = {
            "info": {"description": "luxonis-eval", "version": __version__},
            "images": self.images,
            "annotations": self.gt_annotations,
            "categories": self.categories or [],
        }

        with suppress_stdout():
            coco_target = COCO()
            coco_target.dataset = coco_target_dict  # type: ignore
            coco_target.createIndex()

            if len(self.pred_results) == 0:
                return {"AP": 0.0, "AP50": 0.0}

            coco_pred = coco_target.loadRes(self.pred_results)  # type: ignore
            coco_eval = COCOeval(coco_target, coco_pred, iouType=self.iou_type)  # type: ignore
            if self.iou_type == "keypoints" and self.num_keypoints is not None:
                default_num_sigmas = len(coco_eval.params.kpt_oks_sigmas)
                if self.kpt_oks_sigmas is not None:
                    if len(self.kpt_oks_sigmas) != self.num_keypoints:
                        raise ValueError(
                            "Keypoint OKS sigma count must match the number of "
                            f"keypoints. Expected {self.num_keypoints}, got "
                            f"{len(self.kpt_oks_sigmas)}."
                        )
                    coco_eval.params.kpt_oks_sigmas = self.kpt_oks_sigmas
                elif self.num_keypoints != default_num_sigmas:
                    raise ValueError(
                        "pycocotools defaults to 17 COCO person keypoints for "
                        "OKS evaluation, but this dataset uses "
                        f"{self.num_keypoints}. Set "
                        "`metrics.params.kpt_oks_sigmas` to a list with one "
                        "sigma per keypoint."
                    )
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        s = coco_eval.stats
        return {"AP": float(s[0]), "AP50": float(s[1])}
