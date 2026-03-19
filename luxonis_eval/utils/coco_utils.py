import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

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

    def __init__(self, *, iou_type: str) -> None:
        """Initialize the COCOStore.

        Parameters
        ----------
        iou_type : str
            Type of IoU to evaluate ('bbox', 'segm', etc.).
        """
        self.iou_type = iou_type
        self.images: list[dict[str, Any]] = []
        self.gt_annotations: list[dict[str, Any]] = []
        self.dt_results: list[dict[str, Any]] = []
        self.categories: list[dict[str, Any]] | None = None
        self.category_ids_set: set[int] | None = None
        self.next_ann_id: int = 1
        self.next_img_id: int = 1

    def reset(self) -> None:
        """Reset internal storage."""
        self.images.clear()
        self.gt_annotations.clear()
        self.dt_results.clear()
        self.categories = None
        self.category_ids_set = None
        self.next_ann_id = 1
        self.next_img_id = 1

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
        self.gt_annotations.append(ann)
        self.next_ann_id = max(self.next_ann_id, int(ann["id"]) + 1)

    def add_dt(self, res: dict[str, Any]) -> None:
        """Add a detection result.

        Parameters
        ----------
        res : dict[str, Any]
            Detection result in COCO format.
        """
        self.dt_results.append(res)

    def evaluate(self) -> dict[str, float]:
        """Evaluate detection results using COCO evaluation.

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

            if len(self.dt_results) == 0:
                return {"AP": 0.0, "AP50": 0.0}

            coco_dt = coco_target.loadRes(self.dt_results)  # type: ignore
            coco_eval = COCOeval(coco_target, coco_dt, iouType=self.iou_type)  # type: ignore
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        s = coco_eval.stats
        return {"AP": float(s[0]), "AP50": float(s[1])}
