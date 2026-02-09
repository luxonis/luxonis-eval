from typing import Any

from luxonis_eval.tasks.base_task import BaseInferTask
from luxonis_eval.utils.utils import yolo_norm_to_coco_xywh


class InstanceSegmentationTask(BaseInferTask):
    """Instance segmentation inference task."""

    def parse_predictions(
        self,
        raw_output: Any,
        **kwargs: Any,
    ) -> Any:
        """Parse backend output into predictions.

        Parameters
        ----------
        raw_output : Any
            Backend output.
        backend : str
            Backend identifier.
        **kwargs : Any
            Additional parser options.

        Returns
        -------
        Any
            Parsed predictions.
        """
        # Retrieve additional task-specific options
        native_class_map = kwargs.get("native_class_map", {})

        return self.parser.parse(raw_output, class_map=native_class_map)

    def metric_update_payload(
        self,
        predictions: Any,
        target: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Prepare metric update payload.

        Parameters
        ----------
        predictions : Any
            Model predictions.
        target : Any
            Ground-truth data.
        **kwargs : Any
            Additional context.

        Returns
        -------
        tuple[Any, Any, dict[str, Any]]
            Predictions, ground truths, and metric context.
        """
        # Retrieve additional task-specific options
        width = kwargs.get("width", -1)
        height = kwargs.get("height", -1)
        native_class_map = kwargs.get("native_class_map", {})
        class_index_map = kwargs.get("class_index_map", {})

        ctx = {
            "width": width,
            "height": height,
            "native_class_map": native_class_map,
            "category_ids": sorted(native_class_map.keys()),
            "class_index_map": class_index_map,
            "target_converter": yolo_norm_to_coco_xywh,
        }
        # TODO: fix that target is a list of boxes and masks to be consistent across the codebase for tasks that require > 1 target and tasks that require exactly 1 target
        return predictions, target, ctx
