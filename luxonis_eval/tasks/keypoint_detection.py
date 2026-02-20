from typing import Any

from luxonis_eval.metrics.metrics_utils import yolo_norm_to_coco_xywh
from luxonis_eval.tasks.base_task import BaseInferTask


class KeypointDetectionTask(BaseInferTask):
    """Keypoint detection inference task."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the keypoint detection task."""
        super().__init__(**kwargs)

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
        return self.parser.parse(raw_output, **kwargs)

    def metric_extra_context(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Provide additional context for metric updates.

        Parameters
        ----------
        **kwargs : Any
            Additional context.

        Returns
        -------
        dict[str, Any]
            Additional metric context.
        """
        # Retrieve additional task-specific options
        width = kwargs.get("width", -1)
        height = kwargs.get("height", -1)
        class_map = kwargs.get("class_map", {})
        class_index_map = kwargs.get("class_index_map", {})

        return {
            "width": width,
            "height": height,
            "class_map": class_map,
            "category_ids": sorted(class_map.keys()),
            "class_index_map": class_index_map,
            "target_converter": yolo_norm_to_coco_xywh,
        }
