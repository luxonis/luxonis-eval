from typing import Any

from luxonis_eval.tasks.base_task import BaseInferTask
from luxonis_eval.utils.utils import yolo_norm_to_coco_xywh


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
        # Retrieve additional task-specific options
        native_class_map = kwargs.get("native_class_map", {})

        return self.parser.parse(raw_output, class_map=native_class_map)

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
        native_class_map = kwargs.get("native_class_map", {})
        class_index_map = kwargs.get("class_index_map", {})

        return {
            "width": width,
            "height": height,
            "native_class_map": native_class_map,
            "category_ids": sorted(native_class_map.keys()),
            "class_index_map": class_index_map,
            "target_converter": yolo_norm_to_coco_xywh,
        }
