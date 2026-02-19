from typing import Any

from luxonis_eval.tasks.base_task import BaseInferTask


class SemanticSegmentationTask(BaseInferTask):
    """Semantic segmentation inference task."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the semantic segmentation task."""
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
        ldf_class_map = kwargs.get("ldf_class_map", {})
        class_index_map = kwargs.get("class_index_map", {})

        ldf_name_to_idx = {v: k for k, v in ldf_class_map.items()}

        return {
            "target_bg": ldf_name_to_idx.get("background"),
            "target_class_map": ldf_class_map,
            "class_index_map": class_index_map,
        }
