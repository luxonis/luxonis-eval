from typing import Any

from luxonis_eval.tasks.base_task import BaseInferTask


class ClassificationTask(BaseInferTask):
    """Classification inference task."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the classification task."""
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
        apply_softmax = kwargs.get("apply_softmax", False)

        return self.parser.parse(raw_output, apply_softmax=apply_softmax)

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
        class_index_map = kwargs.get("class_index_map", {})

        return {"class_index_map": class_index_map}
