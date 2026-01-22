from typing import Any

from luxonis_eval.metrics.classification import ClassificationMetric
from luxonis_eval.parsers.base_parser import BaseParser
from luxonis_eval.tasks.base_task import BaseInferTask


class ClassificationTask(BaseInferTask):
    """Classification inference task."""

    NAME = "classification"

    def __init__(
        self,
        *,
        topk: tuple[int, int] = (1, 5),
    ):
        """Initialize the classification task.

        Parameters
        ----------
        topk : tuple[int, int], default=(1, 5)
            Top-k values used for evaluation.
        """
        self.topk = topk

    def target_key(self) -> str:
        """Return the ground-truth key.

        Returns
        -------
        str
            Ground-truth key path.
        """
        return "/classification"

    def build_metric(self, **kwargs: Any) -> Any:
        """Create the classification metric.

        Parameters
        ----------
        **kwargs : Any
            Metric configuration.

        Returns
        -------
        Any
            Classification metric instance.
        """
        return ClassificationMetric(**kwargs)

    def parse_predictions(
        self,
        raw_output: Any,
        backend: str,
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
        pcls = BaseParser.select(task=self.NAME, backend=backend)
        parser = pcls()

        # Retrieve additional task-specific options
        apply_softmax = kwargs.get("apply_softmax", False)

        return parser.parse(raw_output, apply_softmax=apply_softmax)

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
        class_index_map = kwargs.get("class_index_map", {})

        return (
            predictions,
            target,
            {"class_index_map": class_index_map, "topk": self.topk},
        )
