from typing import Any

from luxonis_eval.metrics.base_metric import BaseMetric


class BaseTaskMetric(BaseMetric):
    """A task-level base metric that aggregates multiple primitive metrics."""

    def __init__(self, *, metrics: list[BaseMetric]) -> None:
        """Initialize the task metric.

        Parameters
        ----------
        metrics : list[BaseMetric]
            List of primitive metrics to aggregate.
        """
        self._metrics = metrics
        super().__init__()

    def _reset_impl(self) -> None:
        """Reset internal metric state."""
        for m in self._metrics:
            m.reset()

    def _update_impl(
        self, predictions: Any, target: Any, **kwargs: Any
    ) -> None:
        """Update internal metric state.

        Parameters
        ----------
        predictions : Any
            Model predictions.
        target : Any
            Ground-truth data.
        **kwargs : Any
            Additional context.
        """
        for m in self._metrics:
            m.update(predictions, target, **kwargs)

    def _compute_impl(self) -> dict[str, float]:
        """Compute final metric values.

        Returns
        -------
        dict[str, float]
            Computed metric results.
        """
        out: dict[str, float] = {}
        for m in self._metrics:
            out.update(m.compute())
        return out
