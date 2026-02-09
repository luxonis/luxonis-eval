from abc import ABC, abstractmethod
from typing import Any

from luxonis_ml.utils.registry import AutoRegisterMeta, Registry

METRICS_REGISTRY: Registry[type["BaseMetric"]] = Registry(name="metrics")


class BaseMetric(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=METRICS_REGISTRY,
    register=False,
):
    """Base class for evaluation metrics."""

    def __init__(self) -> None:
        """Initialize the metric."""
        self.reset()

    def reset(self) -> None:
        """Reset the metric state."""
        self._reset_impl()

    def update(self, predictions: Any, target: Any, **kwargs: Any) -> None:
        """Update the metric with predictions and ground truths.

        Parameters
        ----------
        predictions : Any
            Model predictions.
        target : Any
            Ground-truth data.
        **kwargs : Any
            Additional context.
        """
        self.validate_target_keys(target)
        self._update_impl(predictions, target, **kwargs)

    def compute(self) -> dict[str, float]:
        """Compute final metric values.

        Returns
        -------
        dict[str, float]
            Computed metric results.
        """
        results = self._compute_impl()
        results["metric"] = self.__class__.__name__  # type: ignore
        return results

    def as_log(self) -> dict[str, Any]:
        """Return metric results formatted for logging.

        Returns
        -------
        dict[str, Any]
            Metric results and throughput information.
        """
        return {
            "metric": self.__class__.__name__,
            **self.compute(),
        }

    def validate_target_keys(self, target: Any) -> None:
        """Validate that the target contains the required keys for the metric.

        Parameters
        ----------
        target : Any
            Ground-truth data.
        """
        metric_keys = set(self.metric_keys())
        target_keys = set(target)
        if not metric_keys.issubset(target_keys):
            raise ValueError(
                f"Target is missing required keys for {self.__class__.__name__}. "
                f"Expected at least: {self.metric_keys()}, but got: {list(target.keys())}."
            )

    @abstractmethod
    def metric_keys(self) -> list[str]:
        """Return the ground-truth keys required by the metric."""
        ...

    @abstractmethod
    def _reset_impl(self) -> None:
        """Reset internal metric state."""
        ...

    @abstractmethod
    def _update_impl(
        self, predictions: Any, target: Any, **kwargs: Any
    ) -> None:
        """Update internal metric state."""
        ...

    @abstractmethod
    def _compute_impl(self) -> dict[str, float]:
        """Compute metric results."""
        ...
