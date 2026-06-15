from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from luxonis_ml.utils.registry import AutoRegisterMeta

from luxonis_eval.registry import METRICS_REGISTRY


class BaseMetric(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=METRICS_REGISTRY,
    register=False,
):
    """Base class for evaluation metrics."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the metric.

        Parameters
        ----------
        **kwargs : Any
            Metric basic configuration.
        """
        self.reset()

    def reset(self) -> None:
        """Reset the metric state."""
        self._reset_impl()

    def update(
        self, predictions: Any, target: dict[str, np.ndarray], **kwargs: Any
    ) -> None:
        """Update the metric with predictions and ground truths.

        Parameters
        ----------
        predictions : Any
            Model predictions.
        target : dict[str, np.ndarray]
            Ground-truth data.
        **kwargs : Any
            Additional context.
        """
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

    @abstractmethod
    def required_target_keys(self) -> list[str]:
        """Return the ground-truth keys required by the metric."""
        ...

    @abstractmethod
    def _reset_impl(self) -> None:
        """Reset internal metric state."""
        ...

    @abstractmethod
    def _update_impl(
        self, predictions: Any, target: dict[str, np.ndarray], **kwargs: Any
    ) -> None:
        """Update internal metric state."""
        ...

    @abstractmethod
    def _compute_impl(self) -> dict[str, float]:
        """Compute metric results."""
        ...
