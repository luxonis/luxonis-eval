from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from luxonis_ml.utils.registry import AutoRegisterMeta

from luxonis_eval.parsers.predictions import Prediction
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

    @abstractmethod
    def required_target_keys(self) -> list[str]:
        """Return the ground-truth keys required by the metric."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the metric state."""
        ...

    @abstractmethod
    def update(
        self,
        predictions: Prediction,
        target: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> None:
        """Update the metric with predictions and ground truths.

        Parameters
        ----------
        predictions : Prediction
            Structured model predictions.
        target : dict[str, np.ndarray]
            Ground-truth data.
        **kwargs : Any
            Additional context.
        """
        ...

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Compute final metric values.

        Returns
        -------
        dict[str, float]
            Computed metric results.
        """
        ...
