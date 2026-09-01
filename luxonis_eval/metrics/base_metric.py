from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from luxonis_ml.utils.registry import AutoRegisterMeta

from luxonis_eval.core.context import EvalContext
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
        del kwargs
        self._context: EvalContext | None = None
        self.reset()

    def attach_context(self, context: EvalContext) -> None:
        """Attach evaluation runtime metadata after setup."""
        self._context = context

    def require_context(self) -> EvalContext:
        """Return the attached evaluation context."""
        if self._context is None:
            raise RuntimeError(
                f"{type(self).__name__} is missing evaluation context. "
                "Call attach_context() during setup before update()."
            )
        return self._context

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
        predictions: Any,
        target: dict[str, np.ndarray],
    ) -> None:
        """Update the metric with predictions and ground truths.

        Parameters
        ----------
        predictions : Any
            Model predictions.
        target : dict[str, np.ndarray]
            Ground-truth data.
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
