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
        self.validate_target_keys(target)
        self._update_impl(predictions, target, **kwargs)

    @staticmethod
    def _normalize_target_key(key: str) -> str:
        return key.lstrip("/")

    def resolve_target_key(
        self, expected_key: str, target: dict[str, np.ndarray]
    ) -> str:
        """Resolve a required metric key against target labels.

        Supports both bare LDF keys like ``/boundingbox`` and task-prefixed
        keys like ``barcode-detection/boundingbox``.
        """
        if expected_key in target:
            return expected_key

        expected_suffix = self._normalize_target_key(expected_key)
        matches = [
            key
            for key in target
            if self._normalize_target_key(key) == expected_suffix
            or key.endswith(f"/{expected_suffix}")
        ]

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous target key for {self.__class__.__name__}: "
                f"expected '{expected_key}', matched {matches}."
            )

        raise ValueError(
            f"Target is missing required key '{expected_key}' for "
            f"{self.__class__.__name__}. Available keys: {list(target.keys())}."
        )

    def get_target_value(
        self, expected_key: str, target: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Return a target array resolved from an expected metric key."""
        return target[self.resolve_target_key(expected_key, target)]

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

    def validate_target_keys(self, target: dict[str, np.ndarray]) -> None:
        """Validate that the target contains the required keys for the
        metric.

        Parameters
        ----------
        target : dict[str, np.ndarray]
            Ground-truth data.
        """
        for expected_key in self.metric_keys():
            self.resolve_target_key(expected_key, target)

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
        self, predictions: Any, target: dict[str, np.ndarray], **kwargs: Any
    ) -> None:
        """Update internal metric state."""
        ...

    @abstractmethod
    def _compute_impl(self) -> dict[str, float]:
        """Compute metric results."""
        ...
