import time
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

    NAME = "base"

    def __init__(self) -> None:
        """Initialize the metric."""
        self.reset()

    @property
    def num_updates(self) -> int:
        """Return the number of metric updates.

        Returns
        -------
        int
            Number of updates performed.
        """
        return self._num_updates

    def reset(self) -> None:
        """Reset the metric state."""
        self._num_updates = 0

        self._t0 = time.perf_counter()
        self._t_last = self._t0

        self._reset_impl()

    def step(self) -> None:
        """Increment the update counter."""
        self._num_updates += 1

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
        self.step()
        self._update_impl(predictions, target, **kwargs)

    def throughput(self) -> dict[str, float]:
        """Compute throughput statistics.

        Returns
        -------
        dict[str, float]
            Throughput and latency metrics.
        """
        elapsed = max(time.perf_counter() - self._t0, 1e-12)
        sps = self.num_updates / elapsed
        msp = (
            (elapsed / self.num_updates) * 1000.0 if self.num_updates else 0.0
        )

        return {
            "elapsed_s": float(elapsed),
            "samples": int(self.num_updates),
            "samples_per_s": float(sps),
            "ms_per_sample": float(msp),
        }

    def compute(self) -> dict[str, float]:
        """Compute final metric values.

        Returns
        -------
        dict[str, float]
            Computed metric results.
        """
        return self._compute_impl()

    def as_log(self) -> dict[str, Any]:
        """Return metric results formatted for logging.

        Returns
        -------
        dict[str, Any]
            Metric results and throughput information.
        """
        return {
            "metric": self.__class__.__name__,
            "updates": self.num_updates,
            **self.compute(),
            **self.throughput(),
        }

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
