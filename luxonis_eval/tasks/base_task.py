from abc import ABC, abstractmethod
from typing import Any

from luxonis_ml.utils.registry import AutoRegisterMeta, Registry

TASKS_REGISTRY: Registry[type["BaseInferTask"]] = Registry(name="infer_tasks")


class BaseInferTask(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=TASKS_REGISTRY,
    register=False,
):
    """Base class for inference tasks."""

    NAME: str = "base"

    @abstractmethod
    def target_key(self) -> str:
        """Return the ground-truth key for a sample."""
        ...

    @abstractmethod
    def build_metric(self, **kwargs: Any) -> Any:
        """Create the metric instance."""
        ...

    @abstractmethod
    def parse_predictions(
        self,
        raw_output: Any,
        backend: str,
        **kwargs: Any,
    ) -> Any:
        """Convert raw backend output to predictions."""
        ...

    @abstractmethod
    def metric_update_payload(
        self,
        predictions: Any,
        target: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Prepare metric update inputs."""
        ...
