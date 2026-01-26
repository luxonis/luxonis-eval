from abc import ABC, abstractmethod
from typing import Any

from loguru import logger
from luxonis_ml.utils.registry import AutoRegisterMeta, Registry

from luxonis_eval.metrics import METRICS_REGISTRY, ThroughputMetric
from luxonis_eval.parsers import PARSERS_REGISTRY

TASKS_REGISTRY: Registry[type["BaseInferTask"]] = Registry(name="infer_tasks")


class BaseInferTask(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=TASKS_REGISTRY,
    register=False,
):
    """Base class for inference tasks."""

    def build_metric(self, **kwargs: Any) -> None:
        """Create the metric instance.

        Parameters
        ----------
        **kwargs : Any
            Metric configuration.

        Returns
        -------
        Any
            Classification metric instance.
        """
        metric_name = kwargs.pop("name", None)
        if not metric_name:
            raise ValueError("Metric configuration must include a 'name' key.")
        try:
            metric_cls = METRICS_REGISTRY[metric_name]
            logger.info(f"{metric_name} metric initialized.")
        except KeyError as e:
            raise ValueError(
                f"Unknown metric: {metric_name}. "
                f"Available metrics: {list(METRICS_REGISTRY._module_dict)}"
            ) from e
        self.metric = metric_cls(**kwargs)

    def build_throughput_metric(self) -> None:
        """Create the throughput metric instance."""
        self.throughput_metric = ThroughputMetric()
        logger.info("Throughput metric initialized.")

    def build_parser(self, **kwargs: Any) -> None:
        """Create the parser instance.

        Parameters
        ----------
        **kwargs : Any
            Parser configuration.

        Returns
        -------
        Any
            Parser instance.
        """
        parser_name = kwargs.pop("name", None)
        if not parser_name:
            raise ValueError("Parser configuration must include a 'name' key.")
        try:
            parser_cls = PARSERS_REGISTRY[parser_name]
            logger.info(f"{parser_name} parser initialized.")
        except KeyError as e:
            raise ValueError(
                f"Unknown parser: {parser_name}. "
                f"Available parsers: {list(PARSERS_REGISTRY._module_dict)}"
            ) from e
        self.parser = parser_cls(**kwargs)

    @abstractmethod
    def target_key(self) -> str:
        """Return the ground-truth key for a sample."""
        ...

    @abstractmethod
    def parse_predictions(
        self,
        raw_output: Any,
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
