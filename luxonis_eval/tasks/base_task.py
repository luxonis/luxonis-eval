from abc import ABC, abstractmethod
from typing import Any

from loguru import logger
from luxonis_ml.utils.registry import AutoRegisterMeta

from luxonis_eval.metrics import ThroughputMetric
from luxonis_eval.registry import (
    METRICS_REGISTRY,
    PARSERS_REGISTRY,
    TASKS_REGISTRY,
    from_registry,
)


class BaseInferTask(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=TASKS_REGISTRY,
    register=False,
):
    """Base class for inference tasks."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the inference task.

        Parameters
        ---------
        **kwargs : Any
            Inference task basic configuration.
        """

    def build_metrics(self, **kwargs: Any) -> None:
        """Create the metrics instance.

        Parameters
        ----------
        **kwargs : Any
            Metrics configuration.

        Returns
        -------
        Any
            Metrics instance.
        """
        metrics_list = kwargs.get("metrics")
        if not metrics_list:
            raise ValueError(
                "Metric configuration must include a 'metrics' key with a list of metrics."
            )
        self.metrics = []
        for metric_cfg in metrics_list:
            metric_name = metric_cfg.get("name")
            if not metric_name:
                raise ValueError(
                    "Each metric configuration must include a 'name' key."
                )
            try:
                metric_instance = from_registry(
                    METRICS_REGISTRY,
                    metric_name,
                    **metric_cfg.get("params", {}),
                )
                logger.info(f"{metric_name} metric initialized.")
            except KeyError as e:
                raise ValueError(
                    f"Unknown metric: {metric_name}. "
                    f"Available metrics: {list(METRICS_REGISTRY._module_dict)}"
                ) from e
            self.metrics.append(metric_instance)

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
        parser_name = kwargs.get("name")
        if not parser_name:
            raise ValueError("Parser configuration must include a 'name' key.")
        try:
            self.parser = from_registry(
                PARSERS_REGISTRY, parser_name, **kwargs.get("params", {})
            )
            logger.info(f"{parser_name} parser initialized.")
        except KeyError as e:
            raise ValueError(
                f"Unknown parser: {parser_name}. "
                f"Available parsers: {list(PARSERS_REGISTRY._module_dict)}"
            ) from e

    @abstractmethod
    def parse_predictions(
        self,
        raw_output: Any,
        **kwargs: Any,
    ) -> Any:
        """Convert raw backend output to predictions."""
        ...

    @abstractmethod
    def metric_extra_context(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Provide additional context for metric updates."""
        ...
