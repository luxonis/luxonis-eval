from abc import ABC, abstractmethod
from typing import Any

from luxonis_ml.utils.registry import AutoRegisterMeta

from luxonis_eval.engines.base_engine import ModelSpec
from luxonis_eval.engines.io import EngineOutput
from luxonis_eval.registry import PARSERS_REGISTRY


class BaseParser(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=PARSERS_REGISTRY,
    register=False,
):
    """Base class for model output parsers."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the parser.

        Parameters
        ---------
        **kwargs : Any
            Parser basic configuration.
        """

    @abstractmethod
    def parse(
        self,
        output: EngineOutput,
        model_spec: ModelSpec,
        *,
        **kwargs: Any,
    ) -> Any:
        """Parse raw backend output into predictions."""
        ...
