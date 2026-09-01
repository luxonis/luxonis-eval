from abc import ABC, abstractmethod
from typing import Any

from luxonis_ml.utils.registry import AutoRegisterMeta

from luxonis_eval.core.context import EvalContext
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
        del kwargs
        self._context: EvalContext | None = None

    def attach_context(self, context: EvalContext) -> None:
        """Attach evaluation runtime metadata after setup."""
        self._context = context

    @property
    def context(self) -> EvalContext:
        """Return the attached evaluation context."""
        if self._context is None:
            raise RuntimeError(
                f"{type(self).__name__} is missing evaluation context. "
                "Call attach_context() during setup before parse()."
            )
        return self._context

    @abstractmethod
    def parse(self, output: EngineOutput) -> Any:
        """Parse raw backend output into predictions."""
        ...
