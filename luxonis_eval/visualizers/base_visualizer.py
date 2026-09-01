from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from luxonis_ml.utils.registry import AutoRegisterMeta

from luxonis_eval.core.context import EvalContext
from luxonis_eval.registry import VISUALIZERS_REGISTRY

class BaseVisualizer(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=VISUALIZERS_REGISTRY,
    register=False,
):
    """Base class for evaluation visualizers."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the visualizer.

        Parameters
        ----------
        **kwargs : Any
            Visualizer basic configuration.
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
                "Call attach_context() during setup before visualize()."
            )
        return self._context

    @abstractmethod
    def visualize(
        self,
        predictions: Any,
        vis_frame: np.ndarray,
    ) -> None:
        """Visualize the evaluation results."""
        ...
