from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from luxonis_ml.utils.registry import AutoRegisterMeta

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

    @abstractmethod
    def visualize(
        self,
        predictions: Any,
        target: Any,
        vis_frame: np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Visualize the evaluation results.

        Parameters
        ----------
        predictions : Any
            Model predictions.
        target : Any
            Ground truth target.
        vis_frame : np.ndarray
            Frame to visualize on.
        **kwargs : Any
            Additional visualization options.
        """
        ...
