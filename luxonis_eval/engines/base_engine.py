from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from luxonis_ml.utils.registry import AutoRegisterMeta

from luxonis_eval.registry import ENGINES_REGISTRY


class BaseEngine(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=ENGINES_REGISTRY,
    register=False,
):
    """Abstract base class for inference engines."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the engine.

        Parameters
        ---------
        **kwargs : Any
            Engine basic configuration.
        """

    @abstractmethod
    def setup(self) -> None:
        """Initialize backend resources."""
        ...

    @abstractmethod
    def infer_once(self, img: np.ndarray) -> Any:
        """Run inference on a single image."""
        ...

    @abstractmethod
    def vis_frame(self) -> np.ndarray:
        """Return a visualization frame."""
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Release backend resources."""
        ...
