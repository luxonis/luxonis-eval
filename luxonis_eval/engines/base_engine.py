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

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        """Initialize the engine.

        Parameters
        ----------
        model_path : str
            Path to the model file.
        **kwargs : Any
            Engine basic configuration.
        """
        self.model_path = model_path
        self.width, self.height = self.get_input_shape()
        if self.width is None or self.height is None:
            raise ValueError(
                "Invalid input shape: width and height must be defined."
            )
        self.platform_name = self.get_platform_name()
        if self.platform_name is None:
            raise ValueError("Platform name must be defined.")

    @abstractmethod
    def setup(self) -> None:
        """Initialize backend resources."""
        ...

    @abstractmethod
    def get_input_shape(self) -> tuple[int, int]:
        """Get the input shape (width, height) from the loaded model."""
        ...

    @abstractmethod
    def get_platform_name(self) -> str:
        """Get the platform name."""
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
