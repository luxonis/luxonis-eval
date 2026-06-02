from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from luxonis_ml.typing import PathType
from luxonis_ml.utils.registry import AutoRegisterMeta

from luxonis_eval.registry import ENGINES_REGISTRY


class BaseEngine(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=ENGINES_REGISTRY,
    register=False,
):
    """Abstract base class for inference engines."""

    def __init__(self, model_path: PathType, **kwargs: Any) -> None:
        """Initialize the engine.

        Parameters
        ----------
        model_path : PathType
            Path to the model file.
        **kwargs : Any
            Engine basic configuration.
        """
        self.model_path = model_path
        self.width: int | None = None
        self.height: int | None = None
        self.platform_name: str | None = None

    @abstractmethod
    def setup(self) -> None:
        """Initialize backend resources."""
        ...

    def _set_runtime_metadata(self) -> None:
        """Populate and validate runtime metadata during setup.

        Runtime metadata is intentionally unset before `setup()` and
        after `close()`. Concrete engines should call this before
        `setup()` returns.
        """
        width, height = self.get_input_shape()
        if width is None or height is None:
            raise ValueError(
                "Invalid input shape: width and height must be defined."
            )

        platform_name = self.get_platform_name()
        if platform_name is None:
            raise ValueError("Platform name must be defined.")

        self.width = width
        self.height = height
        self.platform_name = platform_name

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
    def close(self) -> None:
        """Release backend resources."""
        ...
