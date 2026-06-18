from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from luxonis_ml.typing import PathType
from luxonis_ml.utils.registry import AutoRegisterMeta

from luxonis_eval.registry import ENGINES_REGISTRY


@dataclass(frozen=True)
class ModelSpec:
    """Validated model input specification."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "ModelSpec width and height must be positive integers."
            )


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
        self.model_spec: ModelSpec | None = None

    @abstractmethod
    def setup(self) -> ModelSpec:
        """Allocate backend resources and return the model input
        spec."""
        ...

    def _set_model_spec(self, model_spec: ModelSpec) -> ModelSpec:
        """Store and return the validated model spec."""
        self.model_spec = model_spec
        return model_spec

    def _get_model_spec(self) -> ModelSpec:
        """Return the configured model spec or fail if setup was
        skipped."""
        if self.model_spec is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.setup() must be called first."
            )
        return self.model_spec

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
