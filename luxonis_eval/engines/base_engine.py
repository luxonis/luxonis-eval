import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from luxonis_ml.typing import PathType
from luxonis_ml.utils.registry import AutoRegisterMeta

from luxonis_eval.engines.io import EngineOutput, ModelSpec
from luxonis_eval.registry import ENGINES_REGISTRY


class BaseEngine(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=ENGINES_REGISTRY,
    register=False,
):
    """Abstract base class for inference engines."""

    output_type: type[EngineOutput] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        if "output_type" not in cls.__dict__ or cls.output_type is None:
            raise TypeError(
                f"{cls.__name__} must define `output_type` as its EngineOutput class."
            )
        if not issubclass(cls.output_type, EngineOutput):
            raise TypeError(
                f"{cls.__name__}.output_type must be a subclass of EngineOutput."
            )

    def __init__(self, model_path: PathType, **kwargs: Any) -> None:
        """Initialize the engine.

        Parameters
        ----------
        model_path : PathType
            Path to the model file.
        **kwargs : Any
            Engine basic configuration.
        """
        self.model_path = Path(model_path)
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
    def infer_once(self, img: np.ndarray) -> EngineOutput:
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
