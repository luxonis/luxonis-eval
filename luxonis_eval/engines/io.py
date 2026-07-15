from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

TensorLayout = Literal["NCHW", "NHWC", "NC", "CHW", "HWC", "HW"]


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    layout: TensorLayout | None = None


@dataclass(frozen=True, slots=True)
class ModelSpec:
    input: TensorSpec
    outputs: tuple[TensorSpec, ...]

    def __post_init__(self) -> None:
        if self.input.shape is None or len(self.input.shape) != 4:
            raise ValueError(
                "ModelSpec input shape must be a statically defined 4D tensor."
            )
        if self.input.layout not in {"NCHW", "NHWC"}:
            raise ValueError(
                "ModelSpec input layout must be either 'NCHW' or 'NHWC'."
            )
        if not self.outputs:
            raise ValueError("ModelSpec must define at least one output.")

    @property
    def width(self) -> int:
        width_idx = 3 if self.input.layout == "NCHW" else 2
        width = self.input.shape[width_idx]
        if not isinstance(width, int) or width <= 0:
            raise ValueError(
                f"ModelSpec width must be a positive integer, got {width!r}."
            )
        return width

    @property
    def height(self) -> int:
        height_idx = 2 if self.input.layout == "NCHW" else 1
        height = self.input.shape[height_idx]
        if not isinstance(height, int) or height <= 0:
            raise ValueError(
                f"ModelSpec height must be a positive integer, got {height!r}."
            )
        return height


class EngineOutput(ABC):
    """Parser-facing abstraction over backend-specific inference
    outputs.

    Implemented engines wrap their native output type behind this
    interface
    """

    @abstractmethod
    def names(self) -> tuple[str, ...]:
        """Return output tensor names in engine-defined order."""

    @abstractmethod
    def get(
        self,
        name: str,
        *,
        layout: TensorLayout | None = None,
    ) -> np.ndarray:
        """Return one named tensor as a NumPy array."""

    @abstractmethod
    def select(self, names: Sequence[str] | None) -> EngineOutput:
        """Return a filtered view over this output."""

    def first(self) -> tuple[str, np.ndarray]:
        """Convenience helper for single-output models."""
        names = self.names()
        if len(names) != 1:
            raise ValueError(
                f"Expected exactly one output tensor, got {list(names)}."
            )
        name = names[0]
        return name, self.get(name)
