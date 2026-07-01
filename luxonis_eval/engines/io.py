from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import depthai as dai
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
    """Parser-facing abstraction over backend-specific inference outputs.

    Implemented engines wrap their native output type behind this interface
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
        """Return one named tensor as a NumPy array.
        """

    @abstractmethod
    def select(self, names: Sequence[str] | None) -> "EngineOutput":
        """Return a filtered view over this output.
        """

    def first(self) -> tuple[str, np.ndarray]:
        """Convenience helper for single-output models."""
        names = self.names()
        if len(names) != 1:
            raise ValueError(
                f"Expected exactly one output tensor, got {list(names)}."
            )
        name = names[0]
        return name, self.get(name)


@dataclass(frozen=True, slots=True)
class NumpyEngineOutput(EngineOutput):
    tensors: dict[str, np.ndarray]

    def names(self) -> tuple[str, ...]:
        return tuple(self.tensors.keys())

    def get(
        self,
        name: str,
        *,
        layout: TensorLayout | None = None,
    ) -> np.ndarray:
        del layout
        try:
            return self.tensors[name]
        except KeyError as err:
            raise ValueError(
                f"Requested output tensor {name!r} is not available. "
                f"Available tensors: {list(self.tensors)}."
            ) from err

    def select(self, names: Sequence[str] | None) -> "NumpyEngineOutput":
        if names is None:
            return self

        missing = [name for name in names if name not in self.tensors]
        if missing:
            raise ValueError(
                f"Requested outputs are missing from engine result: {missing}"
            )

        return NumpyEngineOutput(
            tensors={name: self.tensors[name] for name in names}
        )


@dataclass(frozen=True, slots=True)
class DepthAIEngineOutput(EngineOutput):
    raw_output: dai.NNData
    _selected_names: tuple[str, ...] | None = None

    def names(self) -> tuple[str, ...]:
        available = tuple(self.raw_output.getAllLayerNames())
        if self._selected_names is None:
            return available

        missing = [name for name in self._selected_names if name not in available]
        if missing:
            raise ValueError(
                f"Requested outputs are missing from engine result: {missing}"
            )
        return self._selected_names

    def get(
        self,
        name: str,
        *,
        layout: TensorLayout | None = None,
    ) -> np.ndarray:
        if name not in self.names():
            raise ValueError(
                f"Requested output tensor {name!r} is not available. "
                f"Available tensors: {list(self.names())}."
            )

        get_tensor_kwargs: dict[str, object] = {"dequantize": True}
        if layout == "NCHW":
            get_tensor_kwargs["storageOrder"] = (
                dai.TensorInfo.StorageOrder.NCHW
            )
        elif layout == "NHWC":
            get_tensor_kwargs["storageOrder"] = (
                dai.TensorInfo.StorageOrder.NHWC
            )

        tensor = self.raw_output.getTensor(name, **get_tensor_kwargs)
        return np.asarray(tensor)

    def select(self, names: Sequence[str] | None) -> "DepthAIEngineOutput":
        if names is None:
            return self

        requested_names = tuple(names)
        missing = [name for name in requested_names if name not in self.names()]
        if missing:
            raise ValueError(
                f"Requested outputs are missing from engine result: {missing}"
            )
        return DepthAIEngineOutput(self.raw_output, requested_names)
