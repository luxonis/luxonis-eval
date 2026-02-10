from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEngine(ABC):
    """Abstract base class for inference engines."""

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
