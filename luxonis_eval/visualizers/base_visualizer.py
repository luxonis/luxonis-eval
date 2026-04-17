from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
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

    def __init__(self, *, save_dir: Path | None = None, **kwargs: Any) -> None:
        """Initialize the visualizer.

        Parameters
        ----------
        save_dir : Path | None, optional
            Directory to save visualization frames to.
        **kwargs : Any
            Visualizer basic configuration.
        """
        self.save_dir = save_dir
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

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

    def _render(
        self,
        frame: np.ndarray,
        filename: str,
        window_title: str = "Visualization",
    ) -> None:
        """Display a frame interactively or save it to disk.

        Parameters
        ----------
        frame : np.ndarray
            Image to display or save.
        filename : str
            Output filename used when saving to disk.
        window_title : str, optional
            Window title used when displaying interactively.
        """
        if self.save_dir is not None:
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(self.save_dir / filename), frame)
        else:
            cv2.imshow(window_title, frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
