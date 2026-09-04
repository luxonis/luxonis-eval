from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from luxonis_ml.utils.registry import AutoRegisterMeta
from torch import Tensor
from torchvision.io import write_png
from torchvision.transforms.functional import resize

from luxonis_eval.core.context import EvalContext
from luxonis_eval.registry import VISUALIZERS_REGISTRY


@dataclass(frozen=True, slots=True)
class VisualizationData:
    """Task-keyed tensors shared by every visualizer.

    Prediction values are lists because postprocessed batches can contain a
    variable number of predictions per image. Target tensors retain the batch
    representation expected by the drawing functions.
    """

    predictions: dict[str, list[Tensor]]
    targets: dict[str, Tensor]


class BaseVisualizer(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=VISUALIZERS_REGISTRY,
    register=False,
):
    """Base class for evaluation visualizers."""

    def __init__(
        self,
        display: bool = False,
        save: bool = True,
        save_dir: str | Path = "visualizations",
        **kwargs: Any,
    ) -> None:
        """Initialize the visualizer.

        Parameters
        ----------
        **kwargs : Any
            Visualizer basic configuration.
        """
        del kwargs
        if not display and not save:
            raise ValueError(
                "At least one of 'display' or 'save' must be enabled."
            )

        self.display = display
        self.save = save
        self.save_dir = Path(save_dir)
        self._output_index = 0
        self._display_enabled = display
        self._window_title: str | None = None
        self._context: EvalContext | None = None

    def attach_context(self, context: EvalContext) -> None:
        """Attach evaluation runtime metadata after setup."""
        self._context = context

    @property
    def context(self) -> EvalContext:
        """Return the attached evaluation context."""
        if self._context is None:
            raise RuntimeError(
                f"{type(self).__name__} is missing evaluation context. "
                "Call attach_context() during setup before run()."
            )
        return self._context

    def run(
        self,
        predictions: Any,
        target: dict[str, np.ndarray],
        vis_frame: np.ndarray,
    ) -> None:
        """Convert one evaluation result and render its visualization."""
        self.visualize(self.convert(predictions, target), vis_frame)

    @property
    @abstractmethod
    def required_target_keys(self) -> list[str]:
        """Return ground-truth keys required by the visualizer."""
        ...

    def reset(self) -> None:
        """Reset output numbering for a new evaluation."""
        self._output_index = 0
        self._display_enabled = self.display

    def close(self) -> None:
        """Close the visualizer's display window, if one is open."""
        if self._window_title is None:
            return
        with suppress(cv2.error):
            cv2.destroyWindow(self._window_title)
        self._window_title = None

    @staticmethod
    def scale_canvas(canvas: Tensor, scale: float = 1.0) -> Tensor:
        """Resize a BCHW visualization canvas."""
        if scale == 1.0:
            return canvas
        height = max(1, round(canvas.shape[-2] * scale))
        width = max(1, round(canvas.shape[-1] * scale))
        return resize(canvas, [height, width], antialias=True)

    @abstractmethod
    def convert(
        self,
        predictions: Any,
        target: dict[str, np.ndarray],
    ) -> VisualizationData:
        """Convert runtime results into canonical visualization data."""
        ...

    @abstractmethod
    def visualize(
        self,
        data: VisualizationData,
        vis_frame: np.ndarray,
    ) -> None:
        """Render canonical visualization data on an image."""
        ...

    def _render(
        self,
        visualization: Tensor,
        *,
        filename_prefix: str,
        window_title: str,
    ) -> None:
        image = visualization.detach().cpu().to(torch.uint8)
        if image.ndim != 3:
            raise ValueError(
                "Rendered visualization must use CHW layout, got "
                f"shape {tuple(image.shape)}."
            )

        if self.save:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            output_path = self._next_output_path(filename_prefix)
            write_png(image, str(output_path))
        if self.display:
            self._display(image, window_title)

    def _display(self, image: Tensor, window_title: str) -> None:
        if not self._display_enabled:
            return

        display_image = image.permute(1, 2, 0).contiguous().numpy()
        display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

        if self._window_title is None:
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            self._window_title = window_title

        cv2.resizeWindow(
            window_title,
            width=display_image.shape[1],
            height=display_image.shape[0],
        )
        cv2.imshow(window_title, display_image)
        if cv2.waitKey(0) & 0xFF in {27, ord("q")}:
            self._display_enabled = False
            self.close()

    def _next_output_path(self, filename_prefix: str) -> Path:
        output_path = (
            self.save_dir / f"{filename_prefix}_{self._output_index:05d}.png"
        )
        self._output_index += 1
        return output_path
