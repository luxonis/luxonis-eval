from typing import Any

import depthai as dai
import numpy as np
import torch
from loguru import logger
from torch import Tensor

from .base_visualizer import BaseVisualizer, VisualizationData
from .utils import (
    Color,
    combine_visualizations,
    convert_visualization_data,
    draw_segmentation_masks,
    get_color,
    numpy_to_batched_canvas,
    scale_masks,
)


class SegmentationVisualizer(BaseVisualizer):
    """Render semantic-segmentation targets and predictions side by side."""

    def __init__(
        self,
        colors: Color | list[Color] | None = None,
        background_class: int | None = 0,
        background_color: Color = "#000000",
        alpha: float = 0.6,
        scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize semantic-segmentation drawing options."""
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between zero and one.")
        if scale <= 0:
            raise ValueError("scale must be greater than zero.")
        normalized_colors: list[Color] | None
        if colors is None:
            normalized_colors = None
        elif (
            isinstance(colors, list)
            and len(colors) == 3
            and all(isinstance(channel, int) for channel in colors)
        ):
            normalized_colors = [self._normalize_color(colors)]
        elif isinstance(colors, list):
            normalized_colors = [
                self._normalize_color(color) for color in colors
            ]
        else:
            normalized_colors = [self._normalize_color(colors)]

        self.colors = normalized_colors
        self.background_class = background_class
        self.background_color = self._normalize_color(background_color)
        self.alpha = alpha
        self.scale = scale
        self._warn_colors = True
        super().__init__(**kwargs)

    def required_target_keys(self) -> list[str]:
        return ["/segmentation"]

    def convert(
        self,
        predictions: dai.SegmentationMask,
        target: dict[str, np.ndarray],
    ) -> VisualizationData:
        """Convert an indexed DepthAI mask and dataset mask to BCHW tensors."""
        if not isinstance(predictions, dai.SegmentationMask):
            raise TypeError(
                "SegmentationVisualizer expects predictions of type "
                f"dai.SegmentationMask, got {type(predictions).__name__}."
            )

        return convert_visualization_data(
            predictions,
            target,
            self.context,
            self.required_target_keys(),
        )

    def visualize(
        self,
        data: VisualizationData,
        vis_frame: np.ndarray,
    ) -> None:
        """Render one prepared semantic-segmentation comparison."""
        canvas = numpy_to_batched_canvas(vis_frame)
        prediction_canvas = self.scale_canvas(canvas, self.scale)
        target_canvas = self.scale_canvas(canvas, self.scale)
        visualization = self.forward(
            prediction_canvas,
            target_canvas,
            torch.stack(data.predictions["segmentation"]),
            data.targets["segmentation"],
        )
        self._render(
            combine_visualizations(visualization),
            filename_prefix="segmentation",
            window_title="Semantic Segmentation Visualization",
        )

    def draw_predictions(
        self,
        canvas: Tensor,
        predictions: Tensor,
        colors: list[Color],
    ) -> Tensor:
        return self._draw_masks(canvas, predictions, colors)

    def draw_targets(
        self,
        canvas: Tensor,
        targets: Tensor,
        colors: list[Color],
    ) -> Tensor:
        return self._draw_masks(canvas, targets, colors)

    def forward(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        predictions: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Draw prepared semantic-segmentation targets and predictions."""
        if predictions.shape != targets.shape:
            raise ValueError(
                "Prediction and target segmentation tensors must have the "
                f"same shape, got {tuple(predictions.shape)} and "
                f"{tuple(targets.shape)}."
            )
        colors = self._adjust_colors(predictions.shape[1])
        predictions_viz = self.draw_predictions(
            prediction_canvas, predictions, colors
        )
        targets_viz = self.draw_targets(target_canvas, targets, colors)
        return targets_viz, predictions_viz

    def _draw_masks(
        self,
        canvas: Tensor,
        masks: Tensor,
        colors: list[Color],
    ) -> Tensor:
        if masks.ndim != 4 or masks.shape[0] != canvas.shape[0]:
            raise ValueError(
                "Segmentation masks must use BCHW layout and match the "
                "canvas batch size."
            )

        visualization = torch.zeros_like(canvas)
        for index in range(len(canvas)):
            image_masks = scale_masks(masks[index], self.scale)
            visualization[index] = draw_segmentation_masks(
                canvas[index].clone(),
                image_masks,
                alpha=self.alpha,
                colors=colors,
            )
        return visualization

    def _adjust_colors(self, n_classes: int) -> list[Color]:
        if self.colors is not None and len(self.colors) == n_classes:
            colors = list(self.colors)
        else:
            if self._warn_colors:
                if self.colors is None:
                    logger.warning(
                        "No segmentation colors provided. Using generated colors."
                    )
                else:
                    logger.warning(
                        f"Number of segmentation colors ({len(self.colors)}) "
                        f"does not match number of classes ({n_classes}). "
                        "Using generated colors."
                    )
                self._warn_colors = False
            colors = [get_color(index) for index in range(n_classes)]

        if (
            self.background_class is not None
            and n_classes > 1
            and 0 <= self.background_class < n_classes
        ):
            colors[self.background_class] = self.background_color
        return colors

    @staticmethod
    def _normalize_color(color: object) -> Color:
        if isinstance(color, str):
            return color
        if (
            isinstance(color, list | tuple)
            and len(color) == 3
            and all(isinstance(channel, int) for channel in color)
        ):
            return int(color[0]), int(color[1]), int(color[2])
        raise ValueError(
            "Colors must be color names, hex strings, or RGB triples."
        )
