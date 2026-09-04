from collections.abc import Mapping
from typing import Any

import depthai as dai
import numpy as np
import torch
from torch import Tensor
from torchvision.utils import draw_bounding_boxes

from .base_visualizer import BaseVisualizer, VisualizationData
from .utils import (
    Color,
    combine_visualizations,
    convert_visualization_data,
    draw_bounding_box_targets,
    get_color,
    get_prediction_labels,
    numpy_to_batched_canvas,
)


class BBoxVisualizer(BaseVisualizer):

    def __init__(
        self,
        labels: dict[int, str] | list[str] | None = None,
        draw_labels: bool = True,
        draw_scores: bool = False,
        colors: dict[str, Color] | list[Color] | None = None,
        fill: bool = False,
        width: int | None = None,
        font: str | None = None,
        font_size: int | None = None,
        scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize bounding-box drawing options."""
        if scale <= 0:
            raise ValueError("scale must be greater than zero.")
        if width is not None and width <= 0:
            raise ValueError("width must be greater than zero.")
        if font_size is not None and font_size <= 0:
            raise ValueError("font_size must be greater than zero.")

        self._configured_labels = (
            dict(enumerate(labels)) if isinstance(labels, list) else labels
        )
        self._configured_colors = colors
        self.draw_labels = draw_labels
        self.draw_scores = draw_scores
        self.fill = fill
        self.width = width
        self.font = font
        self.font_size = font_size
        self.scale = scale
        super().__init__(**kwargs)

    def required_target_keys(self) -> list[str]:
        return ["/boundingbox"]

    def convert(
        self,
        predictions: dai.ImgDetections,
        target: dict[str, np.ndarray],
    ) -> VisualizationData:
        """Convert detections and normalized targets into drawing tensors."""
        if not isinstance(predictions, dai.ImgDetections):
            raise TypeError(
                "BBoxVisualizer expects predictions of type "
                f"dai.ImgDetections, got {type(predictions).__name__}."
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
        """Render one prepared target and prediction pair."""
        canvas = numpy_to_batched_canvas(vis_frame)
        prediction_canvas = self.scale_canvas(canvas, self.scale)
        target_canvas = self.scale_canvas(canvas, self.scale)

        visualization = self.forward(
            prediction_canvas,
            target_canvas,
            data.predictions["boundingbox"],
            data.targets["boundingbox"],
        )
        self._render(
            combine_visualizations(visualization),
            filename_prefix="bbox",
            window_title="Bounding Box Visualization",
        )

    def draw_targets(self, canvas: Tensor, targets: Tensor) -> Tensor:
        viz = torch.zeros_like(canvas)
        labels = self.label_dict
        colors = self.color_dict

        for index in range(len(canvas)):
            target = targets[targets[:, 0] == index]
            if target.shape[0] == 0:
                viz[index] = canvas[index]
                continue
            target_classes = target[:, 1].int()
            class_labels = (
                [labels.get(int(class_id), str(int(class_id))) for class_id in target_classes]
                if self.draw_labels
                else None
            )
            class_colors = [
                colors.get(int(class_id), get_color(int(class_id)))
                for class_id in target_classes
            ]

            height, width = canvas.shape[-2:]
            line_width = self.width or max(1, int(min(height, width) / 100))
            viz[index] = draw_bounding_box_targets(
                canvas[index].clone(),
                target[:, 2:],
                width=line_width,
                labels=class_labels,
                colors=class_colors,
                fill=self.fill,
                font=self.font,
                font_size=self.font_size,
            )
        return viz

    def draw_predictions(
        self, canvas: Tensor, predictions: list[Tensor], scale: float = 1.0
    ) -> Tensor:
        viz = torch.zeros_like(canvas)
        labels = self.label_dict
        colors = self.color_dict

        for index in range(len(canvas)):
            prediction = predictions[index]
            if prediction.shape[0] == 0:
                viz[index] = canvas[index]
                continue
            boxes = prediction[:, :4].clone() * scale
            prediction_classes = prediction[:, 5].int()
            class_labels = get_prediction_labels(
                prediction,
                labels,
                draw_labels=self.draw_labels,
                draw_scores=self.draw_scores,
            )
            class_colors = [
                colors.get(int(class_id), get_color(int(class_id)))
                for class_id in prediction_classes
            ]

            height, width = canvas.shape[-2:]
            line_width = self.width or max(1, int(min(height, width) / 100))
            viz[index] = draw_bounding_boxes(
                canvas[index].clone(),
                boxes,
                width=line_width,
                labels=class_labels,
                colors=class_colors,
                fill=self.fill,
                font=self.font,
                font_size=self.font_size,
            )
        return viz

    def forward(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        predictions: list[Tensor],
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Draw prepared bounding-box targets and predictions."""
        predictions_viz = self.draw_predictions(
            prediction_canvas, predictions, scale=self.scale
        )
        targets_viz = self.draw_targets(target_canvas, targets)
        return targets_viz, predictions_viz

    @property
    def label_dict(self) -> Mapping[int, str]:
        return self._configured_labels or self.context.class_map

    @property
    def color_dict(self) -> dict[int, Color]:
        labels = self.label_dict
        configured = self._configured_colors
        if configured is None:
            return {class_id: get_color(class_id) for class_id in labels}
        if isinstance(configured, list):
            return {
                class_id: self._normalize_color(color)
                for class_id, color in zip(
                    sorted(labels), configured, strict=False
                )
            }

        by_name = {
            name: self._normalize_color(color)
            for name, color in configured.items()
        }
        return {
            class_id: by_name.get(name, get_color(class_id))
            for class_id, name in labels.items()
        }

    @staticmethod
    def _normalize_color(color: Color) -> Color:
        if isinstance(color, list):
            return tuple(color)  # type: ignore[return-value]
        return color
