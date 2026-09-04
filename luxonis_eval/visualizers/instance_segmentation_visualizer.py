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
    draw_segmentation_masks,
    get_color,
    get_prediction_labels,
    numpy_to_batched_canvas,
    scale_masks,
)


class InstanceSegmentationVisualizer(BaseVisualizer):
    """Render instance-mask ground truth and predictions side by side."""

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
        alpha: float = 0.6,
        scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if scale <= 0:
            raise ValueError("scale must be greater than zero.")
        if width is not None and width <= 0:
            raise ValueError("width must be greater than zero.")
        if font_size is not None and font_size <= 0:
            raise ValueError("font_size must be greater than zero.")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between zero and one.")

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
        self.alpha = alpha
        self.scale = scale
        super().__init__(**kwargs)

    @property
    def required_target_keys(self) -> list[str]:
        return ["/boundingbox", "/instance_segmentation"]

    def convert(
        self,
        predictions: dai.ImgDetections,
        target: dict[str, np.ndarray],
    ) -> VisualizationData:
        """Convert detections, masks, and targets into drawing tensors."""
        if not isinstance(predictions, dai.ImgDetections):
            raise TypeError(
                "InstanceSegmentationVisualizer expects predictions of type "
                f"dai.ImgDetections, got {type(predictions).__name__}."
            )

        return convert_visualization_data(
            predictions,
            target,
            self.context,
            self.required_target_keys,
        )

    def visualize(
        self,
        data: VisualizationData,
        vis_frame: np.ndarray,
    ) -> None:
        """Render one prepared instance-segmentation comparison."""
        canvas = numpy_to_batched_canvas(vis_frame)
        prediction_canvas = self.scale_canvas(canvas, self.scale)
        target_canvas = self.scale_canvas(canvas, self.scale)
        visualization = self.forward(
            prediction_canvas,
            target_canvas,
            data.predictions["boundingbox"],
            data.predictions["instance_segmentation"],
            data.targets["boundingbox"],
            data.targets["instance_segmentation"],
        )
        self._render(
            combine_visualizations(visualization),
            filename_prefix="instance_segmentation",
            window_title="Instance Segmentation Visualization",
        )

    def draw_predictions(
        self,
        canvas: Tensor,
        boundingbox: list[Tensor],
        instance_segmentation: list[Tensor],
    ) -> Tensor:
        viz = torch.zeros_like(canvas)
        labels = self.label_dict
        colors = self.color_dict
        for index in range(len(canvas)):
            image_boxes = boundingbox[index]
            image_masks = scale_masks(
                instance_segmentation[index], self.scale
            )
            image = canvas[index].clone()
            prediction_classes = image_boxes[:, 5].int()
            class_colors = [
                colors.get(int(class_id), get_color(int(class_id)))
                for class_id in prediction_classes
            ]
            image = draw_segmentation_masks(
                image,
                image_masks,
                alpha=self.alpha,
                colors=class_colors,
            )
            if image_boxes.shape[0] == 0:
                viz[index] = image
                continue

            boxes = image_boxes[:, :4].clone() * self.scale
            height, width = canvas.shape[-2:]
            line_width = self.width or max(1, int(min(height, width) / 100))
            viz[index] = draw_bounding_boxes(
                image,
                boxes,
                width=line_width,
                labels=get_prediction_labels(
                    image_boxes,
                    labels,
                    draw_labels=self.draw_labels,
                    draw_scores=self.draw_scores,
                ),
                colors=class_colors,
                fill=self.fill,
                font=self.font,
                font_size=self.font_size,
            )
        return viz

    def draw_targets(
        self,
        canvas: Tensor,
        target_boundingbox: Tensor,
        target_instance_segmentation: Tensor,
    ) -> Tensor:
        viz = torch.zeros_like(canvas)
        labels = self.label_dict
        colors = self.color_dict
        for index in range(len(canvas)):
            keep = target_boundingbox[:, 0] == index
            image_boxes = target_boundingbox[keep]
            image_masks = scale_masks(
                target_instance_segmentation[keep], self.scale
            )
            target_classes = image_boxes[:, 1].int()
            class_labels = (
                [
                    labels.get(int(class_id), str(int(class_id)))
                    for class_id in target_classes
                ]
                if self.draw_labels
                else None
            )
            class_colors = [
                colors.get(int(class_id), get_color(int(class_id)))
                for class_id in target_classes
            ]
            image = draw_segmentation_masks(
                canvas[index].clone(),
                image_masks,
                alpha=self.alpha,
                colors=class_colors,
            )
            height, width = canvas.shape[-2:]
            line_width = self.width or max(1, int(min(height, width) / 100))
            viz[index] = draw_bounding_box_targets(
                image,
                image_boxes[:, 2:],
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
        boundingbox: list[Tensor],
        instance_segmentation: list[Tensor],
        target_boundingbox: Tensor,
        target_instance_segmentation: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Draw prepared instance masks and their bounding boxes."""
        predictions_viz = self.draw_predictions(
            prediction_canvas,
            boundingbox,
            instance_segmentation,
        )
        targets_viz = self.draw_targets(
            target_canvas,
            target_boundingbox,
            target_instance_segmentation,
        )
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
