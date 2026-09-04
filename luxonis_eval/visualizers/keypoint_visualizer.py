from typing import Any

import depthai as dai
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torchvision.utils import draw_keypoints

from .base_visualizer import VisualizationData
from .bbox_visualizer import BBoxVisualizer
from .utils import (
    Color,
    combine_visualizations,
    convert_visualization_data,
    numpy_to_batched_canvas,
)


class KeypointVisualizer(BBoxVisualizer):
    """Render bounding boxes and instance keypoints side by side."""

    def __init__(
        self,
        visibility_threshold: float = 0.5,
        connectivity: list[tuple[int, int]] | None = None,
        visible_color: Color = "red",
        nonvisible_color: Color | None = None,
        radius: int | None = None,
        draw_indices: bool = False,
        **kwargs: Any,
    ) -> None:
        if not 0 <= visibility_threshold <= 1:
            raise ValueError("visibility_threshold must be between zero and one.")
        if radius is not None and radius <= 0:
            raise ValueError("radius must be greater than zero.")

        self.visibility_threshold = visibility_threshold
        self.connectivity = self._normalize_connectivity(connectivity)
        self.visible_color = self._normalize_color(visible_color)
        self.nonvisible_color = (
            self._normalize_color(nonvisible_color)
            if nonvisible_color is not None
            else None
        )
        self.radius = radius
        self.draw_indices = draw_indices
        self._inferred_connectivity: list[tuple[int, int]] | None = None
        super().__init__(**kwargs)

    def required_target_keys(self) -> list[str]:
        return ["/boundingbox", "/keypoints"]

    def convert(
        self,
        predictions: dai.ImgDetections,
        target: dict[str, np.ndarray],
    ) -> VisualizationData:
        """Convert DepthAI detections and normalized targets to tensors."""
        if not isinstance(predictions, dai.ImgDetections):
            raise TypeError(
                "KeypointVisualizer expects predictions of type "
                f"dai.ImgDetections, got {type(predictions).__name__}."
            )

        data = convert_visualization_data(
            predictions,
            target,
            self.context,
            self.required_target_keys(),
        )
        keypoints = data.predictions["keypoints"][0]
        connectivity = self._prediction_connectivity(predictions)
        self._validate_connectivity(connectivity, keypoints.shape[1])
        self._inferred_connectivity = connectivity

        return data

    def visualize(
        self,
        data: VisualizationData,
        vis_frame: np.ndarray,
    ) -> None:
        """Render one prepared keypoint target and prediction pair."""
        canvas = numpy_to_batched_canvas(vis_frame)
        prediction_canvas = self.scale_canvas(canvas, self.scale)
        target_canvas = self.scale_canvas(canvas, self.scale)
        connectivity = (
            self.connectivity
            if self.connectivity is not None
            else self._inferred_connectivity
        )
        visualization = self._forward_keypoints(
            prediction_canvas,
            target_canvas,
            data.predictions["keypoints"],
            data.predictions["boundingbox"],
            data.targets["keypoints"],
            data.targets["boundingbox"],
            connectivity=connectivity,
        )
        self._render(
            combine_visualizations(visualization),
            filename_prefix="keypoints",
            window_title="Keypoint Visualization",
        )

    def _forward_keypoints(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        keypoints: list[Tensor],
        boundingbox: list[Tensor],
        target_keypoints: Tensor,
        target_boundingbox: Tensor,
        *,
        connectivity: list[tuple[int, int]] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Draw prepared bounding boxes and keypoints."""
        prediction_visualization = super().draw_predictions(
            prediction_canvas,
            boundingbox,
            scale=self.scale,
        )
        target_visualization = super().draw_targets(
            target_canvas,
            target_boundingbox,
        )

        prediction_radius = self.radius or self._get_radius(
            prediction_canvas
        )
        target_radius = self.radius or self._get_radius(target_canvas)
        prediction_visualization = self.draw_keypoint_predictions(
            prediction_visualization,
            keypoints,
            connectivity=connectivity,
            radius=prediction_radius,
        )
        target_visualization = self.draw_keypoint_targets(
            target_visualization,
            target_keypoints,
            connectivity=connectivity,
            radius=target_radius,
        )
        return target_visualization, prediction_visualization

    def draw_keypoint_predictions(
        self,
        canvas: Tensor,
        predictions: list[Tensor],
        *,
        connectivity: list[tuple[int, int]] | None,
        radius: int,
    ) -> Tensor:
        visualization = torch.zeros_like(canvas)
        for index in range(len(canvas)):
            prediction = predictions[index]
            xy = prediction[..., :2].clone() * self.scale
            visibility = prediction[..., 2] >= self.visibility_threshold
            self._clip_keypoints(xy, canvas)
            image = self._draw_keypoint_set(
                canvas[index].clone(),
                xy,
                visibility,
                connectivity=connectivity,
                color=self.visible_color,
                radius=radius,
            )
            if self.nonvisible_color is not None:
                image = self._draw_keypoint_set(
                    image,
                    xy,
                    ~visibility,
                    connectivity=connectivity,
                    color=self.nonvisible_color,
                    radius=radius,
                )
            visualization[index] = image
        return visualization

    def draw_keypoint_targets(
        self,
        canvas: Tensor,
        targets: Tensor,
        *,
        connectivity: list[tuple[int, int]] | None,
        radius: int,
    ) -> Tensor:
        visualization = torch.zeros_like(canvas)
        for index in range(len(canvas)):
            target = targets[targets[:, 0] == index][:, 1:]
            if target.shape[0] == 0:
                visualization[index] = canvas[index]
                continue
            keypoints = target.reshape(target.shape[0], -1, 3)
            xy = keypoints[..., :2].clone()
            xy[..., 0] *= canvas.shape[-1]
            xy[..., 1] *= canvas.shape[-2]
            self._clip_keypoints(xy, canvas)
            visualization[index] = self._draw_keypoint_set(
                canvas[index].clone(),
                xy,
                keypoints[..., 2] > 0,
                connectivity=connectivity,
                color=self.visible_color,
                radius=radius,
            )
        return visualization

    def _draw_keypoint_set(
        self,
        image: Tensor,
        xy: Tensor,
        visibility: Tensor,
        *,
        connectivity: list[tuple[int, int]] | None,
        color: Color,
        radius: int,
    ) -> Tensor:
        if xy.numel() == 0:
            return image
        output = draw_keypoints(
            image,
            xy,
            connectivity=connectivity,
            colors=color,
            radius=radius,
            visibility=visibility,
        )
        if self.draw_indices:
            output = self.draw_keypoint_indices(
                output,
                xy,
                visibility,
                color=color,
            )
        return output

    @staticmethod
    def draw_keypoint_indices(
        canvas: Tensor,
        keypoints: Tensor,
        visibility: Tensor,
        *,
        color: Color,
        offset: tuple[int, int] = (7, 7),
    ) -> Tensor:
        """Draw visible keypoint indices with cycled text offsets."""
        image = Image.fromarray(
            canvas.permute(1, 2, 0).detach().cpu().numpy()
        )
        draw = ImageDraw.Draw(image)
        offset_y, offset_x = offset
        offset_modes = (
            (offset_y, -offset_x),
            (offset_y, offset_x),
            (-offset_y, offset_x),
            (-offset_y, -offset_x),
        )

        for instance_keypoints, instance_visibility in zip(
            keypoints, visibility, strict=True
        ):
            for keypoint_index, ((x, y), is_visible) in enumerate(
                zip(instance_keypoints, instance_visibility, strict=True)
            ):
                if not bool(is_visible):
                    continue
                label = str(keypoint_index)
                left, top, right, bottom = draw.textbbox((0, 0), label)
                text_width = right - left
                text_height = bottom - top
                delta_y, delta_x = offset_modes[
                    keypoint_index % len(offset_modes)
                ]
                draw.text(
                    (
                        int(x) - text_width // 2 + delta_x,
                        int(y) - text_height // 2 + delta_y,
                    ),
                    label,
                    fill=color,
                )

        output = np.asarray(image).copy()
        return torch.from_numpy(output).permute(2, 0, 1)

    @staticmethod
    def _get_radius(canvas: Tensor) -> int:
        height, width = canvas.shape[-2:]
        if height < 96 and width < 96:
            return 1
        if height > 512 or width > 512:
            return 5
        return 2

    def _prediction_connectivity(
        self, predictions: dai.ImgDetections
    ) -> list[tuple[int, int]] | None:
        if self.connectivity is not None:
            return self.connectivity
        for detection in predictions.detections:
            edges = detection.getEdges()
            if edges:
                return [(int(start), int(end)) for start, end in edges]
        return None

    @staticmethod
    def _clip_keypoints(keypoints: Tensor, canvas: Tensor) -> None:
        keypoints[..., 0].clamp_(0, canvas.shape[-1] - 1)
        keypoints[..., 1].clamp_(0, canvas.shape[-2] - 1)

    @staticmethod
    def _normalize_connectivity(
        connectivity: list[tuple[int, int]] | None,
    ) -> list[tuple[int, int]] | None:
        if connectivity is None:
            return None
        normalized = [tuple(edge) for edge in connectivity]
        if not all(
            len(edge) == 2 and all(isinstance(index, int) for index in edge)
            for edge in normalized
        ):
            raise ValueError(
                "connectivity must contain pairs of keypoint indices."
            )
        return [(edge[0], edge[1]) for edge in normalized]

    @staticmethod
    def _validate_connectivity(
        connectivity: list[tuple[int, int]] | None,
        n_keypoints: int,
    ) -> None:
        if connectivity is None:
            return
        invalid = [
            edge
            for edge in connectivity
            if min(edge) < 0 or max(edge) >= n_keypoints
        ]
        if invalid:
            raise ValueError(
                f"Connectivity contains invalid edges {invalid} for "
                f"{n_keypoints} keypoints."
            )
