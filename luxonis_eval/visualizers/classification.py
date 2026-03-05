from typing import Any

import cv2
import numpy as np
from depthai_nodes import Classifications

from luxonis_eval.visualizers.base_visualizer import BaseVisualizer


class ClassificationVisualizer(BaseVisualizer):
    """Visualizer for classification tasks."""

    def __init__(
        self, *, max_visualizations: int | None = None, **kwargs: Any
    ) -> None:
        """Initialize the classification visualizer.

        Parameters
        ----------
        max_visualizations : int | None, optional
            Maximum number of visualizations to display. If None, visualizes all samples.
        **kwargs : Any
            Additional visualization options.
        """
        self.max_visualizations = max_visualizations
        self.num_visualized = 0
        super().__init__(**kwargs)

    @staticmethod
    def _fit_text(
        text: str,
        font: int,
        font_scale: float,
        thickness: int,
        max_text_w: int,
    ) -> str:
        """Truncate text to fit within the image width.

        Parameters
        ----------
        text : str
            Text to truncate.
        font : int
            Font to use.
        font_scale : float
            Font scale.
        thickness : int
            Text thickness.
        max_text_w : int
            Maximum width of the text.

        Returns
        -------
        str
            Truncated text.
        """
        (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        if tw <= max_text_w:
            return text
        while len(text) > 1:
            text = text[:-1]
            (tw, _), _ = cv2.getTextSize(
                text + "...", font, font_scale, thickness
            )
            if tw <= max_text_w:
                return text + "..."
        return text

    @staticmethod
    def _draw_text(
        frame: np.ndarray,
        text: str,
        pos: tuple[int, int],
        color: tuple[int, int, int],
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.5,
        thickness: int = 1,
        outline_thickness: int = 3,
    ) -> None:
        """Draw text with a dark outline for readability.

        Parameters
        ----------
        frame : np.ndarray
            Image to draw on.
        text : str
            Text to draw.
        pos : tuple[int, int]
            Bottom-left corner of the text.
        color : tuple[int, int, int]
            Text color in BGR format.
        font : int, optional
            Font to use, by default cv2.FONT_HERSHEY_SIMPLEX
        font_scale : float, optional
            Font scale, by default 0.5
        thickness : int, optional
            Text thickness, by default 1
        outline_thickness : int, optional
            Thickness of the text outline, by default 3
        """
        cv2.putText(
            frame,
            text,
            pos,
            font,
            font_scale,
            (0, 0, 0),
            outline_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            pos,
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _short_name(name: str) -> str:
        """Keep only the first comma-separated name.

        Parameters
        ----------
        name : str
            Original name.
        Returns
        -------
        str
            Shortened name.
        """
        return name.split(",", maxsplit=1)[0].strip()

    def visualize(
        self,
        predictions: Classifications,
        target: Any,
        vis_frame: np.ndarray,
        *,
        resize_ratio: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Visualize the classification results.

        Parameters
        ----------
        predictions : Classifications
            Model predictions.
        target : Any
            Ground truth target.
        vis_frame : np.ndarray
            Frame to visualize on.
        resize_ratio : float | None, optional
            Ratio to resize the visualization frame by.
        **kwargs : Any
            Additional visualization options.
        """
        if (
            self.max_visualizations is not None
            and self.num_visualized >= self.max_visualizations
        ):
            return

        if resize_ratio is not None:
            h, w = vis_frame.shape[:2]
            new_w = int(w * resize_ratio)
            new_h = int(h * resize_ratio)
            vis_frame = cv2.resize(vis_frame, (new_w, new_h))

        top_k = kwargs.get("top_k", 5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img_w = vis_frame.shape[1]
        font_scale = max(0.3, img_w / 1000)
        thickness = max(1, int(img_w / 500))
        line_height = int(font_scale * 40)
        y_offset = line_height + 5
        margin = 10
        max_text_w = img_w - 2 * margin

        outline_thickness = thickness + 2

        pred_classes = predictions.classes[:top_k]
        pred_scores = predictions.scores[:top_k]

        for i, (cls_name, score) in enumerate(
            zip(pred_classes, pred_scores, strict=True)
        ):
            text = self._fit_text(
                f"{self._short_name(cls_name)}: {score:.2%}",
                font,
                font_scale,
                thickness,
                max_text_w,
            )
            y = y_offset + i * line_height
            self._draw_text(
                vis_frame,
                text,
                (margin, y),
                (0, 255, 0),
                font,
                font_scale,
                thickness,
                outline_thickness,
            )

        if target is not None:
            class_index_map = kwargs.get("class_index_map")
            class_map = kwargs.get("class_map", {})
            inv_class_map = {v: k for k, v in class_map.items()}

            cls_target = target.get("/classification")
            if cls_target is not None:
                tgt = np.asarray(cls_target)
                target_idx = (
                    int(np.argmax(tgt))
                    if tgt.ndim > 0 and tgt.size > 1
                    else int(tgt)
                )
                if class_index_map is not None:
                    target_idx = int(class_index_map[target_idx])
                gt_label = inv_class_map.get(target_idx, str(target_idx))
            else:
                gt_label = str(target)

            gt_text = self._fit_text(
                f"GT: {self._short_name(gt_label)}",
                font,
                font_scale,
                thickness,
                max_text_w,
            )
            gt_y = y_offset + len(pred_classes) * line_height + 10
            self._draw_text(
                vis_frame,
                gt_text,
                (margin, gt_y),
                (0, 0, 255),
                font,
                font_scale,
                thickness,
                outline_thickness,
            )

        cv2.imshow("Classification Visualization", vis_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.num_visualized += 1
