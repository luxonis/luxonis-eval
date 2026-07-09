from typing import Any

import depthai as dai
import numpy as np
from depthai_nodes.message.creators import create_segmentation_message
from depthai_nodes.node.parsers.segmentation import (
    SegmentationParser as DepthAINodesSegmentationParser,
)

from luxonis_eval.engines.base_engine import ModelSpec
from luxonis_eval.engines.io import EngineOutput

from .base_parser import BaseParser


class SegmentationParser(BaseParser):
    """Parser for semantic segmentation model outputs."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the segmentation parser."""
        super().__init__(**kwargs)

    def parse(
        self,
        output: EngineOutput,
        model_spec: ModelSpec,
        *,
        classes_in_one_layer: bool = False,
        **kwargs: Any,
    ) -> dai.SegmentationMask:
        """Parse backend output into segmentation predictions."""
        del kwargs
        output_names = output.names()
        if len(output_names) != 1:
            raise ValueError(
                f"Expected exactly one output tensor, got {list(output_names)}."
            )

        output_name = output_names[0]
        output_layout = next(
            (
                output_spec.layout
                for output_spec in model_spec.outputs
                if output_spec.name == output_name
            ),
            None,
        )
        segmentation_mask = output.get(output_name, layout=output_layout)
        class_map = self._parse_segmentation_output(
            np.asarray(segmentation_mask),
            classes_in_one_layer=classes_in_one_layer,
        )

        return create_segmentation_message(class_map)

    @staticmethod
    def _parse_segmentation_output(
        segmentation_mask: np.ndarray,
        *,
        classes_in_one_layer: bool,
    ) -> np.ndarray:
        mask = np.asarray(segmentation_mask)

        if not classes_in_one_layer:
            maybe_probability_map = (
                SegmentationParser._maybe_parse_binary_probability_map(mask)
            )
            if maybe_probability_map is not None:
                return maybe_probability_map

        return DepthAINodesSegmentationParser.compute(
            mask,
            classes_in_one_layer=classes_in_one_layer,
        )

    @staticmethod
    def _maybe_parse_binary_probability_map(
        segmentation_mask: np.ndarray,
    ) -> np.ndarray | None:
        mask = np.asarray(segmentation_mask)
        if mask.ndim == 4:
            mask = mask[0]

        if mask.ndim != 3:
            return None

        if np.argmin(mask.shape) == len(mask.shape) - 1:
            mask = mask.transpose(2, 0, 1)

        if mask.shape[0] != 1:
            return None

        score_map = mask[0]
        if score_map.dtype.kind not in {"f", "c"}:
            return None

        if np.min(score_map) < 0.0 or np.max(score_map) > 1.0:
            return None

        # Some RVC4 segmentation exports emit a single-channel foreground
        # probability map rather than a signed logit map.
        return np.where(score_map >= 0.5, 0, 255).astype(np.uint8)
