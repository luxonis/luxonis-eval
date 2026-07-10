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

    @staticmethod
    def _looks_like_integer_class_mask(
        mask: np.ndarray,
        output_dtype: str | None,
    ) -> bool:
        if np.issubdtype(mask.dtype, np.bool_):
            return True
        if np.issubdtype(mask.dtype, np.integer):
            return True
        if output_dtype is None:
            return False

        normalized_dtype = output_dtype.lower()
        return "bool" in normalized_dtype or "int" in normalized_dtype

    @staticmethod
    def _normalize_single_channel_class_mask(mask: np.ndarray) -> np.ndarray:
        class_map = np.asarray(mask)
        if class_map.ndim != 2:
            raise ValueError(
                f"Expected a 2D class mask after squeezing, got {class_map.ndim}D."
            )

        if np.issubdtype(class_map.dtype, np.bool_):
            return np.where(class_map, 0, 255).astype(np.uint8)

        class_map = class_map.astype(np.int64, copy=False)
        if np.any(class_map < 0) or np.any(class_map > 255):
            raise ValueError(
                "Segmentation class mask values must be in the range [0, 255]."
            )

        if np.all((class_map == 0) | (class_map == 1)):
            return np.where(class_map > 0, 0, 255).astype(np.uint8)

        return class_map.astype(np.uint8, copy=False)

    @classmethod
    def _maybe_parse_single_channel_class_mask(
        cls,
        mask: np.ndarray,
        *,
        output_dtype: str | None,
    ) -> np.ndarray | None:
        if not cls._looks_like_integer_class_mask(mask, output_dtype):
            return None

        squeezed_mask = np.asarray(mask)
        if squeezed_mask.ndim == 4:
            if squeezed_mask.shape[0] != 1:
                return None
            squeezed_mask = squeezed_mask[0]

        if squeezed_mask.ndim == 2:
            return cls._normalize_single_channel_class_mask(squeezed_mask)

        if squeezed_mask.ndim != 3:
            return None

        if squeezed_mask.shape[0] == 1:
            squeezed_mask = squeezed_mask[0]
        elif squeezed_mask.shape[-1] == 1:
            squeezed_mask = squeezed_mask[..., 0]
        else:
            return None

        return cls._normalize_single_channel_class_mask(squeezed_mask)

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
        output_name, segmentation_mask = output.first()
        output_spec = next(
            (
                spec
                for spec in model_spec.outputs
                if spec.name == output_name
            ),
            None,
        )
        segmentation_mask_array = np.asarray(segmentation_mask)

        class_map = self._maybe_parse_single_channel_class_mask(
            segmentation_mask_array,
            output_dtype=output_spec.dtype if output_spec is not None else None,
        )
        if class_map is None:
            class_map = DepthAINodesSegmentationParser.compute(
                segmentation_mask_array,
                classes_in_one_layer=classes_in_one_layer,
            )

        return create_segmentation_message(class_map)
