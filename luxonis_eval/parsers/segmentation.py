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
        class_map = DepthAINodesSegmentationParser.compute(
            np.asarray(segmentation_mask),
            classes_in_one_layer=classes_in_one_layer,
        )

        return create_segmentation_message(class_map)
