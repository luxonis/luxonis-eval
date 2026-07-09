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
        del model_spec, kwargs
        _, segmentation_mask = output.first()
        class_map = DepthAINodesSegmentationParser.compute(
            np.asarray(segmentation_mask),
            classes_in_one_layer=classes_in_one_layer,
        )

        return create_segmentation_message(class_map)
