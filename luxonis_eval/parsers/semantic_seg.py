from typing import Any

import numpy as np
from depthai_nodes import SegmentationMask
from depthai_nodes.message.creators import create_segmentation_message
from depthai_nodes.node.parsers.segmentation import (
    SegmentationParser as DepthAINodesSegmentationParser,
)

from luxonis_eval.engines.base_engine import ModelSpec
from luxonis_eval.engines.io import EngineOutput
from .base_parser import BaseParser


class SemanticSegmentationParser(BaseParser):
    """Parser for semantic segmentation model outputs."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the semantic segmentation parser."""
        super().__init__(**kwargs)

    def parse(
        self,
        output: EngineOutput,
        *,
        model_spec: ModelSpec,
        classes_in_one_layer: bool = False,
        **kwargs: Any,
    ) -> SegmentationMask:
        """Parse backend output into detection predictions.

        Parameters
        ----------
        output : EngineOutput
            Engine-normalized inference output.
        model_spec : ModelSpec
            Resolved model IO metadata.
        classes_in_one_layer : bool, default=False
            Whether the model outputs classes in a single layer.
        **kwargs : Any
            Additional parser arguments.

        Returns
        -------
        SegmentationMask
            Detection results including boxes, scores, classes, and metadata.
        """
        del model_spec, kwargs
        _, segmentation_mask = output.first()
        class_map = DepthAINodesSegmentationParser.compute(
            np.asarray(segmentation_mask),
            classes_in_one_layer=classes_in_one_layer,
        )

        return create_segmentation_message(class_map)
