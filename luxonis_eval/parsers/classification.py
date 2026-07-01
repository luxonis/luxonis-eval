from typing import Any

import numpy as np
from depthai_nodes import Classifications
from depthai_nodes.message.creators import create_classification_message
from depthai_nodes.node.parsers.classification import (
    ClassificationParser as DepthAINodesClassificationParser,
)

from luxonis_eval.engines.base_engine import ModelSpec
from luxonis_eval.engines.io import EngineOutput
from luxonis_eval.parsers.base_parser import BaseParser
from luxonis_eval.utils._depthai_nodes import ordered_class_names


class ClassificationParser(BaseParser):
    """Parser for classification model outputs."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the classification parser."""
        super().__init__(**kwargs)

    def parse(
        self,
        output: EngineOutput,
        model_spec: ModelSpec,
        *,
        class_map: dict[int, str],
        apply_softmax: bool = False,
        **kwargs: Any,
    ) -> Classifications:
        """Parse backend output into class scores.

        Parameters
        ----------
        output : EngineOutput
            Engine-normalized inference output.
        model_spec : ModelSpec
            Resolved model IO metadata.
        apply_softmax : bool, default=False
            Whether to apply softmax to the output scores.
        **kwargs : Any
            Additional parser arguments.

        Returns
        -------
        Classifications
            Classification scores.
        """
        del model_spec, kwargs
        classes = ordered_class_names(class_map)
        _, scores = output.first()
        scores = np.asarray(scores).flatten()
        scores = DepthAINodesClassificationParser.compute(
            scores,
            is_softmax=not apply_softmax,
        )

        return create_classification_message(classes=classes, scores=scores)
