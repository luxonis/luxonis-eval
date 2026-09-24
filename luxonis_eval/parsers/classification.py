from typing import Any

import numpy as np
from depthai_nodes import Classifications
from depthai_nodes.message.creators import create_classification_message
from depthai_nodes.node.parsers.classification import (
    ClassificationParser as DepthAINodesClassificationParser,
)

from luxonis_eval.engines.io import EngineOutput
from luxonis_eval.parsers.base_parser import BaseParser
from luxonis_eval.utils.utils import ordered_class_names


class ClassificationParser(BaseParser):
    """Parser for classification model outputs."""

    def __init__(self, is_softmax: bool = True, **kwargs: Any) -> None:
        """Initialize the parser; set is_softmax=False for raw
        logits."""
        super().__init__(**kwargs)
        self.is_softmax = is_softmax

    def parse(self, output: EngineOutput) -> Classifications:
        """Parse backend output into class scores.

        Parameters
        ----------
        output : EngineOutput
            Engine-normalized inference output.
        Returns
        -------
        Classifications
            Classification scores.
        """
        classes = ordered_class_names(self.context.class_map)
        _, scores = output.get_first()
        scores = np.asarray(scores, dtype=np.float64).flatten()
        if scores.size == 0:
            raise ValueError("Classification output is empty.")

        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "Classification output contains non-finite values before "
                "post-processing."
            )

        if not self.is_softmax:
            # Subtract the largest value first so softmax stays numerically stable.
            scores = scores - np.max(scores)
        scores = DepthAINodesClassificationParser.compute(
            scores,
            is_softmax=self.is_softmax,
        )

        return create_classification_message(classes=classes, scores=scores)
