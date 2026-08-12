from typing import Any

import numpy as np
from depthai_nodes.message.creators import create_classification_message
from depthai_nodes.node.parsers.classification import (
    ClassificationParser as DepthAINodesClassificationParser,
)

from luxonis_eval.engines.base_engine import ModelSpec
from luxonis_eval.engines.io import EngineOutput
from luxonis_eval.parsers.base_parser import BaseParser
from luxonis_eval.parsers.predictions import Prediction
from luxonis_eval.utils.utils import ordered_class_names


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
    ) -> Prediction:
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
        Prediction
            Structured classification scores.
        """
        del model_spec, kwargs
        classes = ordered_class_names(class_map)
        _, scores = output.get_first()
        scores = np.asarray(scores, dtype=np.float64).flatten()
        if scores.size == 0:
            raise ValueError("Classification output is empty.")

        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "Classification output contains non-finite values before "
                "post-processing."
            )

        if apply_softmax:
            # Subtract the largest value first so softmax stays numerically stable.
            scores = scores - np.max(scores)
            scores = DepthAINodesClassificationParser.compute(
                scores,
                is_softmax=False,
            )
        else:
            scores = DepthAINodesClassificationParser.compute(
                scores,
                is_softmax=True,
            )

        return Prediction(
            classification=create_classification_message(
                classes=classes,
                scores=scores,
            )
        )
