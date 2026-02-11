from typing import Any

import depthai as dai
import numpy as np
from loguru import logger

from luxonis_eval.parsers.base_parser import BaseParser


class ClassificationParser(BaseParser):
    """Parser for classification model outputs."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the classification parser."""
        super().__init__(**kwargs)

    def softmax(
        self, x: np.ndarray, axis: int | None = None, keep_dims: bool = False
    ) -> np.ndarray:
        """Apply softmax to an array.

        Parameters
        ----------
        x : np.ndarray
            Input array.
        axis : int | None, optional
            Axis over which to apply softmax.
        keep_dims : bool, default=False
            Whether to keep reduced dimensions.

        Returns
        -------
        np.ndarray
            Softmax-normalized array.
        """
        ex = np.exp(x)
        return ex / np.sum(ex, axis=axis, keepdims=keep_dims)

    def parse(
        self, raw_output: dai.NNData | list[np.ndarray], **kwargs: Any
    ) -> np.ndarray:
        """Parse backend output into class scores.

        Parameters
        ----------
        raw_output : dai.NNData | list[np.ndarray]
            Backend inference output.
        **kwargs : Any
            Additional parser arguments.

        Returns
        -------
        np.ndarray
            Classification scores.
        """
        # Retrieve additional task-specific options
        apply_softmax = kwargs.get("apply_softmax", False)

        if isinstance(raw_output, dai.NNData):
            layer_names = raw_output.getAllLayerNames()
            logger.debug(f"Processing output with layers: {layer_names}")
            output_name = layer_names[0]
            scores = raw_output.getTensor(output_name, dequantize=True)
        elif isinstance(raw_output, list):
            scores = raw_output[0]
        else:
            raise TypeError(
                "raw_output must be dai.NNData or list[np.ndarray]"
            )

        scores = np.array(scores).flatten()

        if apply_softmax:
            scores = self.softmax(scores)

        return scores
