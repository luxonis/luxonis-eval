from typing import Any

import depthai as dai
import numpy as np
from depthai_nodes import SegmentationMask
from depthai_nodes.message.creators import create_segmentation_message
from loguru import logger

from .base_parser import BaseParser


class SemanticSegmentationParser(BaseParser):
    """Parser for semantic segmentation model outputs."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the semantic segmentation parser."""
        super().__init__(**kwargs)

    def parse(
        self,
        raw_output: dai.NNData | list[np.ndarray],
        *,
        classes_in_one_layer: bool = False,
        **kwargs: Any,
    ) -> SegmentationMask:
        """Parse backend output into detection predictions.

        Parameters
        ----------
        raw_output : dai.NNData | list[np.ndarray]
            Backend inference output.
        classes_in_one_layer : bool, default=False
            Whether the model outputs classes in a single layer.
        **kwargs : Any
            Additional parser arguments.

        Returns
        -------
        SegmentationMask
            Detection results including boxes, scores, classes, and metadata.
        """
        if isinstance(raw_output, dai.NNData):
            layer_names = raw_output.getAllLayerNames()
            logger.debug(f"Processing output with layers: {layer_names}")
            output_name = layer_names[0]
            segmentation_mask: np.ndarray = raw_output.getTensor(
                output_name, dequantize=True
            )  # type: ignore
        elif isinstance(raw_output, list):
            segmentation_mask: np.ndarray = raw_output[0]
        else:
            raise TypeError(
                f"Unsupported raw_output type: {type(raw_output)}. Expected dai.NNData or list[np.ndarray]."
            )

        if len(segmentation_mask.shape) == 4:
            segmentation_mask = segmentation_mask[0]

        if len(segmentation_mask.shape) != 3:
            raise ValueError(
                f"Expected 3D output tensor, got {len(segmentation_mask.shape)}D."
            )

        np_function = np.argmax
        mask_shape = segmentation_mask.shape
        min_dim = np.argmin(mask_shape)
        if min_dim == len(mask_shape) - 1:
            segmentation_mask = segmentation_mask.transpose(2, 0, 1)
        adding_unassigned_class = False
        if segmentation_mask.shape[0] == 1:  # shape is (1, H, W)
            if classes_in_one_layer:
                np_function = np.max
            else:
                # If there is only one class, add an unassigned class
                adding_unassigned_class = True
                segmentation_mask = np.vstack(
                    (
                        np.zeros(
                            (
                                1,
                                segmentation_mask.shape[1],
                                segmentation_mask.shape[2],
                            ),
                            dtype=np.float32,
                        ),
                        segmentation_mask,
                    )
                )

        class_map = (
            np_function(segmentation_mask, axis=0)
            .reshape(segmentation_mask.shape[1], segmentation_mask.shape[2])
            .astype(np.int16)
        )
        if adding_unassigned_class:
            class_map = class_map - 1

        return create_segmentation_message(class_map)
