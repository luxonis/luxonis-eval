from __future__ import annotations

from typing import TYPE_CHECKING, Any

import depthai as dai
import numpy as np
from depthai import ADatatype

from luxonis_eval.engines.base_engine import BaseEngine

if TYPE_CHECKING:
    from luxonis_eval.inferer import Inferer


class DepthAIEngine(BaseEngine, register_name="depthai"):
    """DepthAI inference engine."""

    def __init__(self, inferer: Inferer, **kwargs: Any) -> None:
        """Initialize the DepthAI inference engine.

        Parameters
        ----------
        inferer : Inferer
            Inferer instance providing device and model information.
        **kwargs : Any
            Additional engine configuration.
        """
        super().__init__(**kwargs)
        self.inferer = inferer
        self._pipeline = None

    def setup(self) -> None:
        """Set up the DepthAI pipeline."""
        self._pipeline = dai.Pipeline(self.inferer.device)
        nn_node = self._pipeline.create(dai.node.NeuralNetwork)
        nn_node.setNNArchive(self.inferer.nn_archive)

        self._input_queue = nn_node.input.createInputQueue()
        self._output_queue = nn_node.out.createOutputQueue()
        self._passthrough = nn_node.passthrough.createOutputQueue()

        self._pipeline.start()

    def infer_once(self, img: np.ndarray) -> ADatatype:
        """Run inference on a single image using DepthAI.

        Parameters
        ----------
        img : np.ndarray
            Input image.

        Returns
        -------
        ADatatype
            Raw DepthAI inference output.
        """
        assert img.shape[0] == self.inferer.height
        assert img.shape[1] == self.inferer.width

        if self.inferer.platform_name == "RVC2":
            img_frame_type = dai.ImgFrame.Type.BGR888p
            img_for_device = np.transpose(img, (2, 0, 1))
        else:
            img_frame_type = dai.ImgFrame.Type.BGR888i
            img_for_device = img

        new_input = dai.ImgFrame()
        new_input.setFrame(img_for_device)
        new_input.setWidth(self.inferer.width)
        new_input.setHeight(self.inferer.height)
        new_input.setType(img_frame_type)
        self._input_queue.send(new_input)

        return self._output_queue.get()

    def vis_frame(self) -> np.ndarray:
        """Get visualization frame from passthrough.

        Returns
        -------
        np.ndarray
            Visualization frame.
        """
        return self._passthrough.get().getCvFrame()  # type: ignore

    def teardown(self) -> None:
        """Tear down the DepthAI pipeline."""
        self._pipeline = None
