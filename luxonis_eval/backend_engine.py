from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import depthai as dai
import numpy as np
import onnxruntime as ort
from depthai import ADatatype

if TYPE_CHECKING:
    from luxonis_eval.inferer import Inferer


class Backend(ABC):
    """Abstract backend interface."""

    @abstractmethod
    def setup(self) -> None:
        """Initialize backend resources."""
        ...

    @abstractmethod
    def infer_once(self, img: np.ndarray) -> Any:
        """Run inference on a single image."""
        ...

    @abstractmethod
    def vis_frame(self) -> np.ndarray:
        """Return a visualization frame."""
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Release backend resources."""
        ...


class DepthAIBackend(Backend):
    def __init__(self, inferer: Inferer) -> None:
        """Initialize the DepthAI backend.

        Parameters
        ----------
        inferer : Inferer
            Inferer instance providing device and model information.
        """
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


class OnnxBackend(Backend):
    def __init__(
        self,
        inferer: Inferer,
        *,
        providers: list[str] | None = None,
        mean: tuple[float, float, float] | float | None = (
            123.675,
            116.28,
            103.53,
        ),
        std: tuple[float, float, float] | float | None = (
            58.395,
            57.12,
            57.375,
        ),
    ) -> None:
        """Initialize the ONNX backend.

        Parameters
        ----------
        inferer : Inferer
            Inferer instance providing model information.
        providers : list[str] | None, optional
            ONNX Runtime execution providers.
        mean : tuple[float, float, float] | float | None, optional
            Mean used for input normalization.
        std : tuple[float, float, float] | float | None, optional
            Standard deviation used for input normalization.
        """
        self.inferer = inferer
        self.providers = providers or [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def setup(self) -> None:
        """Initialize the ONNX Runtime session."""
        if self.inferer.onnx_path is None:
            raise ValueError("ONNX path is not provided for ONNX inference.")
        self._session = ort.InferenceSession(
            str(self.inferer.onnx_path), providers=self.providers
        )
        self._input_name = self._session.get_inputs()[0].name

    def infer_once(self, img: np.ndarray) -> Any:
        """Run inference on a single image using ONNX Runtime.

        Parameters
        ----------
        img : np.ndarray
            Input image.

        Returns
        -------
        Any
            Raw ONNX Runtime output.
        """
        self._visualization_frame = img.copy()
        x = img.astype(np.float32)
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))
        x = x[None, :, :, :]
        return self._session.run(None, {self._input_name: x})

    def vis_frame(self) -> np.ndarray:
        """Return the visualization frame.

        Returns
        -------
        np.ndarray
            Copy of the input image.
        """
        return self._visualization_frame

    def teardown(self) -> None:
        """Release ONNX Runtime resources."""
        del self._session
