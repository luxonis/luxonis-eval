from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort

from luxonis_eval.engines.base_engine import BaseEngine

if TYPE_CHECKING:
    from luxonis_eval.inferer import Inferer


class OnnxEngine(BaseEngine, register_name="onnx"):
    """ONNX Runtime inference engine."""

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
        **kwargs: Any,
    ) -> None:
        """Initialize the ONNX inference engine.

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
        **kwargs : Any
            Additional engine configuration.
        """
        super().__init__(**kwargs)
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
