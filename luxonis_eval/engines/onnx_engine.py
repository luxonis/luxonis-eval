from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from luxonis_eval.engines.base_engine import BaseEngine, ModelSpec


class OnnxEngine(BaseEngine, register_name="onnx"):
    """ONNX Runtime inference engine."""

    def __init__(
        self,
        model_path: str,
        *,
        providers: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ONNX inference engine.

        Parameters
        ----------
        model_path : str
            Path to the ONNX model file.
        providers : list[str] | None, optional
            ONNX Runtime execution providers.
        **kwargs : Any
            Additional engine configuration.
        """
        super().__init__(model_path=model_path, **kwargs)
        self.model_path = (
            self.model_path
            if isinstance(self.model_path, Path)
            else Path(self.model_path)
        )
        self.providers = providers or ["CPUExecutionProvider"]
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._visualization_frame: np.ndarray | None = None

    def setup(self) -> ModelSpec:
        """Initialize the ONNX Runtime session and resolve model
        spec."""
        if self._session is None:
            self._session = ort.InferenceSession(
                str(self.model_path), providers=self.providers
            )
            self._input_name = self._session.get_inputs()[0].name

        if self.model_spec is not None:
            return self.model_spec

        shape = self._session.get_inputs()[0].shape
        if len(shape) != 4:
            raise ValueError(
                f"Unexpected input shape for ONNX: {shape}. Expected input shape in NCHW format."
            )

        height = shape[2]
        width = shape[3]
        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError(
                f"ONNX input shape must be statically defined. Got {shape}."
            )

        return self._set_model_spec(ModelSpec(width=width, height=height))

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
        if self._session is None or self._input_name is None:
            raise RuntimeError(
                "OnnxEngine.setup() must be called before infer_once()."
            )

        self._visualization_frame = img.copy()
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        x = np.transpose(img, (2, 0, 1))
        x = x[None, :, :, :]
        return self._session.run(None, {self._input_name: x})

    def vis_frame(self) -> np.ndarray:
        """Return the visualization frame.

        Returns
        -------
        np.ndarray
            Copy of the input image.
        """
        if self._visualization_frame is None:
            raise RuntimeError("Visualization frame is unavailable.")
        return self._visualization_frame

    def close(self) -> None:
        """Release ONNX Runtime resources."""
        self._session = None
        self._input_name = None
        self._visualization_frame = None
        self.model_spec = None
