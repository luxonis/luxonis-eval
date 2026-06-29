from typing import Any

import numpy as np
import onnxruntime as ort
from luxonis_ml.nn_archive.utils import is_nn_archive
from luxonis_ml.typing import PathType

from luxonis_eval.config.nn_archive import (
    load_onnx_bytes_from_nn_archive,
)
from luxonis_eval.engines.base_engine import BaseEngine, ModelSpec
from luxonis_eval.engines.io import TensorMapOutput, TensorSpec


class OnnxEngine(BaseEngine, register_name="onnx"):
    """ONNX Runtime inference engine."""

    def __init__(
        self,
        model_path: PathType,
        providers: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ONNX inference engine.

        Parameters
        ----------
        model_path : PathType
            Path to the ONNX model file or NNArchive.
        providers : list[str] | None, optional
            ONNX Runtime execution providers.
        **kwargs : Any
            Additional engine configuration.
        """
        super().__init__(model_path=model_path, **kwargs)
        self.providers = providers or ["CPUExecutionProvider"]
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._output_names: tuple[str, ...] = ()
        self._visualization_frame: np.ndarray | None = None

    def setup(self) -> ModelSpec:
        """Initialize the ONNX Runtime session and resolve model
        spec."""
        if self._session is None:
            session_source: str | bytes
            if is_nn_archive(self.model_path):
                session_source = load_onnx_bytes_from_nn_archive(
                    self.model_path
                )
            else:
                session_source = str(self.model_path)

            self._session = ort.InferenceSession(
                session_source, providers=self.providers
            )
            session_inputs = self._session.get_inputs()
            if len(session_inputs) != 1:
                raise NotImplementedError(
                    "Only single-input ONNX models are supported in luxonis-eval."
                )
            self._input_name = session_inputs[0].name
            self._output_names = tuple(
                output_meta.name for output_meta in self._session.get_outputs()
            )

        if self.model_spec is not None:
            return self.model_spec

        input_meta = self._session.get_inputs()[0]
        shape = input_meta.shape
        if len(shape) != 4:
            raise ValueError(
                f"Unexpected input shape for ONNX: {shape}. Expected input shape in NCHW format."
            )

        if not all(isinstance(dim, int) for dim in shape):
            raise TypeError(
                f"ONNX input shape must be statically defined. Got {shape}."
            )

        output_specs = tuple(
            TensorSpec(
                name=output_meta.name,
                shape=tuple(output_meta.shape)
                if all(isinstance(dim, int) for dim in output_meta.shape)
                else None,
                dtype=output_meta.type,
            )
            for output_meta in self._session.get_outputs()
        )
        return self._set_model_spec(
            ModelSpec(
                input=TensorSpec(
                    name=input_meta.name,
                    shape=tuple(shape),
                    dtype=input_meta.type,
                    layout="NCHW",
                ),
                outputs=output_specs,
            )
        )

    def infer_once(self, img: np.ndarray) -> TensorMapOutput:
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

        if img.ndim == 2:
            x = img[None, None, :, :]
        elif img.ndim == 3 and img.shape[2] == 1:
            x = img[:, :, 0][None, None, :, :]
        elif img.ndim == 3:
            x = np.transpose(img, (2, 0, 1))[None, :, :, :]
        else:
            raise ValueError(
                f"Unsupported image shape for ONNX inference: {img.shape}. "
                "Expected `HW` or `HWC` loader output."
            )

        output_values = self._session.run(None, {self._input_name: x})
        return TensorMapOutput(
            tensors=dict(zip(self._output_names, output_values, strict=True))
        )

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
        self._output_names = ()
        self._visualization_frame = None
        self.model_spec = None
