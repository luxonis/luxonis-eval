from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import onnxruntime as ort
from loguru import logger
from luxonis_ml.nn_archive.utils import is_nn_archive
from luxonis_ml.typing import PathType

from luxonis_eval.config.nn_archive import (
    get_archive_input,
    load_onnx_bytes_from_nn_archive,
)
from luxonis_eval.engines.base_engine import BaseEngine, ModelSpec
from luxonis_eval.engines.io import EngineOutput, TensorLayout, TensorSpec

if TYPE_CHECKING:
    from luxonis_ml.nn_archive.config import Config as NNArchiveConfig


_SUPPORTED_LAYOUTS = {"NCHW", "NHWC", "NC", "CHW", "HWC", "HW"}
_warned_unverified_layout_requests: set[tuple[str, str, TensorLayout]] = set()


def _normalize_layout(layout: Any) -> TensorLayout | None:
    if layout is None:
        return None

    if not isinstance(layout, str):
        layout = getattr(layout, "name", None) or getattr(
            layout, "value", None
        )
    if not isinstance(layout, str):
        return None

    normalized = layout.upper()
    if normalized not in _SUPPORTED_LAYOUTS:
        return None
    return cast(TensorLayout, normalized)


def _transpose_tensor_to_layout(
    tensor: np.ndarray,
    *,
    source_layout: TensorLayout,
    target_layout: TensorLayout,
) -> np.ndarray:
    if source_layout == target_layout:
        return tensor.copy()

    if tensor.ndim != len(source_layout) or tensor.ndim != len(target_layout):
        raise ValueError(
            f"Cannot convert tensor with shape {tensor.shape} from "
            f"{source_layout!r} to {target_layout!r}."
        )
    if sorted(source_layout) != sorted(target_layout):
        raise ValueError(
            f"Cannot convert tensor layout from {source_layout!r} "
            f"to incompatible layout {target_layout!r}."
        )

    axes = tuple(source_layout.index(dim) for dim in target_layout)
    return np.transpose(tensor, axes).copy()

@dataclass(frozen=True, slots=True)
class ONNXEngineOutput(EngineOutput):
    tensors: dict[str, np.ndarray]
    tensor_specs: dict[str, TensorSpec] | None = None
    source_name: str | None = None

    def names(self) -> tuple[str, ...]:
        return tuple(self.tensors.keys())

    def get(
        self,
        name: str,
        *,
        layout: TensorLayout | None = None,
    ) -> np.ndarray:
        try:
            tensor = self.tensors[name]
        except KeyError as err:
            raise ValueError(
                f"Requested output tensor {name!r} is not available. "
                f"Available tensors: {list(self.tensors)}."
            ) from err

        if layout is None:
            return tensor.copy()

        tensor_spec = (
            None if self.tensor_specs is None else self.tensor_specs.get(name)
        )
        resolved_layout = None if tensor_spec is None else tensor_spec.layout
        if resolved_layout is None:
            source_name = self.source_name or "<unknown model>"
            warning_key = (source_name, name, layout)
            if warning_key not in _warned_unverified_layout_requests:
                logger.warning(
                    f"Requested layout {layout!r} for ONNX output tensor {name!r} "
                    f"from {source_name!r} cannot be verified because no layout "
                    "metadata is available. Passing the raw ONNX output through "
                    "as-is."
                )
                _warned_unverified_layout_requests.add(warning_key)
            return tensor.copy()

        return _transpose_tensor_to_layout(
            tensor,
            source_layout=resolved_layout,
            target_layout=layout,
        )

    def select(self, names: Sequence[str] | None) -> "ONNXEngineOutput":
        if names is None:
            return self

        missing = [name for name in names if name not in self.tensors]
        if missing:
            raise ValueError(
                f"Requested outputs are missing from engine result: {missing}"
            )

        return ONNXEngineOutput(
            tensors={name: self.tensors[name] for name in names},
            tensor_specs=(
                None
                if self.tensor_specs is None
                else {name: self.tensor_specs[name] for name in names}
            ),
            source_name=self.source_name,
        )

class OnnxEngine(BaseEngine, register_name="onnx"):
    """ONNX Runtime inference engine."""

    output_type = ONNXEngineOutput

    def __init__(
        self,
        model_path: PathType,
        providers: list[str] | None = None,
        nn_archive_cfg: "NNArchiveConfig | None" = None,
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
        self.nn_archive_cfg = nn_archive_cfg
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
                f"Unexpected input shape for ONNX: {shape}. Expected a 4D input tensor."
            )

        if not all(isinstance(dim, int) for dim in shape):
            raise TypeError(
                f"ONNX input shape must be statically defined. Got {shape}."
            )

        archive_input = (
            get_archive_input(self.nn_archive_cfg)
            if self.nn_archive_cfg is not None
            else None
        )
        input_layout = _normalize_layout(
            getattr(archive_input, "layout", None)
        ) or "NCHW"
        output_specs = tuple(
            self._build_output_spec(idx, output_meta)
            for idx, output_meta in enumerate(self._session.get_outputs())
        )
        return self._set_model_spec(
            ModelSpec(
                input=TensorSpec(
                    name=input_meta.name,
                    shape=tuple(shape),
                    dtype=input_meta.type,
                    layout=input_layout,
                ),
                outputs=output_specs,
            )
        )

    def infer_once(self, img: np.ndarray) -> ONNXEngineOutput:
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

        model_spec = self._get_model_spec()
        self._visualization_frame = img.copy()
        if img.dtype != np.float32:
            img = img.astype(np.float32)

        input_layout = model_spec.input.layout
        if input_layout == "NCHW":
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
        elif input_layout == "NHWC":
            if img.ndim == 2:
                x = img[None, :, :, None]
            elif img.ndim == 3:
                x = img[None, :, :, :]
            else:
                raise ValueError(
                    f"Unsupported image shape for ONNX inference: {img.shape}. "
                    "Expected `HW` or `HWC` loader output."
                )
        else:
            raise ValueError(
                f"Unsupported ONNX input layout {input_layout!r}. "
                "Expected 'NCHW' or 'NHWC'."
            )

        output_values = self._session.run(None, {self._input_name: x})
        return ONNXEngineOutput(
            tensors=dict(zip(self._output_names, output_values, strict=True)),
            tensor_specs={
                output_spec.name: output_spec
                for output_spec in model_spec.outputs
            },
            source_name=str(self.model_path),
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

    def _build_output_spec(
        self, idx: int, output_meta: ort.NodeArg
    ) -> TensorSpec:
        archive_output = self._resolve_archive_output(idx, output_meta.name)
        return TensorSpec(
            name=output_meta.name,
            shape=tuple(output_meta.shape)
            if all(isinstance(dim, int) for dim in output_meta.shape)
            else None,
            dtype=output_meta.type,
            layout=_normalize_layout(getattr(archive_output, "layout", None)),
        )

    def _resolve_archive_output(
        self, idx: int, output_name: str
    ) -> Any | None:
        if self.nn_archive_cfg is None:
            return None

        archive_outputs = (
            getattr(self.nn_archive_cfg.model, "outputs", None) or []
        )
        for archive_output in archive_outputs:
            if getattr(archive_output, "name", None) == output_name:
                return archive_output

        if idx < len(archive_outputs):
            archive_output = archive_outputs[idx]
            archive_name = getattr(archive_output, "name", None)
            if archive_name not in {None, output_name}:
                logger.warning(
                    f"NNArchive output metadata name {archive_name!r} does not match "
                    f"ONNX output name {output_name!r} at index {idx}. "
                    "Using the archive output position to resolve layout metadata."
                )
            return archive_output

        return None
