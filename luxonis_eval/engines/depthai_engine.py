from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import depthai as dai
import numpy as np
from loguru import logger
from luxonis_ml.typing import PathType

from luxonis_eval.engines.base_engine import BaseEngine, ModelSpec
from luxonis_eval.engines.io import EngineOutput, TensorLayout, TensorSpec

if TYPE_CHECKING:
    from luxonis_ml.nn_archive.config import Config as NNArchiveConfig

@dataclass(frozen=True, slots=True)
class DepthAIEngineOutput(EngineOutput):
    raw_output: dai.NNData
    _selected_names: tuple[str, ...] | None = None

    def names(self) -> tuple[str, ...]:
        available = tuple(self.raw_output.getAllLayerNames())
        if self._selected_names is None:
            return available

        missing = [
            name for name in self._selected_names if name not in available
        ]
        if missing:
            raise ValueError(
                f"Requested outputs are missing from engine result: {missing}"
            )
        return self._selected_names

    def get(
        self,
        name: str,
        *,
        layout: TensorLayout | None = None,
    ) -> np.ndarray:
        if name not in self.names():
            raise ValueError(
                f"Requested output tensor {name!r} is not available. "
                f"Available tensors: {list(self.names())}."
            )

        get_tensor_kwargs: dict[str, object] = {"dequantize": True}
        if layout == "NCHW":
            get_tensor_kwargs["storageOrder"] = (
                dai.TensorInfo.StorageOrder.NCHW
            )
        elif layout == "NHWC":
            get_tensor_kwargs["storageOrder"] = (
                dai.TensorInfo.StorageOrder.NHWC
            )

        tensor = self.raw_output.getTensor(name, **get_tensor_kwargs)
        return np.asarray(tensor)

    def select(self, names: Sequence[str] | None) -> "DepthAIEngineOutput":
        if names is None:
            return self

        requested_names = tuple(names)
        missing = [
            name for name in requested_names if name not in self.names()
        ]
        if missing:
            raise ValueError(
                f"Requested outputs are missing from engine result: {missing}"
            )
        return DepthAIEngineOutput(self.raw_output, requested_names)

class DepthAIEngine(BaseEngine, register_name="depthai"):
    """DepthAI inference engine."""

    output_type = DepthAIEngineOutput

    def __init__(
        self,
        model_path: PathType,
        device_ip: str | None = None,
        nn_archive_cfg: "NNArchiveConfig | None" = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DepthAI inference engine.

        Parameters
        ----------
        model_path : PathType
            Path to the model file.
        device_ip : str | None, optional
            IP address of the DepthAI device.
        **kwargs : Any
            Additional engine configuration.
        """
        super().__init__(model_path=model_path, **kwargs)
        self.device_ip = device_ip
        self.nn_archive_cfg = nn_archive_cfg
        self._pipeline: dai.Pipeline | None = None
        self.device: dai.Device | None = None
        self.device_platform: str | None = None
        self.nn_archive: dai.NNArchive | None = None
        self.input_spec: TensorSpec | None = None
        self.output_specs: tuple[TensorSpec, ...] = ()
        self.model_platform: str | None = None
        self._input_queue: Any = None
        self._output_queue: Any = None
        self._passthrough: Any = None

    def setup(self) -> ModelSpec:
        """Set up the DepthAI pipeline and resolve model spec."""
        if self._pipeline is None:
            self.device, self.device_platform = self._setup_device()
            (
                self.nn_archive,
                self.input_spec,
                self.output_specs,
                self.model_platform,
            ) = self._load_nn_archive()

            self._pipeline = dai.Pipeline(self.device)
            nn_node = self._pipeline.create(dai.node.NeuralNetwork)
            nn_node.setNNArchive(self.nn_archive)

            self._input_queue = nn_node.input.createInputQueue()
            self._output_queue = nn_node.out.createOutputQueue()
            self._passthrough = nn_node.passthrough.createOutputQueue()

            self._pipeline.start()

        if self.model_spec is not None:
            return self.model_spec

        return self._set_model_spec(self._resolve_model_spec())

    def _resolve_platform_name(self) -> str:
        """Resolve the backend platform from device and model
        metadata."""
        if (
            self.device_platform
            and self.model_platform
            and self.device_platform != self.model_platform
        ):
            raise ValueError(
                f"Platform mismatch: Device platform is {self.device_platform}, "
                f"but model was converted for {self.model_platform}."
            )
        platform = self.device_platform or self.model_platform
        if not platform:
            raise ValueError(
                "Could not determine platform from device or model."
            )
        return platform

    def _resolve_model_spec(self) -> ModelSpec:
        """Resolve model input dimensions from the loaded archive."""
        if self.input_spec is None or self.input_spec.shape is None:
            raise ValueError("Invalid input shape information.")

        shape = self.input_spec.shape
        platform_name = self._resolve_platform_name()

        if platform_name == "RVC2":
            # RVC2 uses NCHW format: [batch, channels, height, width]
            if len(shape) != 4 or self.input_spec.layout != "NCHW":
                raise ValueError(
                    f"Unexpected input shape for RVC2: {shape}. Expected input shape in NCHW format."
                )
        # RVC4 uses NHWC format: [batch, height, width, channels]
        elif len(shape) != 4 or self.input_spec.layout != "NHWC":
            raise ValueError(
                f"Unexpected input shape for {platform_name}: {shape}. Expected input shape in NHWC format."
            )
        return ModelSpec(input=self.input_spec, outputs=self.output_specs)

    def infer_once(self, img: np.ndarray) -> DepthAIEngineOutput:
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
        model_spec = self._get_model_spec()

        assert img.shape[0] == model_spec.height
        assert img.shape[1] == model_spec.width

        if img.ndim == 2:
            channel_count = 1
        elif img.ndim == 3 and img.shape[2] == 1:
            channel_count = 1
            img = img[:, :, 0]
        elif img.ndim == 3:
            channel_count = img.shape[2]
        else:
            raise ValueError(
                f"Unsupported image shape for DepthAI inference: {img.shape}."
            )

        if channel_count == 1:
            img_frame_type = dai.ImgFrame.Type.GRAY8
            img_for_device = img
        elif self._resolve_platform_name() == "RVC2":
            img_frame_type = dai.ImgFrame.Type.BGR888p
            img_for_device = np.transpose(img, (2, 0, 1))
        else:
            img_frame_type = dai.ImgFrame.Type.BGR888i
            img_for_device = img

        new_input = dai.ImgFrame()
        new_input.setFrame(img_for_device)
        new_input.setWidth(model_spec.width)
        new_input.setHeight(model_spec.height)
        new_input.setType(img_frame_type)
        self._input_queue.send(new_input)

        return DepthAIEngineOutput(self._output_queue.get())

    def vis_frame(self) -> np.ndarray:
        """Get visualization frame from passthrough.

        Returns
        -------
        np.ndarray
            Visualization frame.
        """
        if self._passthrough is None:
            raise RuntimeError("Visualization frame is unavailable.")
        return self._passthrough.get().getCvFrame()  # type: ignore

    def close(self) -> None:
        """Tear down the DepthAI pipeline."""
        if self._passthrough is not None:
            self._passthrough.close()
        if self._output_queue is not None:
            self._output_queue.close()
        if self._pipeline is not None:
            self._pipeline.stop()
        if self.device is not None:
            self.device.close()

        self._pipeline = None
        self.device = None
        self.device_platform = None
        self.nn_archive = None
        self.input_spec = None
        self.output_specs = ()
        self.model_platform = None
        self._input_queue = None
        self._output_queue = None
        self._passthrough = None
        self.model_spec = None

    def _setup_device(self) -> tuple[dai.Device, str]:
        """Set up and connect to a DepthAI device.

        Returns
        -------
        tuple[dai.Device, str]
            Connected device and its platform name.
        """

        logger.info("Setting up device connection...")

        try:
            if self.device_ip:
                device_info = dai.DeviceInfo(self.device_ip)
                device = dai.Device(device_info)
            else:
                device = dai.Device()

            platform = device.getPlatform()
            platform_name = platform.name

            logger.info(
                f"Connected to [{platform.name}]: Name: {device.getDeviceName()} - IP: {device.getDeviceInfo().name} - ID: {device.getDeviceInfo().deviceId}"
            )

        except Exception as e:
            logger.error(f"Failed to connect to device: {e}")
            raise

        return device, platform_name

    def _load_nn_archive(
        self,
    ) -> tuple[
        dai.NNArchive,
        TensorSpec,
        tuple[TensorSpec, ...],
        str | None,
    ]:
        """Load the model from an NNArchive.

        Returns
        -------
        tuple[dai.NNArchive, dict, str | None]
            Loaded NNArchive, input info, and inferred platform.
        """

        logger.info(f"Loading NNArchive model from: {self.model_path!s}")

        if not self.model_path.exists():  # type: ignore
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            nn_archive = dai.NNArchive(self.model_path)  # type: ignore
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        input_spec: TensorSpec | None = None
        output_specs: tuple[TensorSpec, ...] = ()
        infered_platform = None
        try:
            inputs = nn_archive.getConfig().model.inputs
            if len(inputs) != 1:
                raise NotImplementedError(
                    "Only single-input NNArchive models are supported in luxonis-eval."
                )

            input_meta = inputs[0]
            input_shape = tuple(input_meta.shape)
            input_layout = getattr(input_meta, "layout", None)
            if input_layout is not None and not isinstance(input_layout, str):
                input_layout = getattr(input_layout, "name", None) or getattr(
                    input_layout, "value", None
                )
            input_spec = TensorSpec(
                name=getattr(input_meta, "name", "input"),
                shape=input_shape,
                dtype=str(getattr(input_meta, "dtype", None))
                if getattr(input_meta, "dtype", None) is not None
                else None,
                layout=input_layout,
            )
            logger.info(f"Model input shape: {input_shape}")
            if input_layout == "NHWC":
                infered_platform = "RVC4"
            elif input_layout == "NCHW":
                infered_platform = "RVC2"

            outputs = getattr(nn_archive.getConfig().model, "outputs", [])
            output_specs = tuple(
                TensorSpec(
                    name=getattr(output_meta, "name", f"output_{idx}"),
                    shape=tuple(output_meta.shape)
                    if getattr(output_meta, "shape", None) is not None
                    else None,
                    dtype=str(getattr(output_meta, "dtype", None))
                    if getattr(output_meta, "dtype", None) is not None
                    else None,
                    layout=(
                        getattr(output_meta, "layout", None)
                        if isinstance(
                            getattr(output_meta, "layout", None), str
                        )
                        or getattr(output_meta, "layout", None) is None
                        else getattr(
                            getattr(output_meta, "layout", None), "name", None
                        )
                        or getattr(
                            getattr(output_meta, "layout", None), "value", None
                        )
                    ),
                )
                for idx, output_meta in enumerate(outputs)
            )
        except AttributeError:
            logger.warning("Could not extract input shape from model")

        if input_spec is None:
            raise ValueError(
                "Could not extract model input spec from NNArchive."
            )

        return (
            nn_archive,
            input_spec,
            output_specs,
            infered_platform,
        )
