from typing import Any

import depthai as dai
import numpy as np
from depthai import ADatatype
from loguru import logger
from luxonis_ml.typing import PathType

from luxonis_eval.engines.base_engine import BaseEngine, ModelSpec


class DepthAIEngine(BaseEngine, register_name="depthai"):
    """DepthAI inference engine."""

    def __init__(
        self,
        model_path: PathType,
        device_ip: str | None = None,
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
        self._pipeline: dai.Pipeline | None = None
        self.device: dai.Device | None = None
        self.device_platform: str | None = None
        self.nn_archive: dai.NNArchive | None = None
        self.input_info: dict[str, Any] = {}
        self.model_platform: str | None = None
        self._input_queue: Any = None
        self._output_queue: Any = None
        self._passthrough: Any = None

    def setup(self) -> ModelSpec:
        """Set up the DepthAI pipeline and resolve model spec."""
        if self._pipeline is None:
            self.device, self.device_platform = self._setup_device()
            self.nn_archive, self.input_info, self.model_platform = (
                self._load_nn_archive()
            )

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
        if not self.input_info or "shape" not in self.input_info:
            raise ValueError("Invalid input shape information.")

        shape = self.input_info["shape"]
        platform_name = self._resolve_platform_name()

        if platform_name == "RVC2":
            # RVC2 uses NCHW format: [batch, channels, height, width]
            if len(shape) != 4:
                raise ValueError(
                    f"Unexpected input shape for RVC2: {shape}. Expected input shape in NCHW format."
                )
            height, width = shape[2], shape[3]
        else:
            # RVC4 uses NHWC format: [batch, height, width, channels]
            if len(shape) != 4:
                raise ValueError(
                    f"Unexpected input shape for {platform_name}: {shape}. Expected input shape in NHWC format."
                )
            height, width = shape[1], shape[2]

        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError(
                f"DepthAI input shape must be statically defined. Got {shape}."
            )

        return ModelSpec(width=width, height=height)

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

        return self._output_queue.get()

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
        self.input_info = {}
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

    def _load_nn_archive(self) -> tuple[dai.NNArchive, dict, str | None]:
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

        input_info = {}
        infered_platform = None
        try:
            inputs = nn_archive.getConfig().model.inputs
            if inputs:
                input_shape = inputs[0].shape
                input_info = {
                    "shape": input_shape,
                    "name": inputs[0].name
                    if hasattr(inputs[0], "name")
                    else "input",
                }
                logger.info(f"Model input shape: {input_shape}")
                if inputs[0].layout and inputs[0].layout == "NHWC":
                    infered_platform = "RVC4"
                elif inputs[0].layout and inputs[0].layout == "NCHW":
                    infered_platform = "RVC2"
        except AttributeError:
            logger.warning("Could not extract input shape from model")

        return nn_archive, input_info, infered_platform
