from pathlib import Path
from typing import Any

import depthai as dai
import numpy as np
from depthai import ADatatype
from loguru import logger

from luxonis_eval.engines.base_engine import BaseEngine


class DepthAIEngine(BaseEngine, register_name="depthai"):
    """DepthAI inference engine."""

    def __init__(
        self, model_path: str, *, device_ip: str | None = None, **kwargs: Any
    ) -> None:
        """Initialize the DepthAI inference engine.

        Parameters
        ----------
        model_path : str
            Path to the model file.
        device_ip : str | None, optional
            IP address of the DepthAI device.
        **kwargs : Any
            Additional engine configuration.
        """
        self.model_path = (
            model_path if isinstance(model_path, Path) else Path(model_path)
        )
        self.device_ip = device_ip
        self._pipeline = None
        self.setup()
        super().__init__(model_path=model_path, **kwargs)

    def load_nn_archive(self) -> tuple[dai.NNArchive, dict, str | None]:
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

    def setup_device(self) -> tuple[dai.Device, str]:
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

    def setup(self) -> None:
        """Set up the DepthAI pipeline."""
        self.device, self.device_platform = self.setup_device()
        self.nn_archive, self.input_info, self.model_platform = (
            self.load_nn_archive()
        )

        self._pipeline = dai.Pipeline(self.device)
        nn_node = self._pipeline.create(dai.node.NeuralNetwork)
        nn_node.setNNArchive(self.nn_archive)

        self._input_queue = nn_node.input.createInputQueue()
        self._output_queue = nn_node.out.createOutputQueue()
        self._passthrough = nn_node.passthrough.createOutputQueue()

        self._pipeline.start()

    def get_input_shape(self) -> tuple[int, int]:
        """Get model input width and height.

        Returns
        -------
        tuple[int, int]
            Input width and height.
        """
        if not self.input_info or "shape" not in self.input_info:
            raise ValueError("Invalid input shape information.")

        shape = self.input_info["shape"]

        if self.get_platform_name() == "RVC2":
            # RVC2 uses NCHW format: [batch, channels, height, width]
            if len(shape) == 4:
                height, width = shape[2], shape[3]
            else:
                raise ValueError(
                    f"Unexpected input shape for RVC2: {shape}. Expected input shape in NCHW format."
                )
        # RVC4 uses NHWC format: [batch, height, width, channels]
        elif len(shape) == 4:
            height, width = shape[1], shape[2]
        else:
            raise ValueError(
                f"Unexpected input shape for RVC4: {shape}. Expected input shape in NHWC format."
            )

        return width, height

    def get_platform_name(self) -> str:
        """Get the platform name for DepthAI engine.

        Returns
        -------
        str
            Platform name.
        """
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
        assert img.shape[0] == self.height
        assert img.shape[1] == self.width

        if self.get_platform_name() == "RVC2":
            img_frame_type = dai.ImgFrame.Type.BGR888p
            img_for_device = np.transpose(img, (2, 0, 1))
        else:
            img_frame_type = dai.ImgFrame.Type.BGR888i
            img_for_device = img

        new_input = dai.ImgFrame()
        new_input.setFrame(img_for_device)
        new_input.setWidth(self.width)
        new_input.setHeight(self.height)
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
