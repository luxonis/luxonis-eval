from pathlib import Path
from typing import Literal

import depthai as dai
import numpy as np
from loguru import logger
from luxonis_ml.data.loaders import BaseLoader, LuxonisLoader
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from luxonis_eval.backend_engine import DepthAIBackend, OnnxBackend
from luxonis_eval.tasks import TASKS_REGISTRY
from luxonis_eval.utils import (
    get_class_index_mapping,
    get_dataset_class_mapping,
    get_onnx_input_info,
    make_report_table,
)


class Inferer:
    """Run model inference and evaluation using DepthAI or ONNX backends."""

    def __init__(
        self,
        nn_archive_path: Path,
        onnx_path: Path | None = None,
        backend: Literal["depthai", "onnx", "all"] = "depthai",
        device_ip: str | None = None,
    ):
        """Initialize the inferer.

        Parameters
        ----------
        nn_archive_path : Path
            Path to the NNArchive model.
        onnx_path : Path | None, optional
            Path to the ONNX model.
        backend : Literal["depthai", "onnx", "all"], optional
            Backend selection.
        device_ip : str | None, optional
            IP address of the DepthAI device.
        """
        self.nn_archive_path = nn_archive_path
        self.onnx_path = onnx_path
        self.backend = backend
        self.device_ip = device_ip

        device_platform = None
        if self.backend in ("depthai", "all"):
            self.device, device_platform = self.setup_device()
            self.nn_archive, input_info, model_platform = (
                self.load_nn_archive()
            )
            self.platform_name = self.resolve_platform(
                device_platform, model_platform
            )
        else:
            self.platform_name = "Host CPU/GPU"
            input_info = get_onnx_input_info(self.onnx_path)

        self.width, self.height = self.get_input_shape(input_info)

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

    def load_nn_archive(self) -> tuple[dai.NNArchive, dict, str | None]:
        """Load the model from an NNArchive.

        Returns
        -------
        tuple[dai.NNArchive, dict, str | None]
            Loaded NNArchive, input info, and inferred platform.
        """

        logger.info(f"Loading NNArchive model from: {self.nn_archive_path!s}")

        if not self.nn_archive_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.nn_archive_path}"
            )

        try:
            nn_archive = dai.NNArchive(self.nn_archive_path)
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

    def resolve_platform(
        self,
        device_platform: str | None,
        model_platform: str | None,
    ) -> str:
        """Resolve the platform from device and model information.

        Parameters
        ----------
        device_platform : str | None
            Platform name reported by the connected device.
        model_platform : str | None
            Platform name inferred from the model.

        Returns
        -------
        str
            Resolved platform name.
        """

        if (
            device_platform
            and model_platform
            and device_platform != model_platform
        ):
            raise ValueError(
                f"Platform mismatch: Device platform is {device_platform}, "
                f"but model was converted for {model_platform}."
            )
        platform = device_platform or model_platform
        if not platform:
            raise ValueError(
                "Could not determine platform from device or model."
            )
        return platform

    def get_input_shape(self, input_info: dict) -> tuple[int, int]:
        """Get model input width and height from input info.

        Parameters
        ----------
        input_info : dict
            Dictionary containing input shape information.

        Returns
        -------
        tuple[int, int]
            Input width and height.
        """
        logger.warning(f"Input info: {input_info}")
        if not input_info or "shape" not in input_info:
            raise ValueError("Invalid input shape information.")

        shape = input_info["shape"]

        if self.backend in ["depthai", "all"]:
            if self.platform_name == "RVC2":
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
        # ONNX models use the NCHW format
        elif len(shape) == 4:
            height, width = shape[2], shape[3]
        else:
            raise ValueError(
                f"Unexpected input shape for ONNX: {shape}. Expected input shape in NCHW format."
            )

        return width, height

    @staticmethod
    def get_class_mapping(
        dataloader: BaseLoader,
    ) -> tuple[dict, dict | None]:
        """Get native class map and optional class index mapping.

        Parameters
        ----------
        dataloader : BaseLoader
            Dataloader to extract class mappings from.

        Returns
        -------
        tuple[dict, dict | None]
            Native class map and class index map (if available).
        """

        # TODO: Support loaders inheriting from BaseLoader. Find a way to get class map from them. If its not provided, go though the dataset, which if it inherits from LuxonisDataset (it should), we can get the native classes from there.
        if isinstance(dataloader, LuxonisLoader):
            ldf_class_map = dataloader.classes[""]
            ldf_class_map = {v: k for k, v in ldf_class_map.items()}
        else:
            raise NotImplementedError(
                "Only LuxonisLoader is currently supported."
            )

        # TODO: Find a better way to determine dataset type/name because the dataset name can be arbitrary and may not contain 'imagenet' or 'coco'.
        if "imagenet" in dataloader.dataset.dataset_name:
            native_class_map = get_dataset_class_mapping("imagenet")
        elif "coco" in dataloader.dataset.dataset_name:
            native_class_map = get_dataset_class_mapping("coco")
        else:
            native_class_map = {}

        class_index_map = None
        if native_class_map:
            class_index_map = get_class_index_mapping(
                ldf_class_map, native_class_map
            )

        return native_class_map, class_index_map

    def infer(
        self,
        dataloader: BaseLoader,
        *,
        task_name: str,
        task_cfg: dict | None = None,
        metric_cfg: dict | None = None,
        onnx_cfg: dict | None = None,
    ) -> None:
        """Run inference and compute metrics on a dataloader.

        Parameters
        ----------
        dataloader : BaseLoader
            Dataloader to run inference on.
        task_name : str
            Task name registered in the task registry.
        task_cfg : dict | None, optional
            Task configuration.
        metric_cfg : dict | None, optional
            Metric configuration.
        onnx_cfg : dict | None, optional
            ONNX backend configuration.
        """

        task_cfg = task_cfg or {}
        metric_cfg = metric_cfg or {}
        onnx_cfg = onnx_cfg or {}

        native_class_map, class_index_map = self.get_class_mapping(dataloader)

        try:
            task_cls = TASKS_REGISTRY[task_name]
        except KeyError as e:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Available tasks: {list(TASKS_REGISTRY._module_dict)}"
            ) from e
        task = task_cls(**task_cfg)

        metric = task.build_metric(**metric_cfg)
        metric.reset()

        if self.backend == "depthai":
            be = DepthAIBackend(self)
        else:
            be = OnnxBackend(self, **onnx_cfg)

        be.setup()

        try:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            ) as progress:
                ptask = progress.add_task(
                    f"Running {self.backend.upper()} inference ({task.NAME})...",
                    total=len(dataloader),
                )

                for sample in dataloader:
                    img: np.ndarray = sample[0]  # type: ignore
                    target = sample[1][task.target_key()]

                    raw_output = be.infer_once(img)
                    predictions = task.parse_predictions(
                        raw_output,
                        backend=self.backend,
                        native_class_map=native_class_map,
                    )

                    predictions_m, target_m, kwargs = (
                        task.metric_update_payload(
                            predictions=predictions,
                            target=target,
                            width=self.width,
                            height=self.height,
                            native_class_map=native_class_map,
                            class_index_map=class_index_map,
                        )
                    )
                    metric.update(
                        predictions=predictions_m, target=target_m, **kwargs
                    )

                    progress.update(ptask, advance=1)
        finally:
            be.teardown()

        results = metric.compute()
        tp = metric.throughput()

        table = make_report_table(
            backend=self.backend,
            task=task,
            device=self.platform_name,
            tp=tp,
            results=results,
        )

        logger.info(f"\n{table}")
