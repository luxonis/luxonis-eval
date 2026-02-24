from pathlib import Path

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

from luxonis_eval.registry import (
    ENGINES_REGISTRY,
    TASKS_REGISTRY,
    from_registry,
)
from luxonis_eval.utils.utils import (
    get_class_index_mapping,
    get_dataset_class_mapping,
    get_onnx_input_info,
    make_report_table,
)


class Inferer:
    """Run model inference and evaluation using DepthAI or ONNX backends."""

    def __init__(
        self,
        model_path: Path,
        backend: str,
        device_ip: str | None = None,
    ):
        """Initialize the inferer.

        Parameters
        ----------
        model_path : Path
            Path to the model file.
        backend : str
            Backend selection.
        device_ip : str | None, optional
            IP address of the DepthAI device.
        """
        self.model_path = model_path
        self.backend = backend
        self.device_ip = device_ip

        device_platform = None
        if self.backend == "depthai":
            self.device, device_platform = self.setup_device()
            self.nn_archive, input_info, model_platform = (
                self.load_nn_archive()
            )
            self.platform_name = self.resolve_platform(
                device_platform, model_platform
            )
        else:
            self.platform_name = "Host CPU/GPU"
            input_info = get_onnx_input_info(self.model_path)

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
        if not input_info or "shape" not in input_info:
            raise ValueError("Invalid input shape information.")

        shape = input_info["shape"]

        if self.backend == "depthai":
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
        **kwargs,
    ) -> tuple[dict, dict, dict | None]:
        """Get native class map and optional class index mapping.

        Parameters
        ----------
        dataloader : BaseLoader
            Dataloader to extract class mappings from.
        **kwargs
            Additional dataset-specific parameters.

        Returns
        -------
        tuple[dict, dict, dict | None]
            LDF class map, native class map and class index map (if available).
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
            native_class_map = kwargs.get("class_mapping", {})

        class_index_map = None
        if native_class_map:
            class_index_map = get_class_index_mapping(
                ldf_class_map, native_class_map
            )

        return ldf_class_map, native_class_map, class_index_map

    def infer(
        self,
        dataloader: BaseLoader,
        *,
        dataset_cfg: dict,
        task_cfg: dict,
        parser_cfg: dict,
        metrics_cfg: dict,
        visualizer_cfg: dict,
        engine_cfg: dict,
    ) -> None:
        """Run inference and compute metrics on a dataloader.

        Parameters
        ----------
        dataloader : BaseLoader
            Dataloader to run inference on.
        dataset_cfg : dict
            Dataset configuration.
        task_cfg : dict
            Task configuration.
        parser_cfg : dict
            Parser configuration.
        metrics_cfg : dict
            Metrics configuration.
        visualizer_cfg : dict
            Visualizer configuration.
        engine_cfg : dict
            Engine configuration.
        """
        task_name = task_cfg.get("name")
        if not task_name:
            raise ValueError("Task configuration must include a 'name' key.")

        try:
            task = from_registry(
                TASKS_REGISTRY, task_name, **task_cfg.get("params", {})
            )
            logger.info(f"Loading inference task: {task_name}")
        except KeyError as e:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Available tasks: {list(TASKS_REGISTRY._module_dict)}"
            ) from e

        ldf_class_map, class_map, class_index_map = self.get_class_mapping(
            dataloader, **dataset_cfg.get("params", {})
        )

        task.build_parser(**parser_cfg)
        task.build_metrics(**metrics_cfg)
        task.build_throughput_metric()
        if visualizer_cfg.get("visualize"):
            task.build_visualizer(**visualizer_cfg)

        infer_engine = from_registry(
            ENGINES_REGISTRY,
            self.backend,
            self,
            **engine_cfg.get("params", {}),
        )

        infer_engine.setup()
        try:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            ) as progress:
                ptask = progress.add_task(
                    f"Running {self.backend.upper()} inference ({task.__class__.__name__})...",
                    total=len(dataloader),
                )

                for sample in dataloader:
                    img: np.ndarray = sample[0]  # type: ignore
                    target = sample[1]

                    raw_output = infer_engine.infer_once(img)
                    predictions = task.parse_predictions(
                        raw_output,
                        class_map=class_map,
                        **parser_cfg.get("params", {}),
                    )

                    metric_ctx = task.metric_extra_context(
                        width=self.width,
                        height=self.height,
                        ldf_class_map=ldf_class_map,
                        class_map=class_map,
                        class_index_map=class_index_map,
                    )
                    for metric in task.metrics:
                        metric.update(
                            predictions=predictions,
                            target=target,
                            **metric_ctx,
                        )
                    task.throughput_metric.update()

                    if visualizer_cfg.get("visualize"):
                        task.visualizer.visualize(
                            predictions,
                            infer_engine.vis_frame(),
                            **visualizer_cfg.get("params", {}),
                        )

                    progress.update(ptask, advance=1)
        finally:
            infer_engine.teardown()

        results = [metric.compute() for metric in task.metrics]
        tp = task.throughput_metric.compute()

        table = make_report_table(
            backend=self.backend,
            task_name=task.__class__.__name__,
            device=self.platform_name,
            tp=tp,
            results=results,
        )

        logger.info(f"\n{table}")
