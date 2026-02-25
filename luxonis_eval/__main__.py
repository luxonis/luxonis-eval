from importlib.metadata import version
from typing import Literal

import numpy as np
from cyclopts import App, Group
from loguru import logger
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.loaders import BaseLoader, LuxonisLoader
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from luxonis_eval import BaseEngine
from luxonis_eval.registry import (
    ENGINES_REGISTRY,
    TASKS_REGISTRY,
    from_registry,
)
from luxonis_eval.utils.config import EvalConfig
from luxonis_eval.utils.utils import get_class_mapping, make_report_table

app = App(
    help="Luxonis Eval CLI",
    version=lambda: f"LuxonisEval v{version('luxonis_eval')}",
)
app.meta.group_parameters = Group("Global Parameters", sort_key=0)
app["--help"].group = app.meta.group_parameters
app["--version"].group = app.meta.group_parameters


def eval_setup(eval_cfg: EvalConfig) -> tuple[BaseEngine, BaseLoader]:
    """Setup evaluation configuration.

    Parameters
    ----------
    eval_cfg : EvalConfig
        Evaluation configuration.

    Returns
    -------
    tuple[BaseEngine, BaseLoader]
        The initialized inference engine and the initialized dataloader.
    """
    logger.info("Setting up evaluation configuration.")

    # -------------------------------------------------------------------------
    # Inference engine initialization
    # -------------------------------------------------------------------------
    infer_engine = from_registry(
        ENGINES_REGISTRY,
        eval_cfg.engine_cfg.name,
        eval_cfg.engine_cfg.model_path,
        **eval_cfg.engine_cfg.params,
    )

    # -------------------------------------------------------------------------
    # Dataset and dataloader initialization
    # -------------------------------------------------------------------------
    # TODO: This code is placeholder, we need to implement a proper way to handle different datasets. The datasets should always inherit from BaseDataset (from luxonis_ml).
    dataset = LuxonisDataset(eval_cfg.dataset_cfg.name)
    augmentation_config = []
    if eval_cfg.dataset_cfg.preprocessing.normalize.active:
        augmentation_config.append(
            {
                "name": "Normalize",
                "params": eval_cfg.dataset_cfg.preprocessing.normalize.params,
            }
        )
    # TODO: This code is placeholder, we need to implement a proper way to handle different loaders based on the dataset and model type. The loaders should always inherit from BaseLoader (from luxonis_ml).
    dataloader = LuxonisLoader(
        dataset,
        view=eval_cfg.dataset_cfg.params.get("view", ["val"]),  # type: ignore
        augmentation_config=augmentation_config,
        height=infer_engine.height,
        width=infer_engine.width,
        keep_aspect_ratio=eval_cfg.dataset_cfg.preprocessing.keep_aspect_ratio,
        color_space=eval_cfg.dataset_cfg.preprocessing.color_space,
    )
    logger.info(
        f"Dataset loaded with {len(dataloader)} samples with images of size {infer_engine.height}x{infer_engine.width}."
    )

    # -------------------------------------------------------------------------
    # Configuration compatibility checks
    # -------------------------------------------------------------------------
    if (
        eval_cfg.engine_cfg.name == "depthai"
        and eval_cfg.dataset_cfg.preprocessing.normalize.active
    ):
        logger.warning(
            "Normalization is usually part of the model's preprocessing pipeline in DepthAI. Consider disabling normalization in the dataset config."
        )
    if (
        eval_cfg.engine_cfg.name == "depthai"
        and eval_cfg.dataset_cfg.preprocessing.color_space == "RGB"
    ):
        logger.warning(
            "Color space is set to RGB in the dataset config. DepthAI expects BGR color space."
        )

    return infer_engine, dataloader


def eval_run(
    eval_cfg: EvalConfig,
    infer_engine: BaseEngine,
    dataloader: BaseLoader,
) -> None:
    """Run evaluation with the given configuration and dataloader.

    Parameters
    ----------
    eval_cfg : EvalConfig
        Evaluation configuration.
    infer_engine : BaseEngine
        The inference engine to use for evaluation.
    dataloader : BaseLoader
        The dataloader to use for evaluation.
    """
    # -------------------------------------------------------------------------
    # Task initialization
    # -------------------------------------------------------------------------
    task_name = eval_cfg.task_cfg.name
    if not task_name:
        raise ValueError("Task configuration must include a 'name' key.")

    try:
        task = from_registry(
            TASKS_REGISTRY, task_name, **eval_cfg.task_cfg.params
        )
        logger.info(f"Loading inference task: {task_name}")
    except KeyError as e:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {list(TASKS_REGISTRY._module_dict)}"
        ) from e

    ldf_class_map, class_map, class_index_map = get_class_mapping(
        dataloader, **eval_cfg.dataset_cfg.params
    )

    # -------------------------------------------------------------------------
    # Parser, Metrics, and Visualizer initialization
    # -------------------------------------------------------------------------
    task.build_parser(**eval_cfg.parser_cfg.model_dump())
    task.build_metrics(**eval_cfg.metrics_cfg.model_dump())
    task.build_throughput_metric()
    if eval_cfg.visualizer_cfg and eval_cfg.visualizer_cfg.visualize:
        task.build_visualizer(**eval_cfg.visualizer_cfg.model_dump())

    backend: str = eval_cfg.engine_cfg.name

    # -------------------------------------------------------------------------
    # Main evaluation loop
    # -------------------------------------------------------------------------
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            ptask = progress.add_task(
                f"Running {backend.upper()} inference ({task.__class__.__name__})...",
                total=len(dataloader),
            )

            for sample in dataloader:
                img: np.ndarray = sample[0]  # type: ignore
                target = sample[1]

                raw_output = infer_engine.infer_once(img)
                predictions = task.parse_predictions(
                    raw_output,
                    class_map=class_map,
                    **eval_cfg.parser_cfg.params,
                )

                metric_ctx = task.metric_extra_context(
                    width=infer_engine.width,
                    height=infer_engine.height,
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

                if (
                    eval_cfg.visualizer_cfg
                    and eval_cfg.visualizer_cfg.visualize
                ):
                    task.visualizer.visualize(
                        predictions,
                        infer_engine.vis_frame(),
                        **eval_cfg.visualizer_cfg.params,
                    )

                progress.update(ptask, advance=1)
    finally:
        infer_engine.teardown()

    # -------------------------------------------------------------------------
    # Results computation and reporting
    # -------------------------------------------------------------------------
    results = [metric.compute() for metric in task.metrics]
    tp = task.throughput_metric.compute()

    table = make_report_table(
        backend=backend,
        task_name=task.__class__.__name__,
        device=infer_engine.platform_name,
        tp=tp,
        results=results,
    )

    logger.info(f"\n{table}")


@app.command()
def eval(
    dataset_name: str | None = None,
    nn_archive: str | None = None,
    onnx: str | None = None,
    backend: Literal["depthai", "onnx"] | None = None,
    device_ip: str | None = None,
    config: str | None = None,
):
    """Run evaluation on a dataset using a specified neural network.

    Parameters
    ----------
    dataset_name : str | None, optional
        Name of the dataset to evaluate on.
    nn_archive : str | None, optional
        Path to the neural network NNArchive file. Required if backend is set to 'depthai' or 'all'.
    onnx : str | None, optional
        Path to the ONNX model file, required if backend is 'onnx' or 'all'. Required if backend is set to 'onnx' or 'all'.
    backend : Literal["depthai", "onnx"] | None, optional
        Backend to use for inference.
    device_ip : str | None, optional
        IP address of the device to connect to. Only applicable for RVC4 devices.
    config : str | None, optional
        Path to the evaluation configuration file in YAML format.
    """
    overrides = {}
    if dataset_name is not None:
        overrides["dataset_name"] = dataset_name
    if nn_archive is not None:
        overrides["nn_archive"] = nn_archive
    if onnx is not None:
        overrides["onnx"] = onnx
    if backend is not None:
        overrides["backend"] = backend
    if device_ip is not None:
        overrides["device_ip"] = device_ip

    eval_cfg = EvalConfig.get_config(cfg=config, overrides=overrides)

    infer_engine, dataloader = eval_setup(eval_cfg)

    eval_run(eval_cfg, infer_engine, dataloader)


if __name__ == "__main__":
    app()
