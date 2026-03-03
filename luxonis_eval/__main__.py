import types
from importlib.metadata import version
from typing import Literal

import numpy as np
from cyclopts import App, Group
from loguru import logger
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.loaders import LuxonisLoader
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from luxonis_eval import BaseEngine, BaseEvalLoader
from luxonis_eval.registry import (
    DATALOADERS_REGISTRY,
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


def eval_setup(
    eval_cfg: EvalConfig,
) -> tuple[BaseEngine, BaseEvalLoader | LuxonisLoader]:
    """Setup evaluation configuration.

    Parameters
    ----------
    eval_cfg : EvalConfig
        Evaluation configuration.

    Returns
    -------
    tuple[BaseEngine, BaseEvalLoader | LuxonisLoader]
        The initialized inference engine and the initialized dataloader.
    """
    logger.info("Setting up evaluation configuration.")

    # -------------------------------------------------------------------------
    # Inference engine initialization
    # -------------------------------------------------------------------------
    try:
        infer_engine = from_registry(
            ENGINES_REGISTRY,
            eval_cfg.engine_cfg.name,
            eval_cfg.engine_cfg.model_path,
            **eval_cfg.engine_cfg.params,
        )
    except KeyError as e:
        raise ValueError(
            f"Unknown engine: {eval_cfg.engine_cfg.name}. "
            f"Available engines: {list(ENGINES_REGISTRY._module_dict)}"
        ) from e

    # -------------------------------------------------------------------------
    # Dataset and dataloader initialization
    # -------------------------------------------------------------------------
    try:
        if eval_cfg.dataloader_cfg.name == "LuxonisLoader":
            dataset_name: str = eval_cfg.dataloader_cfg.params.get(
                "dataset_name"
            )  # type: ignore
            dataset = LuxonisDataset(dataset_name)
            augmentation_config = []
            if eval_cfg.dataloader_cfg.preprocessing.normalize.active:
                augmentation_config.append(
                    {
                        "name": "Normalize",
                        "params": eval_cfg.dataloader_cfg.preprocessing.normalize.params,
                    }
                )
            dataloader = LuxonisLoader(
                dataset,
                view=eval_cfg.dataloader_cfg.params.get("view", ["val"]),  # type: ignore
                augmentation_config=augmentation_config,
                height=infer_engine.height,
                width=infer_engine.width,
                keep_aspect_ratio=eval_cfg.dataloader_cfg.preprocessing.keep_aspect_ratio,
                color_space=eval_cfg.dataloader_cfg.preprocessing.color_space,
            )

            dataloader.get_class_mapping = types.MethodType(  # type: ignore
                get_class_mapping, dataloader
            )
        else:
            dataloader = from_registry(
                DATALOADERS_REGISTRY,
                eval_cfg.dataloader_cfg.name,
                **eval_cfg.dataloader_cfg.params,
            )
        logger.info(f"{eval_cfg.dataloader_cfg.name} dataloader initialized.")
        logger.info(
            f"Dataset loaded with {len(dataloader)} samples and images of shape {infer_engine.height}x{infer_engine.width}."
        )
    except KeyError as e:
        raise ValueError(
            f"Unknown loader: {eval_cfg.dataloader_cfg.name}. "
            f"Available loaders: {list(DATALOADERS_REGISTRY._module_dict)}"
        ) from e

    # -------------------------------------------------------------------------
    # Configuration compatibility checks
    # -------------------------------------------------------------------------
    if (
        eval_cfg.engine_cfg.name == "depthai"
        and eval_cfg.dataloader_cfg.preprocessing.normalize.active
    ):
        logger.warning(
            "Normalization is usually part of the model's preprocessing pipeline in DepthAI. Consider disabling normalization in the dataset config."
        )
    if (
        eval_cfg.engine_cfg.name == "depthai"
        and eval_cfg.dataloader_cfg.preprocessing.color_space == "RGB"
    ):
        logger.warning(
            "Color space is set to RGB in the dataset config. DepthAI expects BGR color space."
        )

    return infer_engine, dataloader


def eval_run(
    eval_cfg: EvalConfig,
    infer_engine: BaseEngine,
    dataloader: BaseEvalLoader | LuxonisLoader,
) -> None:
    """Run evaluation with the given configuration and dataloader.

    Parameters
    ----------
    eval_cfg : EvalConfig
        Evaluation configuration.
    infer_engine : BaseEngine
        The inference engine to use for evaluation.
    dataloader : BaseEvalLoader | LuxonisLoader
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

    ldf_class_map, class_map, class_index_map = dataloader.get_class_mapping(  # type: ignore
        **eval_cfg.dataloader_cfg.params
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
    config: str,
    dataset_name: str | None = None,
    model_path: str | None = None,
    backend: Literal["depthai", "onnx"] | None = None,
    device_ip: str | None = None,
):
    """Run evaluation on a dataset using a specified neural network.

    Parameters
    ----------
    config : str
        Path to the evaluation configuration file in YAML format.
    dataset_name : str | None, optional
        Name of the dataset to evaluate on.
    model_path : str | None, optional
        Path to the model file (NNArchive or ONNX).
    backend : Literal["depthai", "onnx"] | None, optional
        Backend to use for inference.
    device_ip : str | None, optional
        IP address of the device to connect to. Only applicable for RVC4 devices.
    """
    overrides = {}
    if dataset_name is not None:
        overrides["dataloader_cfg.params.dataset_name"] = dataset_name
    if model_path is not None:
        overrides["engine_cfg.model_path"] = model_path
    if backend is not None:
        overrides["engine_cfg.name"] = backend
    if device_ip is not None:
        overrides["engine_cfg.params.device_ip"] = device_ip

    eval_cfg = EvalConfig.get_config(cfg=config, overrides=overrides)

    infer_engine, dataloader = eval_setup(eval_cfg)

    eval_run(eval_cfg, infer_engine, dataloader)


if __name__ == "__main__":
    app()
