import time
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

from luxonis_eval import (
    BaseEngine,
    BaseEvalLoader,
    BaseMetric,
    BaseParser,
    BaseVisualizer,
)
from luxonis_eval.metrics import ThroughputMetric
from luxonis_eval.registry import (
    DATALOADERS_REGISTRY,
    ENGINES_REGISTRY,
    METRICS_REGISTRY,
    PARSERS_REGISTRY,
    VISUALIZERS_REGISTRY,
    from_registry,
)
from luxonis_eval.utils.config import EvalConfig
from luxonis_eval.utils.utils import (
    get_class_mapping,
    get_metric_ctx,
    get_model_name,
    make_report_table,
)

app = App(
    help="Luxonis Eval CLI",
    version=lambda: f"LuxonisEval v{version('luxonis_eval')}",
)
app.meta.group_parameters = Group("Global Parameters", sort_key=0)
app["--help"].group = app.meta.group_parameters
app["--version"].group = app.meta.group_parameters


def eval_setup(
    eval_cfg: EvalConfig,
) -> tuple[
    BaseEngine,
    BaseEvalLoader | LuxonisLoader,
    BaseParser,
    list[BaseMetric],
    ThroughputMetric,
    BaseVisualizer | None,
]:
    """Setup evaluation configuration.

    Parameters
    ----------
    eval_cfg : EvalConfig
        Evaluation configuration.

    Returns
    -------
    tuple[BaseEngine, BaseEvalLoader | LuxonisLoader, BaseParser, list[BaseMetric], ThroughputMetric, BaseVisualizer | None]
        The initialized inference engine, dataloader, parser, metrics, throughput metric, and visualizer (if enabled).
    """
    logger.info("Setting up evaluation configuration.")

    # -------------------------------------------------------------------------
    # Inference engine initialization
    # -------------------------------------------------------------------------
    try:
        infer_engine = from_registry(
            ENGINES_REGISTRY,
            eval_cfg.engine.name,
            eval_cfg.engine.model_path,
            **eval_cfg.engine.params,
        )
        assert isinstance(infer_engine, BaseEngine), (
            f"{eval_cfg.engine.name} engine must be an instance of BaseEngine."
        )
        logger.info(f"{eval_cfg.engine.name} inference engine initialized.")
    except KeyError as e:
        raise ValueError(
            f"Unknown engine: {eval_cfg.engine.name}. "
            f"Available engines: {list(ENGINES_REGISTRY._module_dict)}"
        ) from e

    # -------------------------------------------------------------------------
    # Dataset and dataloader initialization
    # -------------------------------------------------------------------------
    try:
        if eval_cfg.loader.name == "LuxonisLoader":
            dataset_name: str = eval_cfg.loader.params.get("dataset_name")  # type: ignore
            dataset = LuxonisDataset(dataset_name)
            augmentation_config = []
            if eval_cfg.loader.preprocessing.normalize.active:
                augmentation_config.append(
                    {
                        "name": "Normalize",
                        "params": eval_cfg.loader.preprocessing.normalize.params,
                    }
                )
            dataloader = LuxonisLoader(
                dataset,
                view=eval_cfg.loader.params.get("view", ["val"]),  # type: ignore
                augmentation_config=augmentation_config,
                height=infer_engine.height,
                width=infer_engine.width,
                keep_aspect_ratio=eval_cfg.loader.preprocessing.keep_aspect_ratio,
                color_space=eval_cfg.loader.preprocessing.color_space,
            )

            dataloader.get_class_mapping = types.MethodType(  # type: ignore
                get_class_mapping, dataloader
            )
        else:
            dataloader = from_registry(
                DATALOADERS_REGISTRY,
                eval_cfg.loader.name,
                **eval_cfg.loader.params,
            )
            assert isinstance(dataloader, BaseEvalLoader), (
                f"{eval_cfg.loader.name} custom dataloader must be an instance of BaseEvalLoader."
            )
        logger.info(f"{eval_cfg.loader.name} dataloader initialized.")
        logger.info(
            f"Dataset loaded with {len(dataloader)} samples and images of shape {infer_engine.height}x{infer_engine.width}."
        )
    except KeyError as e:
        raise ValueError(
            f"Unknown loader: {eval_cfg.loader.name}. "
            f"Available loaders: {list(DATALOADERS_REGISTRY._module_dict)}"
        ) from e

    # -------------------------------------------------------------------------
    # Parser initialization
    # -------------------------------------------------------------------------
    try:
        parser = from_registry(
            PARSERS_REGISTRY,
            eval_cfg.parser.name,
            **eval_cfg.parser.params,
        )
        assert isinstance(parser, BaseParser), (
            f"{eval_cfg.parser.name} parser must be an instance of BaseParser."
        )
        logger.info(f"{eval_cfg.parser.name} parser initialized.")
    except KeyError as e:
        raise ValueError(
            f"Unknown parser: {eval_cfg.parser.name}. "
            f"Available parsers: {list(PARSERS_REGISTRY._module_dict)}"
        ) from e

    # -------------------------------------------------------------------------
    # Metrics initialization
    # -------------------------------------------------------------------------
    metrics = []
    for metric_cfg in eval_cfg.metrics.metrics:
        metric_name = metric_cfg.name
        try:
            metric = from_registry(
                METRICS_REGISTRY,
                metric_name,
                **metric_cfg.params,
            )
            assert isinstance(metric, BaseMetric), (
                f"{metric_name} metric must be an instance of BaseMetric."
            )
            logger.info(f"{metric_name} metric initialized.")
        except KeyError as e:
            raise ValueError(
                f"Unknown metric: {metric_name}. "
                f"Available metrics: {list(METRICS_REGISTRY._module_dict)}"
            ) from e
        metrics.append(metric)
    if not metrics:
        raise ValueError(
            "At least one metric must be specified in the configuration."
        )

    # -------------------------------------------------------------------------
    # Throughput metric initialization
    # -------------------------------------------------------------------------
    throughput_metric = ThroughputMetric()
    logger.info("Throughput metric initialized.")

    # -------------------------------------------------------------------------
    # Visualizer initialization
    # -------------------------------------------------------------------------
    if eval_cfg.visualizer and eval_cfg.visualizer.visualize:
        try:
            visualizer = from_registry(
                VISUALIZERS_REGISTRY,
                eval_cfg.visualizer.name,
                **eval_cfg.visualizer.params,
            )
            assert isinstance(visualizer, BaseVisualizer), (
                f"{eval_cfg.visualizer.name} visualizer must be an instance of BaseVisualizer."
            )
            logger.info(f"{eval_cfg.visualizer.name} visualizer initialized.")
        except KeyError as e:
            raise ValueError(
                f"Unknown visualizer: {eval_cfg.visualizer.name}. "
                f"Available visualizers: {list(VISUALIZERS_REGISTRY._module_dict)}"
            ) from e
    else:
        visualizer = None
        logger.info("Visualization is disabled.")

    # -------------------------------------------------------------------------
    # Configuration compatibility checks
    # -------------------------------------------------------------------------
    if (
        eval_cfg.engine.name == "depthai"
        and eval_cfg.loader.preprocessing.normalize.active
    ):
        logger.warning(
            "Normalization is usually part of the model's preprocessing pipeline in DepthAI. Consider disabling normalization in the dataset config."
        )
    if (
        eval_cfg.engine.name == "depthai"
        and eval_cfg.loader.preprocessing.color_space == "RGB"
    ):
        logger.warning(
            "Color space is set to RGB in the dataset config. DepthAI expects BGR color space."
        )

    return (
        infer_engine,
        dataloader,
        parser,
        metrics,
        throughput_metric,
        visualizer,
    )


def eval_run(
    eval_cfg: EvalConfig,
) -> None:
    """Run evaluation with the given configuration and dataloader.

    Parameters
    ----------
    eval_cfg : EvalConfig
        Evaluation configuration.
    """

    # -------------------------------------------------------------------------
    # Inference engine and dataloader setup
    # -------------------------------------------------------------------------
    (
        infer_engine,
        dataloader,
        parser,
        metrics,
        throughput_metric,
        visualizer,
    ) = eval_setup(eval_cfg)

    backend: str = eval_cfg.engine.name
    model_name: str = get_model_name(eval_cfg.engine.model_path)

    ldf_class_map, class_map, class_index_map = dataloader.get_class_mapping(  # type: ignore
        **eval_cfg.loader.params
    )

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
                f"Running {backend.upper()} inference ({model_name})...",
                total=len(dataloader),
            )

            for sample in dataloader:
                img: np.ndarray = sample[0]  # type: ignore
                target = sample[1]

                inference_t0 = time.perf_counter()
                raw_output = infer_engine.infer_once(img)
                inference_elapsed = time.perf_counter() - inference_t0

                parsing_t0 = time.perf_counter()
                predictions = parser.parse(
                    raw_output,
                    class_map=class_map,
                    **eval_cfg.parser.params,
                )
                parsing_elapsed = time.perf_counter() - parsing_t0

                metric_update_t0 = time.perf_counter()
                for metric in metrics:
                    base_metric_ctx = eval_cfg.metrics.metrics[
                        metrics.index(metric)
                    ].params
                    metric_ctx = get_metric_ctx(
                        base_ctx=base_metric_ctx,
                        width=infer_engine.width,
                        height=infer_engine.height,
                        ldf_class_map=ldf_class_map,
                        class_map=class_map,
                        class_index_map=class_index_map,
                    )
                    metric.update(
                        predictions=predictions,
                        target=target,
                        **metric_ctx,
                    )
                metric_update_elapsed = time.perf_counter() - metric_update_t0

                throughput_metric.update(
                    inference=inference_elapsed,
                    parsing=parsing_elapsed,
                    metric_update=metric_update_elapsed,
                )

                if visualizer:
                    visualizer.visualize(
                        predictions,
                        target,
                        infer_engine.vis_frame(),
                        class_map=class_map,
                        class_index_map=class_index_map,
                        **eval_cfg.visualizer.params,  # type: ignore
                    )

                progress.update(ptask, advance=1)
    finally:
        infer_engine.teardown()

    # -------------------------------------------------------------------------
    # Results computation and reporting
    # -------------------------------------------------------------------------
    metric_compute_t0 = time.perf_counter()
    results = [metric.compute() for metric in metrics]
    metric_compute_elapsed = time.perf_counter() - metric_compute_t0
    tp = throughput_metric.compute(metric_compute=metric_compute_elapsed)

    table = make_report_table(
        backend=backend,
        model_name=model_name,
        device=infer_engine.platform_name,
        tp=tp,
        results=results,
    )

    logger.warning(
        "Throughput values are end-to-end pipeline measurements and not isolated model-only benchmarks. Lower numbers than modelconverter benchmark results are expected."
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
        overrides["loader.params.dataset_name"] = dataset_name
    if model_path is not None:
        overrides["engine.model_path"] = model_path
    if backend is not None:
        overrides["engine.name"] = backend
    if device_ip is not None:
        overrides["engine.params.device_ip"] = device_ip

    eval_cfg = EvalConfig.get_config(cfg=config, overrides=overrides)

    eval_run(eval_cfg)


if __name__ == "__main__":
    app()
