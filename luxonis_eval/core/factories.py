from loguru import logger
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.loaders import LuxonisLoader

from luxonis_eval.config import EvalConfig, EvaluatorConfig
from luxonis_eval.core.runtime import resolve_luxonis_task_name
from luxonis_eval.engines.base_engine import BaseEngine
from luxonis_eval.loaders.base_loader import BaseEvalLoader
from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.parsers.base_parser import BaseParser
from luxonis_eval.registry import (
    DATALOADERS_REGISTRY,
    ENGINES_REGISTRY,
    METRICS_REGISTRY,
    PARSERS_REGISTRY,
    VISUALIZERS_REGISTRY,
    from_registry,
)
from luxonis_eval.visualizers.base_visualizer import BaseVisualizer


def create_engine(cfg: EvalConfig) -> BaseEngine:
    try:
        engine = from_registry(
            ENGINES_REGISTRY,
            cfg.pipeline.engine.name,
            cfg.pipeline.engine.model_path,
            **cfg.pipeline.engine.params,
        )
    except KeyError as e:
        raise ValueError(
            f"Unknown engine: {cfg.pipeline.engine.name}. "
            f"Available engines: {list(ENGINES_REGISTRY._module_dict)}"
        ) from e

    if not isinstance(engine, BaseEngine):
        raise TypeError(
            f"{cfg.pipeline.engine.name} engine must be an instance of BaseEngine."
        )

    logger.info(f"{cfg.pipeline.engine.name} inference engine initialized.")
    return engine


def create_loader(
    cfg: EvalConfig,
    evaluator_cfg: EvaluatorConfig,
    engine: BaseEngine,
) -> tuple[BaseEvalLoader | LuxonisLoader, str | None]:
    if engine.width is None or engine.height is None:
        raise RuntimeError("Engine input shape is unavailable after setup.")

    try:
        if cfg.pipeline.loader.name == "LuxonisLoader":
            return _create_luxonis_loader(cfg, evaluator_cfg, engine)

        dataloader = from_registry(
            DATALOADERS_REGISTRY,
            cfg.pipeline.loader.name,
            **cfg.pipeline.loader.params,
        )
        if not isinstance(dataloader, BaseEvalLoader):
            raise TypeError(
                f"{cfg.pipeline.loader.name} custom dataloader must be an instance of BaseEvalLoader."
            )
        loader_task_name = None
    except KeyError as e:
        raise ValueError(
            f"Unknown loader: {cfg.pipeline.loader.name}. "
            f"Available loaders: {list(DATALOADERS_REGISTRY._module_dict)}"
        ) from e

    logger.info(f"{cfg.pipeline.loader.name} dataloader initialized.")

    return dataloader, loader_task_name


def create_parser(evaluator_cfg: EvaluatorConfig) -> BaseParser:
    try:
        parser = from_registry(
            PARSERS_REGISTRY,
            evaluator_cfg.parser.name,
            **evaluator_cfg.parser.params,
        )
    except KeyError as e:
        raise ValueError(
            f"Unknown parser: {evaluator_cfg.parser.name}. "
            f"Available parsers: {list(PARSERS_REGISTRY._module_dict)}"
        ) from e

    if not isinstance(parser, BaseParser):
        raise TypeError(
            f"{evaluator_cfg.parser.name} parser must be an instance of BaseParser."
        )

    logger.info(f"{evaluator_cfg.parser.name} parser initialized.")
    return parser


def create_metrics(evaluator_cfg: EvaluatorConfig) -> list[BaseMetric]:
    metrics: list[BaseMetric] = []
    for metric_cfg in evaluator_cfg.metrics:
        try:
            metric = from_registry(
                METRICS_REGISTRY,
                metric_cfg.name,
                **metric_cfg.params,
            )
        except KeyError as e:
            raise ValueError(
                f"Unknown metric: {metric_cfg.name}. "
                f"Available metrics: {list(METRICS_REGISTRY._module_dict)}"
            ) from e

        if not isinstance(metric, BaseMetric):
            raise TypeError(
                f"{metric_cfg.name} metric must be an instance of BaseMetric."
            )

        logger.info(f"{metric_cfg.name} metric initialized.")
        metrics.append(metric)

    return metrics


def create_visualizers(
    evaluator_cfg: EvaluatorConfig,
) -> list[BaseVisualizer]:
    active_visualizers = [
        visualizer_cfg
        for visualizer_cfg in evaluator_cfg.visualizers
        if visualizer_cfg.active
    ]
    if not active_visualizers:
        logger.info("Visualization is disabled.")
        return []

    visualizers: list[BaseVisualizer] = []
    for visualizer_cfg in active_visualizers:
        try:
            visualizer = from_registry(
                VISUALIZERS_REGISTRY,
                visualizer_cfg.name,
                **visualizer_cfg.params,
            )
        except KeyError as e:
            raise ValueError(
                f"Unknown visualizer: {visualizer_cfg.name}. "
                f"Available visualizers: {list(VISUALIZERS_REGISTRY._module_dict)}"
            ) from e

        if not isinstance(visualizer, BaseVisualizer):
            raise TypeError(
                f"{visualizer_cfg.name} visualizer must be an instance of BaseVisualizer."
            )

        logger.info(f"{visualizer_cfg.name} visualizer initialized.")
        visualizers.append(visualizer)

    return visualizers


def _create_luxonis_loader(
    cfg: EvalConfig,
    evaluator_cfg: EvaluatorConfig,
    engine: BaseEngine,
) -> tuple[LuxonisLoader, str]:
    loader_params = dict(cfg.pipeline.loader.params)
    loader_params.pop("filter_task_names", None)
    loader_params.pop("class_mapping", None)
    dataset_name: str = loader_params.pop("dataset_name")  # type: ignore

    dataset_kwargs = {}
    for key in (
        "team_id",
        "bucket_type",
        "bucket_storage",
        "delete_local",
        "delete_remote",
    ):
        if key in loader_params:
            dataset_kwargs[key] = loader_params.pop(key)

    dataset = LuxonisDataset(dataset_name, **dataset_kwargs)
    loader_task_name = resolve_luxonis_task_name(
        dataset_name,
        dataset.get_classes(),
        task_name=evaluator_cfg.task_name,
    )
    loader_params["filter_task_names"] = [loader_task_name]

    augmentation_config = []
    if cfg.pipeline.loader.preprocessing.normalize.active:
        augmentation_config.append(
            {
                "name": "Normalize",
                "params": cfg.pipeline.loader.preprocessing.normalize.params,
            }
        )

    dataloader = LuxonisLoader(
        dataset,
        view=loader_params.pop("view", ["val"]),  # type: ignore
        augmentation_config=augmentation_config,
        height=engine.height,
        width=engine.width,
        keep_aspect_ratio=cfg.pipeline.loader.preprocessing.keep_aspect_ratio,
        color_space=cfg.pipeline.loader.preprocessing.color_space,
        **loader_params,  # type: ignore
    )

    logger.info(
        "LuxonisLoader dataloader initialized with "
        f"task_name={loader_task_name!r}."
    )
    return dataloader, loader_task_name
