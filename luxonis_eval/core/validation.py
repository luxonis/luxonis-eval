from typing import Any

from loguru import logger
from luxonis_ml.data.loaders import LuxonisLoader

from luxonis_eval.config import EvaluatorConfig
from luxonis_eval.engines.base_engine import BaseEngine
from luxonis_eval.loaders.base_loader import BaseEvalLoader
from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.parsers.base_parser import BaseParser

from .runtime import normalize_target, select_evaluator_outputs


def validate_engine_setup(engine: BaseEngine) -> None:
    if engine.width is None or engine.height is None:
        raise RuntimeError("Engine setup did not populate the input shape.")
    if engine.platform_name is None:
        raise RuntimeError("Engine setup did not populate the platform name.")


def resolve_evaluator_config(
    evaluators: list[EvaluatorConfig] | None,
) -> EvaluatorConfig:
    if not evaluators:
        raise NotImplementedError(
            "pipeline.evaluators is required for quality evaluation. "
            "Benchmark-only pipeline execution is not implemented yet."
        )
    if len(evaluators) > 1:
        raise NotImplementedError(
            "Multiple pipeline.evaluators are not implemented yet."
        )
    return evaluators[0]


def run_static_compatibility_warnings(
    *,
    engine_name: str,
    normalize_active: bool,
    color_space: str,
) -> None:
    if engine_name == "depthai" and normalize_active:
        logger.warning(
            "Normalization is usually part of the model's preprocessing "
            "pipeline in DepthAI. Consider disabling normalization in the "
            "dataset config."
        )
    if engine_name == "depthai" and color_space == "RGB":
        logger.warning(
            "Color space is set to RGB in the dataset config. DepthAI expects "
            "BGR color space."
        )


def sanity_check_pipeline(
    *,
    loader: BaseEvalLoader | LuxonisLoader,
    engine: BaseEngine,
    parser: BaseParser,
    metrics: list[BaseMetric],
    metric_contexts: list[dict[str, Any]],
    evaluator_cfg: EvaluatorConfig,
    class_map: dict[int, str],
    loader_task_name: str | None,
) -> None:
    if len(loader) == 0:
        raise ValueError(
            "Evaluation loader is empty. Pipeline sanity check requires at "
            "least one sample."
        )

    logger.info("Running pipeline sanity check on one real sample.")

    img, target = loader[0]
    target = normalize_target(
        target,
        loader=loader,
        loader_task_name=loader_task_name,
    )
    raw_output = engine.infer_once(img)
    predictions = parser.parse(
        select_evaluator_outputs(raw_output, evaluator_cfg.outputs),
        class_map=class_map,
        **evaluator_cfg.parser.params,
    )

    for metric, metric_ctx in zip(metrics, metric_contexts, strict=True):
        missing = set(metric.required_target_keys()) - set(target)
        if missing:
            raise ValueError(
                f"Target is missing required keys for {metric.__class__.__name__}: "
                f"{sorted(missing)}. Got keys: {sorted(target.keys())}."
            )

        metric.update(
            predictions=predictions,
            target=target,
            **metric_ctx,
        )
        metric.compute()
        metric.reset()
