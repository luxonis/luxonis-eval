from loguru import logger

from luxonis_eval.config import EvaluatorConfig
from luxonis_eval.engines.base_engine import BaseEngine


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
