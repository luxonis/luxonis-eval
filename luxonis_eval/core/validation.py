from luxonis_eval.config import EvaluatorConfig
from luxonis_eval.engines.base_engine import ModelSpec


def validate_engine_setup(model_spec: ModelSpec) -> None:
    if model_spec.width <= 0 or model_spec.height <= 0:
        raise RuntimeError("Engine setup did not return a valid model spec.")


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
