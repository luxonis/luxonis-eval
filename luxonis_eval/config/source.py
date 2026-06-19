from typing import Annotated, Literal

from luxonis_ml.typing import BaseModelExtraForbid, Params
from pydantic import Field, PlainSerializer, field_validator, model_validator
from pydantic_extra_types.semantic_version import SemanticVersion

import luxonis_eval as lxe
from luxonis_eval.config.shared import (
    EngineConfig,
    MetricConfig,
    ParserConfig,
    RuntimeConfig,
    VisualizerConfig,
    validate_dataloader_name,
    validate_luxonis_loader_dataset,
    validate_normalize_params,
)


class SourceNormalizeAugmentationConfig(BaseModelExtraForbid):
    active: bool | None = None
    params: Params | None = None

    @field_validator("params", mode="after")
    @classmethod
    def validate_params(cls, value: Params | None) -> Params | None:
        if value is None:
            return None
        return validate_normalize_params(value)


class SourcePreProcessingConfig(BaseModelExtraForbid):
    normalize: SourceNormalizeAugmentationConfig | None = None
    color_space: Literal["RGB", "BGR", "GRAY"] | None = None
    keep_aspect_ratio: bool = False


class SourceDataLoaderConfig(BaseModelExtraForbid):
    name: str
    params: Params = {}
    preprocessing: SourcePreProcessingConfig = Field(
        default_factory=SourcePreProcessingConfig
    )

    @field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_dataloader_name(value)

    @model_validator(mode="after")
    def validate_dataset(self) -> "SourceDataLoaderConfig":
        validate_luxonis_loader_dataset(self)
        return self


class SourceEvaluatorConfig(BaseModelExtraForbid):
    name: str | None = None
    task_name: str | None = None
    outputs: list[str] | None = None
    parser: ParserConfig | None = None
    metrics: list[MetricConfig]
    visualizers: list[VisualizerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_and_resolve(self) -> "SourceEvaluatorConfig":
        if not self.metrics:
            raise ValueError(
                "pipeline.evaluators[*].metrics must contain at least one metric."
            )
        if self.name is None:
            self.name = self.task_name or "task_0"
        return self


class SourcePipelineConfig(BaseModelExtraForbid):
    loader: SourceDataLoaderConfig
    engine: EngineConfig
    evaluators: list[SourceEvaluatorConfig] | None = None
    benchmark: Params | None = None

    @model_validator(mode="after")
    def validate_evaluators(self) -> "SourcePipelineConfig":
        if self.evaluators is not None:
            if len(self.evaluators) == 0:
                raise ValueError(
                    "pipeline.evaluators must not be empty when provided."
                )
            if len(self.evaluators) > 1:
                raise NotImplementedError(
                    "Multiple pipeline.evaluators are not implemented yet."
                )
        return self


class SourceEvalConfig(BaseModelExtraForbid):
    version: Annotated[
        SemanticVersion,
        Field(frozen=True),
        PlainSerializer(str),
    ] = lxe.__semver__

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    pipeline: SourcePipelineConfig
