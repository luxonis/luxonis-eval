from typing import TYPE_CHECKING, Annotated, Literal

from luxonis_ml.typing import BaseModelExtraForbid, Params
from luxonis_ml.utils.config import LuxonisConfig
from pydantic import (
    Field,
    PlainSerializer,
    PrivateAttr,
    field_validator,
    model_validator,
)
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

if TYPE_CHECKING:
    from luxonis_eval.config.nn_archive import NNArchiveConfig


class NormalizeAugmentationConfig(BaseModelExtraForbid):
    active: bool = False
    params: Params = Field(default_factory=dict)

    @field_validator("params", mode="after")
    @classmethod
    def validate_params(cls, value: Params) -> Params:
        if not value:
            return {}
        return validate_normalize_params(value)


class PreProcessingConfig(BaseModelExtraForbid):
    normalize: NormalizeAugmentationConfig = Field(
        default_factory=NormalizeAugmentationConfig
    )
    color_space: Literal["RGB", "BGR", "GRAY"] = "RGB"
    keep_aspect_ratio: bool = True


class DataLoaderConfig(BaseModelExtraForbid):
    name: str
    params: Params = {}
    preprocessing: PreProcessingConfig = Field(
        default_factory=PreProcessingConfig
    )

    @field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_dataloader_name(value)

    @model_validator(mode="after")
    def validate_dataset(self) -> "DataLoaderConfig":
        validate_luxonis_loader_dataset(self)
        return self


class EvaluatorConfig(BaseModelExtraForbid):
    name: str | None = None
    task_name: str | None = None
    outputs: list[str] | None = None
    parser: ParserConfig
    metrics: list[MetricConfig]
    visualizers: list[VisualizerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_and_resolve(self) -> "EvaluatorConfig":
        if not self.metrics:
            raise ValueError(
                "pipeline.evaluators[*].metrics must contain at least one metric."
            )
        if self.name is None:
            self.name = self.task_name or "task_0"
        return self


class PipelineConfig(BaseModelExtraForbid):
    loader: DataLoaderConfig
    engine: EngineConfig
    evaluators: list[EvaluatorConfig] | None = None
    benchmark: Params | None = None

    @model_validator(mode="after")
    def validate_evaluators(self) -> "PipelineConfig":
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


class ResolvedEvalConfig(LuxonisConfig):
    version: Annotated[
        SemanticVersion,
        Field(frozen=True),
        PlainSerializer(str),
    ] = lxe.__semver__

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    pipeline: PipelineConfig

    _nn_archive_cfg: "NNArchiveConfig | None" = PrivateAttr(default=None)

    @property
    def nn_archive_cfg(self) -> "NNArchiveConfig | None":
        return self._nn_archive_cfg

    def _set_nn_archive_cfg(
        self, nn_archive_cfg: "NNArchiveConfig | None"
    ) -> None:
        self._nn_archive_cfg = nn_archive_cfg
