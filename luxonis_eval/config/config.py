from pathlib import Path
from typing import Annotated, Literal

from loguru import logger
from luxonis_ml.data import BucketStorage, LuxonisDataset
from luxonis_ml.typing import BaseModelExtraForbid, ConfigItem, Params
from luxonis_ml.utils.config import LuxonisConfig
from pydantic import Field, PlainSerializer, field_validator, model_validator
from pydantic_extra_types.semantic_version import SemanticVersion

import luxonis_eval as lxe
from luxonis_eval.registry import (
    DATALOADERS_REGISTRY,
    ENGINES_REGISTRY,
    METRICS_REGISTRY,
    PARSERS_REGISTRY,
    VISUALIZERS_REGISTRY,
)


class NormalizeAugmentationConfig(BaseModelExtraForbid):
    active: bool = False
    params: Params = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    @field_validator("params", mode="after")
    def validate_params(cls, v: Params) -> Params:
        if "mean" not in v or "std" not in v:
            raise ValueError(
                "Both 'mean' and 'std' must be specified in params."
            )
        if not isinstance(v["mean"], (list | tuple)):
            v["mean"] = [v["mean"]] * 3
        if not isinstance(v["std"], (list | tuple)):
            v["std"] = [v["std"]] * 3
        return v


class PreProcessingConfig(BaseModelExtraForbid):
    normalize: NormalizeAugmentationConfig = Field(
        default_factory=NormalizeAugmentationConfig
    )
    color_space: Literal["RGB", "BGR", "GRAY"] = "RGB"
    keep_aspect_ratio: bool = False


class DataLoaderConfig(ConfigItem):
    preprocessing: PreProcessingConfig = Field(
        default_factory=PreProcessingConfig
    )

    @field_validator("name", mode="after")
    def validate_name(cls, v: str) -> str:
        if v not in DATALOADERS_REGISTRY:
            raise ValueError(
                f"Invalid dataloader name: {v}. Must be one of {list(DATALOADERS_REGISTRY._module_dict)}."
            )
        return v

    @model_validator(mode="after")
    def _validate_dataset(self) -> "DataLoaderConfig":
        dataset_name = self.params.get("dataset_name")
        if self.name == "LuxonisLoader":
            if dataset_name is None or dataset_name == "":
                raise ValueError(
                    "LuxonisLoader requires the 'dataset_name' parameter to be set."
                )

            filter_task_names = self.params.get("filter_task_names")
            if filter_task_names is not None:
                logger.warning(
                    "loader.params.filter_task_names is ignored. "
                    "Use pipeline.evaluators[*].task_name for task selection."
                )

            bucket_storage = self.params.get("bucket_storage", "local")
            luxonis_datasets = LuxonisDataset.list_datasets(
                bucket_storage=BucketStorage(bucket_storage)
            )

            if dataset_name not in luxonis_datasets:
                raise ValueError(
                    f"Dataset '{dataset_name}' does not exist in '{bucket_storage}' bucket storage. Available datasets: {luxonis_datasets}"
                )
        return self


class ParserConfig(ConfigItem):
    @field_validator("name", mode="after")
    def validate_name(cls, v: str) -> str:
        if v not in PARSERS_REGISTRY:
            raise ValueError(
                f"Invalid parser name: {v}. Must be one of {list(PARSERS_REGISTRY._module_dict)}."
            )
        return v


class MetricConfig(ConfigItem):
    @field_validator("name", mode="after")
    def validate_name(cls, v: str) -> str:
        if v not in METRICS_REGISTRY:
            raise ValueError(
                f"Invalid metric name: {v}. Must be one of {list(METRICS_REGISTRY._module_dict)}."
            )
        return v


class VisualizerConfig(ConfigItem):
    active: bool = True

    @field_validator("name", mode="after")
    def validate_name(cls, v: str) -> str:
        if v not in VISUALIZERS_REGISTRY:
            raise ValueError(
                f"Invalid visualizer name: {v}. Must be one of {list(VISUALIZERS_REGISTRY._module_dict)}."
            )
        return v


class EngineConfig(ConfigItem):
    model_path: str

    @field_validator("name", mode="after")
    def validate_name(cls, v: str) -> str:
        if v not in ENGINES_REGISTRY:
            raise ValueError(
                f"Invalid engine name: {v}. Must be one of {list(ENGINES_REGISTRY._module_dict)}."
            )
        return v

    @field_validator("model_path", mode="after")
    def validate_model_path(cls, v: str) -> str:
        if not Path(v).exists():
            raise ValueError(f"Model file '{v}' does not exist.")
        return v

    @model_validator(mode="after")
    def _validate_backend_matches_inputs(self) -> "EngineConfig":
        if self.model_path.endswith(".tar.xz") and self.name != "depthai":
            raise ValueError(
                f"NNArchive model ({self.model_path}) can only be used with the 'depthai' backend."
            )
        if self.model_path.endswith(".onnx") and self.name != "onnx":
            raise ValueError(
                f"ONNX model ({self.model_path}) can only be used with the 'onnx' backend."
            )
        return self


class RuntimeConfig(BaseModelExtraForbid):
    pass


class EvaluatorConfig(BaseModelExtraForbid):
    name: str | None = None
    task_name: str | None = None
    outputs: list[str] | None = None
    parser: ParserConfig
    metrics: list[MetricConfig]
    visualizers: list[VisualizerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_and_resolve(self) -> "EvaluatorConfig":
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
    def _validate_evaluators(self) -> "PipelineConfig":
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


class EvalConfig(LuxonisConfig):
    """Configuration for evaluation."""

    version: Annotated[
        SemanticVersion,
        Field(frozen=True),
        PlainSerializer(str),
    ] = lxe.__semver__

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    pipeline: PipelineConfig
