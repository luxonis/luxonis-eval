from pathlib import Path
from typing import Literal

from luxonis_ml.data import BucketStorage, LuxonisDataset
from luxonis_ml.typing import BaseModelExtraForbid, ConfigItem, Params
from luxonis_ml.utils.config import LuxonisConfig
from pydantic import field_validator, model_validator

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
    normalize: NormalizeAugmentationConfig
    color_space: Literal["RGB", "BGR", "GRAY"] = "RGB"
    keep_aspect_ratio: bool = False


class DataLoaderConfig(ConfigItem):
    preprocessing: PreProcessingConfig

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

            task_name = self.params.get("task_name")
            filter_task_names = self.params.get("filter_task_names")
            if task_name is not None and not isinstance(task_name, str):
                raise ValueError(
                    "loader.params.task_name must be a string when provided."
                )
            if filter_task_names is not None:
                if not isinstance(filter_task_names, list | tuple):
                    raise ValueError(
                        "loader.params.filter_task_names must be a list or tuple when provided."
                    )
                if len(filter_task_names) != 1:
                    raise ValueError(
                        "Only one Luxonis task is supported per evaluation run. "
                        f"Received filter_task_names={list(filter_task_names)}."
                    )
                if not all(
                    isinstance(task, str) for task in filter_task_names
                ):
                    raise ValueError(
                        "loader.params.filter_task_names must contain only strings."
                    )
                if task_name is not None and filter_task_names[0] != task_name:
                    raise ValueError(
                        f"loader.params.task_name={task_name!r} conflicts with "
                        f"filter_task_names={list(filter_task_names)}."
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


class MetricsConfig(BaseModelExtraForbid):
    metrics: list[MetricConfig]


class VisualizerConfig(ConfigItem):
    visualize: bool = True

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


class EvalConfig(LuxonisConfig):
    """Configuration for evaluation."""

    loader: DataLoaderConfig
    parser: ParserConfig
    metrics: MetricsConfig
    visualizer: VisualizerConfig | None = None
    engine: EngineConfig
