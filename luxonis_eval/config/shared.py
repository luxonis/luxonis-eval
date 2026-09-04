from pathlib import Path
from typing import Literal, Protocol

from loguru import logger
from luxonis_ml.data import BucketStorage, LuxonisDataset
from luxonis_ml.nn_archive.utils import is_nn_archive
from luxonis_ml.typing import (
    BaseModelExtraForbid,
    ConfigItem,
    Params,
    PathType,
)
from pydantic import Field, field_validator, model_validator

from luxonis_eval.registry import (
    DATALOADERS_REGISTRY,
    ENGINES_REGISTRY,
    METRICS_REGISTRY,
    PARSERS_REGISTRY,
    VISUALIZERS_REGISTRY,
)

DEFAULT_NORMALIZE_PARAMS: Params = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


class LoaderConfigLike(Protocol):
    name: str
    params: Params


class ParserConfig(ConfigItem):
    @field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if value not in PARSERS_REGISTRY:
            raise ValueError(
                f"Invalid parser name: {value}. Must be one of {list(PARSERS_REGISTRY._module_dict)}."
            )
        return value


class MetricConfig(ConfigItem):
    @field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if value not in METRICS_REGISTRY:
            raise ValueError(
                f"Invalid metric name: {value}. Must be one of {list(METRICS_REGISTRY._module_dict)}."
            )
        return value


class VisualizerConfig(ConfigItem):
    active: bool = True
    mode: Literal["save", "display"] = "save"
    save_dir: Path = Path("visualizations")

    @field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if value not in VISUALIZERS_REGISTRY:
            raise ValueError(
                f"Invalid visualizer name: {value}. Must be one of {list(VISUALIZERS_REGISTRY._module_dict)}."
            )
        return value


class EngineConfig(ConfigItem):
    model_path: str

    @field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if value not in ENGINES_REGISTRY:
            raise ValueError(
                f"Invalid engine name: {value}. Must be one of {list(ENGINES_REGISTRY._module_dict)}."
            )
        return value

    @field_validator("model_path", mode="after")
    @classmethod
    def validate_model_path(cls, value: str) -> str:
        if not Path(value).exists():
            raise ValueError(f"Model file '{value}' does not exist.")
        return value

    @model_validator(mode="after")
    def validate_backend_matches_inputs(self) -> "EngineConfig":
        if is_nn_archive(self.model_path) and self.name not in {
            "depthai",
            "onnx",
        }:
            raise ValueError(
                f"NNArchive model ({self.model_path}) can only be used with the 'depthai' or 'onnx' backend."
            )
        if self.model_path.endswith(".onnx") and self.name != "onnx":
            raise ValueError(
                f"ONNX model ({self.model_path}) can only be used with the 'onnx' backend."
            )
        return self


class LoggingConfig(BaseModelExtraForbid):
    use_rich: bool = True
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = (
        None
    )
    file: PathType | None = None


class RuntimeConfig(BaseModelExtraForbid):
    nn_archive_params_override: bool = False
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def validate_normalize_params(params: Params) -> Params:
    normalized = dict(params)
    if "mean" not in normalized or "std" not in normalized:
        raise ValueError("Both 'mean' and 'std' must be specified in params.")
    if not isinstance(normalized["mean"], list | tuple):
        normalized["mean"] = [normalized["mean"]] * 3
    else:
        normalized["mean"] = list(normalized["mean"])
    if not isinstance(normalized["std"], list | tuple):
        normalized["std"] = [normalized["std"]] * 3
    else:
        normalized["std"] = list(normalized["std"])
    return normalized


def validate_dataloader_name(name: str) -> str:
    if name not in DATALOADERS_REGISTRY:
        raise ValueError(
            f"Invalid dataloader name: {name}. Must be one of {list(DATALOADERS_REGISTRY._module_dict)}."
        )
    return name


def validate_luxonis_loader_dataset(loader_cfg: LoaderConfigLike) -> None:
    dataset_name = loader_cfg.params.get("dataset_name")
    if loader_cfg.name != "LuxonisLoader":
        return

    if dataset_name is None or dataset_name == "":
        raise ValueError(
            "LuxonisLoader requires the 'dataset_name' parameter to be set."
        )

    filter_task_names = loader_cfg.params.get("filter_task_names")
    if filter_task_names is not None:
        logger.warning(
            "loader.params.filter_task_names is ignored. "
            "Use pipeline.evaluators[*].task_name for task selection."
        )

    bucket_storage = loader_cfg.params.get("bucket_storage", "local")
    luxonis_datasets = LuxonisDataset.list_datasets(
        bucket_storage=BucketStorage(bucket_storage)
    )

    if dataset_name not in luxonis_datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' does not exist in '{bucket_storage}' bucket storage. Available datasets: {luxonis_datasets}"
        )
