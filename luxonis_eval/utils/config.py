from pathlib import Path

from luxonis_ml.typing import ConfigItem
from luxonis_ml.utils.config import LuxonisConfig
from pydantic import BaseModel, field_validator, model_validator

from luxonis_eval.registry import (
    ENGINES_REGISTRY,
    METRICS_REGISTRY,
    PARSERS_REGISTRY,
    TASKS_REGISTRY,
)


class DatasetConfig(ConfigItem): ...


class TaskConfig(ConfigItem):
    @field_validator("name", mode="after")
    def validate_name(cls, v: str) -> str:
        if v not in TASKS_REGISTRY:
            raise ValueError(
                f"Invalid task name: {v}. Must be one of {list(TASKS_REGISTRY._module_dict)}."
            )
        return v


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


class MetricsConfig(BaseModel):
    metrics: list[MetricConfig]


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

    dataset_cfg: DatasetConfig
    task_cfg: TaskConfig
    parser_cfg: ParserConfig
    metrics_cfg: MetricsConfig
    engine_cfg: EngineConfig
