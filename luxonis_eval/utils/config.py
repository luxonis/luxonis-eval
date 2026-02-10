from pathlib import Path
from typing import Literal

from luxonis_ml.utils.config import LuxonisConfig
from pydantic import ConfigDict, field_validator, model_validator


class TaskConfig(LuxonisConfig):
    name: str


class ParserConfig(LuxonisConfig):
    name: str
    apply_softmax: bool | None = None


class MetricCfg(LuxonisConfig):
    name: str
    model_config: ConfigDict = ConfigDict(extra="allow")


class MetricsCfg(LuxonisConfig):
    metrics: list[MetricCfg]


class OnnxConfig(LuxonisConfig):
    providers: str | list[str] = "CPUExecutionProvider"
    mean: list[float] | float = 0.0
    std: list[float] | float = 1.0


class EvalConfig(LuxonisConfig):
    """Configuration for evaluation."""

    dataset_name: str
    backend: Literal["depthai", "onnx", "all"]
    nn_archive: str | None = None
    onnx: str | None = None
    device_ip: str | None = None

    task_cfg: TaskConfig
    parser_cfg: ParserConfig
    metrics_cfg: MetricsCfg
    onnx_cfg: OnnxConfig | None = None

    @field_validator("nn_archive", mode="after")
    @classmethod
    def _nn_archive_must_exist(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not Path(v).exists():
            raise ValueError(f"NNArchive file '{v}' does not exist.")
        return v

    @field_validator("onnx", mode="after")
    @classmethod
    def _onnx_must_exist(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not Path(v).exists():
            raise ValueError(f"ONNX model file '{v}' does not exist.")
        return v

    @model_validator(mode="after")
    def _validate_inputs_present(self) -> "EvalConfig":
        if not self.nn_archive and not self.onnx:
            raise ValueError(
                "At least one of nn_archive or onnx must be provided."
            )
        return self

    @model_validator(mode="after")
    def _validate_backend_all_requires_both(self) -> "EvalConfig":
        if self.backend == "all" and not (self.nn_archive and self.onnx):
            raise ValueError(
                "Both nn_archive and onnx must be provided when backend is 'all'."
            )
        return self

    @model_validator(mode="after")
    def _validate_backend_matches_inputs(self) -> "EvalConfig":
        if self.nn_archive and self.backend not in ("depthai", "all"):
            raise ValueError(
                "nn_archive can only be used with backend 'depthai' or 'all'."
            )
        if self.onnx and self.backend not in ("onnx", "all"):
            raise ValueError(
                "onnx can only be used with backend 'onnx' or 'all'."
            )
        return self
