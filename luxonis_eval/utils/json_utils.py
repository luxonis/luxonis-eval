import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from luxonis_eval.core.results import EvaluationResult


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def build_output_json_payload(result: EvaluationResult) -> dict[str, Any]:
    metrics_payload = [
        {"name": metric_name, "values": to_jsonable(metric_values)}
        for metric_name, metric_values in result.metrics
    ]

    return {
        "engine": result.engine,
        "model_name": result.model_name,
        "metrics": metrics_payload,
    }


def write_output_json(path: str, result: EvaluationResult) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_output_json_payload(result)
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
