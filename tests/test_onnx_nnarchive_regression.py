from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from luxonis_ml.data import BucketStorage, LuxonisParser
from luxonis_ml.enums import DatasetType

from luxonis_eval.__main__ import quality_run

RELATIVE_TOLERANCE = 0.02


@dataclass(frozen=True)
class RegressionCase:
    name: str


CASES = [
    RegressionCase(name="detection_shapes_ldf_native"),
    RegressionCase(
        name="instance_segmentation_multiclass_squares_ldf_native_empty_task"
    ),
    RegressionCase(name="keypoint_squares_ldf_native_empty_task"),
    RegressionCase(
        name="semantic_segmentation_squares_ldf_native_empty_task"
    ),
]


def _case_dir(onnx_nnarchive_testdata_root: Path, case: RegressionCase) -> Path:
    return onnx_nnarchive_testdata_root / case.name


def _parse_dataset(case: RegressionCase, onnx_nnarchive_testdata_root: Path) -> None:
    case_dir = _case_dir(onnx_nnarchive_testdata_root, case)
    LuxonisParser(
        str(case_dir / case.name),
        dataset_name=case.name,
        dataset_type=DatasetType.NATIVE,
        bucket_storage=BucketStorage.LOCAL,
        delete_local=True,
    ).parse()


def _load_expected_metrics(
    case: RegressionCase, onnx_nnarchive_testdata_root: Path
) -> dict[str, float]:
    case_dir = _case_dir(onnx_nnarchive_testdata_root, case)
    return {
        key: float(value)
        for key, value in json.loads(
            (case_dir / "metrics_summary.json").read_text(encoding="utf-8")
        ).items()
    }


def _flatten_metrics(
    metrics: list[tuple[str, dict[str, float]]],
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for metric_name, metric_values in metrics:
        for metric_key, metric_value in metric_values.items():
            flattened[f"{metric_name}.{metric_key}"] = float(metric_value)
    return flattened


def _assert_within_tolerance(
    metric_key: str, *, actual: float, expected: float
) -> None:
    if expected == 0:
        assert actual == expected, (
            f"{metric_key} mismatch: actual={actual:.6f}, "
            f"expected={expected:.6f}."
        )
        return

    relative_error = abs(actual - expected) / abs(expected)
    if relative_error <= RELATIVE_TOLERANCE:
        return

    raise AssertionError(
        f"{metric_key} mismatch: actual={actual:.6f}, "
        f"expected={expected:.6f}, rel_error={relative_error:.2%}, "
        f"allowed={RELATIVE_TOLERANCE:.2%}."
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_onnx_nnarchive_regression(
    case: RegressionCase,
    monkeypatch,
    onnx_nnarchive_testdata_root: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LUXONISML_BASE_PATH", str(tmp_path / "luxonis_ml"))
    monkeypatch.setenv("LUXONISML_TEAM_ID", "pytest")

    case_dir = _case_dir(onnx_nnarchive_testdata_root, case)

    _parse_dataset(case, onnx_nnarchive_testdata_root)
    expected_metrics = _load_expected_metrics(
        case, onnx_nnarchive_testdata_root
    )

    result = quality_run(
        case_dir / "onnx_eval.yaml",
        {
            "pipeline.engine.model_path": str(
                case_dir / "onnx_model.onnx.tar.xz"
            ),
            "runtime.logging.level": "WARNING",
            "runtime.logging.use_rich": False,
        },
    )
    actual_metrics = _flatten_metrics(result["metrics"])

    missing_metrics = sorted(set(expected_metrics) - set(actual_metrics))
    assert not missing_metrics, (
        f"{case.name} is missing expected metrics: {missing_metrics}. "
        f"Available metrics: {sorted(actual_metrics)}"
    )

    for metric_key, expected_value in expected_metrics.items():
        _assert_within_tolerance(
            metric_key,
            actual=actual_metrics[metric_key],
            expected=expected_value,
        )
