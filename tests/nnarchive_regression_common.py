from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from luxonis_ml.data import BucketStorage, LuxonisParser
from luxonis_ml.enums import DatasetType

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


def case_dir(nnarchive_testdata_root: Path, case: RegressionCase) -> Path:
    return nnarchive_testdata_root / case.name


def parse_dataset(case: RegressionCase, nnarchive_testdata_root: Path) -> None:
    resolved_case_dir = case_dir(nnarchive_testdata_root, case)
    LuxonisParser(
        str(resolved_case_dir / case.name),
        dataset_name=case.name,
        dataset_type=DatasetType.NATIVE,
        bucket_storage=BucketStorage.LOCAL,
        delete_local=True,
    ).parse()


def load_expected_metrics(
    case: RegressionCase, nnarchive_testdata_root: Path
) -> dict[str, float]:
    resolved_case_dir = case_dir(nnarchive_testdata_root, case)
    return {
        key: float(value)
        for key, value in json.loads(
            (resolved_case_dir / "metrics_summary.json").read_text(
                encoding="utf-8"
            )
        ).items()
    }


def flatten_metrics(
    metrics: list[tuple[str, dict[str, float]]],
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for metric_name, metric_values in metrics:
        for metric_key, metric_value in metric_values.items():
            flattened[f"{metric_name}.{metric_key}"] = float(metric_value)
    return flattened


def assert_within_relative_tolerance(
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
