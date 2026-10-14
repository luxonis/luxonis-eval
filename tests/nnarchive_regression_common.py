from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from luxonis_ml.data import BucketStorage, LuxonisParser
from luxonis_ml.enums import DatasetType

RELATIVE_TOLERANCE = 0.025


@dataclass(frozen=True)
class RegressionCase:
    name: str


CASES = [
    RegressionCase(name="detection_shapes_ldf_native"),
    RegressionCase(
        name="instance_segmentation_multiclass_squares_ldf_native_empty_task"
    ),
    RegressionCase(name="keypoint_squares_ldf_native_empty_task"),
    RegressionCase(name="semantic_segmentation_squares_ldf_native_empty_task"),
    RegressionCase(name="classification_shapes_ldf_native_empty_task"),
    RegressionCase(name="detection_shapes_ldf_native_empty_task"),
    RegressionCase(name="instance_segmentation_squares_ldf_native_empty_task"),
    RegressionCase(name="keypoint_multiclass_squares_ldf_native_empty_task"),
    RegressionCase(
        name="semantic_segmentation_multiclass_shapes_ldf_native_empty_task"
    ),
    RegressionCase(
        name="semantic_segmentation_squares_explicit_background_ldf_native_empty_task"
    ),
]

EVAL_CONFIG_NAMES = {
    "classification": "classification_config",
    "detection": "detection_config",
    "instance": "instance_segmentation",
    "keypoint": "keypoints",
    "semantic": "semantic_segmentation",
}

# LuxonisTrain-to-LuxonisEval metric name mappings, grouped by task family.
TRAIN_TO_EVAL_METRIC_MAP = {
    "classification": {"Accuracy": "TopKAccuracy.top1_acc"},
    "detection": {
        "MeanAveragePrecision": "BboxMeanAveragePrecision.AP",
        "map_50": "BboxMeanAveragePrecision.AP50",
    },
    "instance": {
        "bbox_map": "BboxMeanAveragePrecision.AP",
        "bbox_map_50": "BboxMeanAveragePrecision.AP50",
        "MeanAveragePrecision": "MaskMeanAveragePrecision.AP",
        "segm_map_50": "MaskMeanAveragePrecision.AP50",
    },
    "keypoint": {
        "MeanAveragePrecision": "KeypointMeanAveragePrecision.MeanAveragePrecisionKeypoints",
        **{
            name: f"KeypointMeanAveragePrecision.{name}"
            for name in (
                "kpt_f1_50",
                "kpt_f1_75",
                "kpt_f1_large",
                "kpt_f1_medium",
                "kpt_map_50",
                "kpt_map_75",
                "kpt_map_large",
                "kpt_map_medium",
                "kpt_mar",
                "kpt_mar_50",
                "kpt_mar_75",
                "kpt_mar_large",
                "kpt_mar_medium",
            )
        },
    },
    "semantic": {
        "F1Score": "F1Score.F1Score",
        "JaccardIndex": "JaccardIndex.JaccardIndex",
    },
}


def case_dir(nnarchive_testdata_root: Path, case: RegressionCase) -> Path:
    return nnarchive_testdata_root / case.name


def resolve_case_artifacts(
    case: RegressionCase,
    nnarchive_testdata_root: Path,
    backend: Literal["onnx", "depthai"],
) -> tuple[Path, Path]:
    directory = case_dir(nnarchive_testdata_root, case)
    stem = EVAL_CONFIG_NAMES[case.name.split("_", 1)[0]]
    config_names = (
        ("onnx_eval.yaml", f"{stem}.yaml")
        if backend == "onnx"
        else ("depthai_eval.yaml", f"{stem}_rvc4.yaml")
    )
    config_path = next(
        (
            directory / name
            for name in config_names
            if (directory / name).is_file()
        ),
        None,
    )
    if config_path is None:
        raise FileNotFoundError(
            f"{case.name}: missing {backend} evaluation config in {directory}; "
            f"expected one of {config_names}."
        )

    suffix = "onnx" if backend == "onnx" else "rvc4"
    models = sorted(directory.glob(f"*.{suffix}.tar.xz"))
    if len(models) != 1:
        raise ValueError(
            f"{case.name}: expected one {backend} NNArchive in {directory}, "
            f"found {models}."
        )
    return config_path, models[0]


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
    """Read reference test scores under LuxonisEval result keys.

    For LuxonisTrain summaries, select and rename test metrics using
    TRAIN_TO_EVAL_METRIC_MAP. Summaries already using LuxonisEval result
    keys are returned directly.
    """
    resolved_case_dir = case_dir(nnarchive_testdata_root, case)
    summary = json.loads(
        (resolved_case_dir / "metrics_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if "test" in summary:
        scalars = summary["test"]["scalars"]
    elif any(key.startswith("test/metric/") for key in summary):
        scalars = {
            key.removeprefix("test/"): value
            for key, value in summary.items()
            if key.startswith("test/metric/")
        }
    else:
        # Some fixtures store scores directly under LuxonisEval result keys.
        if not summary:
            raise ValueError(f"{case.name}: metric summary is empty.")
        return {key: float(value) for key, value in summary.items()}

    train_test_metrics = {}
    for key, value in scalars.items():
        if not key.startswith("metric/"):
            continue
        metric_name = key.rsplit("/", 1)[-1]
        if metric_name in train_test_metrics:
            raise ValueError(
                f"{case.name}: multiple LuxonisTrain test metrics use "
                f"the name {metric_name!r}."
            )
        train_test_metrics[metric_name] = value

    task_family = case.name.split("_", 1)[0]
    metric_name_map = TRAIN_TO_EVAL_METRIC_MAP[task_family]
    missing_train_metrics = sorted(
        set(metric_name_map) - train_test_metrics.keys()
    )
    if missing_train_metrics:
        raise ValueError(
            f"{case.name}: missing LuxonisTrain test metrics required "
            f"for comparison: {missing_train_metrics}."
        )
    return {
        eval_key: float(train_test_metrics[train_name])
        for train_name, eval_key in metric_name_map.items()
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
