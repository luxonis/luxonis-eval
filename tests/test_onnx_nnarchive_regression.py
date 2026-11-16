from pathlib import Path

import pytest

from luxonis_eval.__main__ import quality_run

try:
    from tests.nnarchive_regression_common import (
        CASES,
        RegressionCase,
        assert_within_relative_tolerance,
        flatten_metrics,
        load_expected_metrics,
        parse_dataset,
        resolve_case_artifacts,
    )
except ModuleNotFoundError:
    from nnarchive_regression_common import (
        CASES,
        RegressionCase,
        assert_within_relative_tolerance,
        flatten_metrics,
        load_expected_metrics,
        parse_dataset,
        resolve_case_artifacts,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_onnx_nnarchive_regression(
    case: RegressionCase,
    monkeypatch: pytest.MonkeyPatch,
    nnarchive_testdata_root: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LUXONISML_BASE_PATH", str(tmp_path / "luxonis_ml"))
    monkeypatch.setenv("LUXONISML_TEAM_ID", "pytest")

    config_path, model_path = resolve_case_artifacts(
        case, nnarchive_testdata_root, "onnx"
    )

    expected_metrics = load_expected_metrics(case, nnarchive_testdata_root)
    parse_dataset(case, nnarchive_testdata_root)

    result = quality_run(
        config_path,
        {
            "pipeline.engine.model_path": str(model_path),
            "runtime.logging.level": "WARNING",
            "runtime.logging.use_rich": False,
        },
    )
    actual_metrics = flatten_metrics(result.metrics)

    missing_metrics = sorted(set(expected_metrics) - set(actual_metrics))
    assert not missing_metrics, (
        f"{case.name} is missing expected metrics: {missing_metrics}. "
        f"Available metrics: {sorted(actual_metrics)}"
    )

    for metric_key, expected_value in expected_metrics.items():
        assert_within_relative_tolerance(
            metric_key,
            actual=actual_metrics[metric_key],
            expected=expected_value,
        )
