from pathlib import Path

import pytest

from luxonis_eval.__main__ import quality_run
from tests.nnarchive_regression_common import (
    CASES,
    RegressionCase,
    assert_within_relative_tolerance,
    case_dir,
    flatten_metrics,
    load_expected_metrics,
    parse_dataset,
)


@pytest.mark.device
@pytest.mark.rvc4
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_rvc4_nnarchive_regression(
    case: RegressionCase,
    monkeypatch: pytest.MonkeyPatch,
    nnarchive_testdata_root: Path,
    required_rvc4_device_ip: str,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LUXONISML_BASE_PATH", str(tmp_path / "luxonis_ml"))
    monkeypatch.setenv("LUXONISML_TEAM_ID", "pytest")

    resolved_case_dir = case_dir(nnarchive_testdata_root, case)

    parse_dataset(case, nnarchive_testdata_root)
    expected_metrics = load_expected_metrics(case, nnarchive_testdata_root)

    result = quality_run(
        resolved_case_dir / "depthai_eval.yaml",
        {
            "pipeline.engine.model_path": str(
                resolved_case_dir / "rvc4_model.rvc4.tar.xz"
            ),
            "pipeline.engine.params.device_ip": required_rvc4_device_ip,
            "runtime.logging.level": "WARNING",
            "runtime.logging.use_rich": False,
        },
    )
    actual_metrics = flatten_metrics(result["metrics"])

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
