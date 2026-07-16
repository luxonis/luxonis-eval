import os
from pathlib import Path

import pytest
from luxonis_ml.utils import LuxonisFileSystem

os.environ.setdefault("LUXONIS_TELEMETRY_ENABLED", "false")

NORMALIZED_TESTDATA_REMOTE_DIR = (
    "gs://luxonis-test-bucket/luxonis-eval-test-data/NormalizedTestData"
)


@pytest.fixture(scope="session")
def onnx_nnarchive_testdata_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data_dir = tmp_path_factory.mktemp("onnx_nnarchive_testdata")
    return LuxonisFileSystem.download(
        NORMALIZED_TESTDATA_REMOTE_DIR,
        dest=data_dir,
    )
