import os
from pathlib import Path

import pytest
from luxonis_ml.utils import LuxonisFileSystem

os.environ.setdefault("LUXONIS_TELEMETRY_ENABLED", "false")

NORMALIZED_TESTDATA_REMOTE_DIR = (
    "gs://luxonis-test-bucket/luxonis-eval-test-data/NormalizedTestData"
)
LOCAL_TESTDATA_ENV_VAR = "LUXONIS_EVAL_TESTDATA_DIR"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device-ip",
        action="store",
        default=None,
        help=(
            "RVC4 device IP or MXID. If omitted, uses RVC4_IP or "
            "testbed resolution when configured."
        ),
    )
    parser.addoption(
        "--testbed-name",
        action="store",
        default=None,
        help="Optional HIL testbed name used to resolve an RVC4 device.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "device: mark test as requiring a physical device"
    )
    config.addinivalue_line(
        "markers", "rvc4: mark test as requiring an RVC4 device"
    )


@pytest.fixture(scope="session")
def nnarchive_testdata_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    configured_dir = os.environ.get(LOCAL_TESTDATA_ENV_VAR)
    if configured_dir:
        return Path(configured_dir)

    data_dir = tmp_path_factory.mktemp("nnarchive_testdata")
    return LuxonisFileSystem.download(
        NORMALIZED_TESTDATA_REMOTE_DIR,
        dest=data_dir,
    )


@pytest.fixture(scope="session")
def rvc4_device_ip(request: pytest.FixtureRequest) -> str | None:
    configured_device_ip = request.config.getoption("--device-ip")
    if configured_device_ip:
        return configured_device_ip

    env_device_ip = os.environ.get("RVC4_IP")
    if env_device_ip:
        return env_device_ip

    testbed_name = request.config.getoption("--testbed-name") or os.environ.get(
        "HIL_TESTBED"
    )
    if not testbed_name:
        return None

    return _resolve_rvc4_device_ip_from_testbed(testbed_name)


@pytest.fixture(scope="session")
def required_rvc4_device_ip(rvc4_device_ip: str | None) -> str:
    if rvc4_device_ip is not None:
        return rvc4_device_ip

    pytest.skip(
        "No RVC4 device configured. Re-run with --device-ip <ip-or-mxid>, "
        "set RVC4_IP, or provide --testbed-name / HIL_TESTBED."
    )


def _resolve_rvc4_device_ip_from_testbed(testbed_name: str) -> str:
    try:
        from hil_framework.lib_testbed.config.Config import Config
        from hil_framework.lib_testbed.utils.Testbed import Testbed
    except ImportError as exc:
        pytest.exit(
            "hil_framework is required when --testbed-name or HIL_TESTBED is used.",
            returncode=1,
        )
        raise exc

    testbed = Testbed(Config(testbed_name))
    target_matches = [
        camera
        for camera in testbed.cameras
        if str(getattr(camera, "platform", "")).lower() == "rvc4"
    ]
    if len(target_matches) == 1:
        hostname = getattr(target_matches[0], "hostname", None)
        if not hostname:
            pytest.exit(
                f"Selected RVC4 camera in testbed {testbed_name!r} does not expose a hostname.",
                returncode=1,
            )
        return str(hostname)

    available_cameras = ", ".join(
        f"{camera.name}:{getattr(camera, 'hostname', None)}:{getattr(camera, 'platform', None)}"
        for camera in testbed.cameras
    )
    if not target_matches:
        pytest.exit(
            f"No RVC4 camera found in testbed {testbed_name!r}. Available cameras: {available_cameras}",
            returncode=1,
        )

    pytest.exit(
        "Unable to select a unique RVC4 camera from testbed "
        f"{testbed_name!r}. Available cameras: {available_cameras}",
        returncode=1,
    )
