#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install gcsfs
python -m pip install -e ".[dev]"

if [[ -n "${DEPTHAI_VERSION:-}" ]]; then
  python -m pip install "depthai==${DEPTHAI_VERSION}"
fi

if [[ -n "${HIL_FRAMEWORK_TOKEN:-}" && -n "${HIL_TESTBED:-}" ]]; then
  python -m pip install --upgrade \
    --index-url "https://__token__:${HIL_FRAMEWORK_TOKEN}@gitlab.luxonis.com/api/v4/projects/213/packages/pypi/simple" \
    hil-framework
fi

export LUXONIS_TELEMETRY_ENABLED="${LUXONIS_TELEMETRY_ENABLED:-false}"

pytest tests/test_rvc4_nnarchive_regression.py -q "$@"
