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

pytest_args=(tests/test_rvc4_nnarchive_regression.py -q)

if [[ -n "${HIL_TESTBED:-}" ]]; then
  if [[ -n "${HIL_FRAMEWORK_TOKEN:-}" ]]; then
    python -m pip install --upgrade \
      --index-url "https://__token__:${HIL_FRAMEWORK_TOKEN}@gitlab.luxonis.com/api/v4/projects/213/packages/pypi/simple" \
      hil-framework
  fi

  pytest_args+=(--testbed-name "$HIL_TESTBED")
fi

export LUXONIS_TELEMETRY_ENABLED="${LUXONIS_TELEMETRY_ENABLED:-false}"

pytest "${pytest_args[@]}" "$@"
