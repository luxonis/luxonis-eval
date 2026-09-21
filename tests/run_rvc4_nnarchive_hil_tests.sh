#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv --clear venv
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]" "gcsfs==2024.6.1"

if [[ -n "${DEPTHAI_VERSION:-}" ]]; then
  python -m pip install "depthai==${DEPTHAI_VERSION}"
fi

pytest_args=(
  -s
  -v
  tests/test_rvc4_nnarchive_regression.py
  --require-device
)

if [[ -n "${HIL_TESTBED:-}" ]]; then
  hil_wheels=("$ROOT_DIR"/.ci/hil_framework-*.whl)
  if [[ ${#hil_wheels[@]} -ne 1 || ! -f "${hil_wheels[0]}" ]]; then
    echo "Expected exactly one synced hil-framework wheel in $ROOT_DIR/.ci." >&2
    exit 1
  fi

  python -m pip install --upgrade "${hil_wheels[0]}"
  pytest_args+=(--testbed-name "$HIL_TESTBED")
fi

export LUXONIS_TELEMETRY_ENABLED="${LUXONIS_TELEMETRY_ENABLED:-false}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"

echo "HIL test configuration:"
echo "  DEPTHAI_VERSION=${DEPTHAI_VERSION:-<default>}"
echo "  HIL_TESTBED=${HIL_TESTBED:-<empty>}"
printf '  pytest_args:'
printf ' %q' "${pytest_args[@]}"
printf '\n'

pytest "${pytest_args[@]}" "$@"
