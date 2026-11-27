#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.12.0/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --locked

if [[ -n "${DEPTHAI_VERSION:-}" ]]; then
  uv pip install --python .venv/bin/python "depthai==${DEPTHAI_VERSION}"
fi

pytest_args=(
  -s
  -v
  tests/test_rvc4_nnarchive_regression.py
  --require-device
)

if [[ -n "${HIL_TESTBED:-}" ]]; then
  if [[ -n "${HIL_FRAMEWORK_TOKEN:-}" ]]; then
    uv pip install --python .venv/bin/python --upgrade \
      --index-url "https://__token__:${HIL_FRAMEWORK_TOKEN}@gitlab.luxonis.com/api/v4/projects/213/packages/pypi/simple" \
      hil-framework
  fi

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

# Preserve the explicitly requested DepthAI version and HIL package above.
uv run --no-sync pytest "${pytest_args[@]}" "$@"
