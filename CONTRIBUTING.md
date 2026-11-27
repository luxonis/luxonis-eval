# Contributing to LuxonisEval

## Development setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from the repository root:

```bash
uv sync --locked
uv run prek install
```

This creates `.venv` and installs the project in editable mode. The default
`dev` dependency group includes pytest, GCS support for regression tests, and
[prek](https://prek.j178.dev/) for the pre-commit hooks.

Use `uv run` to run commands in this environment, or activate it with
`source .venv/bin/activate` to call the CLI directly.

## Tests and hooks

Run the local metric tests:

```bash
uv run pytest tests/test_segmentation_background.py
```

Run all pre-commit hooks:

```bash
uv run prek run --all-files
```

## Dependency updates

Declare dependencies in `pyproject.toml`, then regenerate the lockfile and pip
compatibility exports:

```bash
scripts/export_requirements.sh
```

To upgrade locked dependencies, run `uv lock --upgrade` before exporting.
Commit `pyproject.toml`, `uv.lock`, and both requirements files together.
The requirements files are generated; do not edit them manually.

## Locked installs with pip

Use pip 23.3 or newer. To install the locked runtime dependencies and package:

```bash
pip install -r requirements.txt
pip install --no-deps .
```

For development, install the development export and the package in editable
mode:

```bash
pip install -r requirements-dev.txt
pip install --no-deps -e .
```
