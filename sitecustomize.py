import os
import sys
from pathlib import Path


def _is_pytest_process() -> bool:
    executable_name = Path(sys.argv[0]).name.lower()
    return "pytest" in executable_name


if _is_pytest_process():
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
