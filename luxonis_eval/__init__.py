from typing import Final

from luxonis_ml.utils import setup_logging
from pydantic_extra_types.semantic_version import SemanticVersion

__version__: Final[str] = "0.0.1"
__semver__: Final[SemanticVersion] = SemanticVersion.parse(__version__)

from .core import LuxonisEval  # noqa: F401
from .engines import *
from .loaders import *
from .metrics import *
from .parsers import *
from .visualizers import *

setup_logging()
