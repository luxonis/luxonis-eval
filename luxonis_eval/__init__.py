from typing import Final

from luxonis_ml.utils import setup_logging
from pydantic_extra_types.semantic_version import SemanticVersion

__version__: Final[str] = "0.0.1"
__semver__: Final[SemanticVersion] = SemanticVersion.parse(__version__)

setup_logging()
