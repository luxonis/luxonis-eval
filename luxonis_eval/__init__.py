from typing import Final

from luxonis_ml.utils import setup_logging
from pydantic_extra_types.semantic_version import SemanticVersion

__version__: Final[str] = "0.0.1"
__semver__: Final[SemanticVersion] = SemanticVersion.parse(__version__)

from .core import (
    EvaluationResult as EvaluationResult,
)
from .core import (
    LuxonisEval as LuxonisEval,
)
from .core import (
    MetricResult as MetricResult,
)
from .core import (
    MetricsResult as MetricsResult,
)
from .core import (
    MetricValues as MetricValues,
)
from .core import (
    ThroughputResult as ThroughputResult,
)
from .engines import *
from .loaders import *
from .metrics import *
from .parsers import *
from .visualizers import *

setup_logging()
