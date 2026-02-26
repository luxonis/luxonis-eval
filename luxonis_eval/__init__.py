from typing import Final

from luxonis_ml.utils import setup_logging
from pydantic_extra_types.semantic_version import SemanticVersion

__version__: Final[str] = "0.0.1"
__semver__: Final[SemanticVersion] = SemanticVersion.parse(__version__)

from luxonis_eval.engines.base_engine import BaseEngine
from luxonis_eval.loaders.base_loader import BaseEvalLoader
from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.parsers.base_parser import BaseParser
from luxonis_eval.tasks.base_task import BaseInferTask
from luxonis_eval.visualizers.base_visualizer import BaseVisualizer

__all__ = [
    "BaseEngine",
    "BaseEvalLoader",
    "BaseInferTask",
    "BaseMetric",
    "BaseParser",
    "BaseVisualizer",
]

setup_logging()
