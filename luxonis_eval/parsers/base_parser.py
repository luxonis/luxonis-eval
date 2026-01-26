from abc import ABC, abstractmethod
from typing import Any

from luxonis_ml.utils.registry import AutoRegisterMeta, Registry

PARSERS_REGISTRY: Registry[type["BaseParser"]] = Registry(name="parsers")


class BaseParser(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=PARSERS_REGISTRY,
    register=False,
):
    """Base class for model output parsers."""

    @abstractmethod
    def parse(self, raw_output: Any, **kwargs: Any) -> Any:
        """Parse raw backend output into predictions."""
        ...
