"""This module implements a metaclass for automatic registration of
classes."""

from typing import TypeVar

from luxonis_ml.utils.registry import Registry

import luxonis_eval as lxeval

ENGINES_REGISTRY: Registry[type["lxeval.BaseEngine"]] = Registry(
    name="engines"
)
METRICS_REGISTRY: Registry[type["lxeval.BaseMetric"]] = Registry(
    name="metrics"
)
VISUALIZERS_REGISTRY: Registry[type["lxeval.BaseVisualizer"]] = Registry(
    name="visualizers"
)
PARSERS_REGISTRY: Registry[type["lxeval.BaseParser"]] = Registry(
    name="parsers"
)
TASKS_REGISTRY: Registry[type["lxeval.BaseInferTask"]] = Registry(
    name="infer_tasks"
)

T = TypeVar("T")


def from_registry(registry: Registry[type[T]], key: str, *args, **kwargs) -> T:
    """Get an instance of the class registered under the given key.

    Parameters
    ----------
    registry : Registry[type[T]]
        Registry to get the class from.
    key : str
        Key to get the class for.
    *args : Any
        Positional arguments to pass to the class constructor.
    **kwargs : Any
        Keyword arguments to pass to the class constructor.
    Returns
    -------
    T
        Instance of the class registered under the given key.
    """
    return registry.get(key)(*args, **kwargs)
