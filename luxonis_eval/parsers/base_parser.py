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

    TASK: str
    BACKENDS: set[str]

    @abstractmethod
    def parse(self, raw_output: Any, **kwargs: Any) -> Any:
        """Parse raw backend output into predictions."""
        ...

    @classmethod
    def supports(cls, *, task: str, backend: str) -> bool:
        """Check whether the parser supports a task-backend pair.

        Parameters
        ----------
        task : str
            Task identifier.
        backend : str
            Backend identifier.

        Returns
        -------
        bool
            True if the parser supports the task and backend.
        """
        return (getattr(cls, "TASK", None) == task) and (
            backend in getattr(cls, "BACKENDS", set())
        )

    @classmethod
    def select(cls, *, task: str, backend: str) -> type["BaseParser"]:
        """Select a parser for a task-backend pair.

        Parameters
        ----------
        task : str
            Task identifier.
        backend : str
            Backend identifier.

        Returns
        -------
        type[BaseParser]
            Parser class matching the task and backend.
        """
        candidates: list[type[BaseParser]] = []
        for pcls in PARSERS_REGISTRY._module_dict.values():
            if pcls.supports(task=task, backend=backend):
                candidates.append(pcls)

        if not candidates:
            raise KeyError(
                f"No parser found for task={task!r}, backend={backend!r}"
            )

        if len(candidates) > 1:
            names = [c.__name__ for c in candidates]
            raise KeyError(
                f"Multiple parsers match task={task!r}, backend={backend!r}: {names}"
            )

        return candidates[0]
