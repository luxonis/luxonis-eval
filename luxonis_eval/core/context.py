from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from luxonis_eval.engines.base_engine import ModelSpec

TargetConverter = Callable[
    [np.ndarray, int, int],
    tuple[np.ndarray, np.ndarray],
]


@dataclass(frozen=True, slots=True)
class EvalContext:
    """Static runtime metadata shared across one evaluation run."""

    model_spec: ModelSpec
    class_map: dict[int, str]
    target_class_map: dict[int, str]
    class_index_map: dict[int, int] | None
    category_ids: tuple[int, ...]
    target_background_index: int | None
    target_converter: TargetConverter

    @property
    def width(self) -> int:
        return self.model_spec.width

    @property
    def height(self) -> int:
        return self.model_spec.height
