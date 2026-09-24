from dataclasses import dataclass

from luxonis_eval.engines.base_engine import ModelSpec


@dataclass(frozen=True, slots=True)
class EvalContext:
    """Static runtime metadata shared across one evaluation run."""

    model_spec: ModelSpec
    class_map: dict[int, str]
    target_class_map: dict[int, str]
    class_index_map: dict[int, int] | None
    category_ids: tuple[int, ...]
    target_background_index: int | None

    @property
    def width(self) -> int:
        return self.model_spec.width

    @property
    def height(self) -> int:
        return self.model_spec.height
