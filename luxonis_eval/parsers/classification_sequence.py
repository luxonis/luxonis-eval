from typing import Any

import numpy as np
from depthai_nodes import Classifications
from depthai_nodes.message.creators import (
    create_classification_sequence_message,
    create_code128_classification_message,
)
from depthai_nodes.node.parsers.utils.classification_sequence import (
    compute_classification_sequence_scores,
)
from depthai_nodes.node.parsers.utils.code128 import (
    beam_search_code128,
    normalize_candidate_scores,
    select_best_code128_candidate,
)

from luxonis_eval.engines.base_engine import ModelSpec
from luxonis_eval.engines.io import EngineOutput
from luxonis_eval.parsers.base_parser import BaseParser
from luxonis_eval.utils.utils import ordered_class_names


class ClassificationSequenceParser(BaseParser):
    """Parser for sequence-style classification outputs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def parse(
        self,
        output: EngineOutput,
        model_spec: ModelSpec,
        *,
        class_map: dict[int, str],
        classes: list[str] | None = None,
        is_softmax: bool = True,
        apply_softmax: bool | None = None,
        ignored_indexes: list[int] | None = None,
        remove_duplicates: bool = False,
        concatenate_classes: bool = False,
        target_mode: str = "payload",
        beam_width: int = 10,
        top_k: int = 5,
        token_prune: int | None = None,
        prefer_valid_checksum: bool = True,
        **kwargs: Any,
    ) -> Classifications:
        del model_spec, kwargs

        _, scores = output.get_first()
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size == 0:
            raise ValueError("Classification sequence output is empty.")

        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "Classification sequence output contains non-finite values "
                "before post-processing."
            )

        output_is_softmax = (
            is_softmax if apply_softmax is None else not apply_softmax
        )
        scores = compute_classification_sequence_scores(
            scores,
            is_softmax=output_is_softmax,
        )

        if target_mode == "codewords":
            candidates = beam_search_code128(
                scores,
                beam_width=beam_width,
                top_k=top_k,
                token_prune=token_prune,
            )
            best_candidate = select_best_code128_candidate(
                candidates,
                prefer_valid_checksum=prefer_valid_checksum,
            )
            ordered_candidates = [best_candidate] + [
                candidate
                for candidate in candidates
                if candidate != best_candidate
            ]
            return create_code128_classification_message(
                ordered_candidates,
                normalize_candidate_scores(ordered_candidates),
                target_mode=target_mode,
                best_candidate=best_candidate,
                beam_width=beam_width,
                token_prune=token_prune,
            )

        resolved_classes = classes or ordered_class_names(class_map)
        return create_classification_sequence_message(
            classes=resolved_classes,
            scores=scores,
            ignored_indexes=ignored_indexes,
            remove_duplicates=remove_duplicates,
            concatenate_classes=concatenate_classes,
        )
