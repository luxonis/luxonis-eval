from typing import Any

import numpy as np
from depthai_nodes import Classifications
from depthai_nodes.node.parsers.utils.code128 import (
    code128_codewords_to_text,
    parse_codeword_sequence,
)

from luxonis_eval.metrics.base_metric import BaseMetric


class Code128TextAccuracy(BaseMetric):
    """Evaluation metric for Code128 CTC outputs decoded with beam search."""

    def __init__(
        self,
        target_key: str = "/metadata/code128_codewords",
        top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        self.target_key = target_key
        self.top_k = int(top_k)
        super().__init__(**kwargs)

    def required_target_keys(self) -> list[str]:
        return [self.target_key]

    def reset(self) -> None:
        self.exact_codeword_matches = 0
        self.exact_payload_matches = 0
        self.topk_codeword_matches = 0
        self.topk_payload_matches = 0
        self.rank_1_matches = 0
        self.rank_2_matches = 0
        self.valid_decodes = 0
        self.total = 0

    def update(
        self,
        predictions: Classifications,
        target: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> None:
        del kwargs
        if not isinstance(predictions, Classifications):
            raise TypeError(
                "Code128TextAccuracy expects predictions as "
                "`depthai_nodes.Classifications`."
            )

        metadata = getattr(predictions, "metadata", {})
        candidates = metadata.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(
                "Code128TextAccuracy expects parser metadata with a non-empty "
                "`candidates` list."
            )

        normalized_candidates = [
            _normalize_candidate(candidate) for candidate in candidates[: self.top_k]
        ]
        best_candidate = normalized_candidates[0]

        target_codewords = parse_codeword_sequence(target[self.target_key])
        target_payload = code128_codewords_to_text(target_codewords)

        if best_candidate["valid"]:
            self.valid_decodes += 1
        if best_candidate["codewords"] == target_codewords:
            self.exact_codeword_matches += 1
        if (
            best_candidate["valid"]
            and best_candidate["text"] == target_payload
        ):
            self.exact_payload_matches += 1

        codeword_errors = _count_sequence_errors(
            best_candidate["codewords"],
            target_codewords,
        )
        if codeword_errors == 1:
            self.rank_1_matches += 1
        if codeword_errors == 2:
            self.rank_2_matches += 1

        if any(
            candidate["codewords"] == target_codewords
            for candidate in normalized_candidates
        ):
            self.topk_codeword_matches += 1
        if any(
            candidate["valid"] and candidate["text"] == target_payload
            for candidate in normalized_candidates
        ):
            self.topk_payload_matches += 1

        self.total += 1

    def compute(self) -> dict[str, float]:
        if self.total == 0:
            return {
                "Code128TextAccuracy": 0.0,
                "exact_codeword_accuracy": 0.0,
                "exact_payload_accuracy": 0.0,
                "valid_decode_rate": 0.0,
                "rank_1": 0.0,
                "rank_2": 0.0,
                f"top{self.top_k}_exact_codeword_accuracy": 0.0,
                f"top{self.top_k}_exact_payload_accuracy": 0.0,
            }

        exact_codeword_accuracy = self.exact_codeword_matches / self.total
        exact_payload_accuracy = self.exact_payload_matches / self.total
        return {
            "Code128TextAccuracy": float(exact_codeword_accuracy),
            "exact_codeword_accuracy": float(exact_codeword_accuracy),
            "exact_payload_accuracy": float(exact_payload_accuracy),
            "valid_decode_rate": float(self.valid_decodes / self.total),
            "rank_1": float(self.rank_1_matches / self.total),
            "rank_2": float(self.rank_2_matches / self.total),
            f"top{self.top_k}_exact_codeword_accuracy": float(
                self.topk_codeword_matches / self.total
            ),
            f"top{self.top_k}_exact_payload_accuracy": float(
                self.topk_payload_matches / self.total
            ),
        }


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    codewords = candidate.get("codewords", [])
    if not isinstance(codewords, list):
        raise TypeError("Code128 candidate `codewords` must be a list.")
    return {
        "text": str(candidate.get("text", "")),
        "codewords": [int(codeword) for codeword in codewords],
        "valid": bool(candidate.get("valid", False)),
        "checksum_valid": bool(candidate.get("checksum_valid", False)),
    }


def _count_sequence_errors(
    prediction: list[int],
    target: list[int],
) -> int:
    max_len = max(len(prediction), len(target))
    return sum(
        (
            prediction[index] if index < len(prediction) else None
        ) != (
            target[index] if index < len(target) else None
        )
        for index in range(max_len)
    )
