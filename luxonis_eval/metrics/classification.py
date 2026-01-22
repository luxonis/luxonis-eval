from collections.abc import Sequence
from typing import Any

import numpy as np

from luxonis_eval.metrics.base_metric import BaseMetric


class ClassificationMetric(BaseMetric):
    """Classification accuracy metric."""

    NAME = "classification"

    def _reset_impl(self) -> None:
        """Reset internal metric state."""
        self.correct_at_k: dict[int, int] = {}
        self.total = 0

    def _update_impl(
        self, predictions: Any, target: Any, **kwargs: Any
    ) -> None:
        """Update metric with predictions and ground truths.

        Parameters
        ----------
        predictions : Any
            Model prediction scores.
        target : Any
            Ground-truth label or one-hot vector.
        **kwargs : Any
            Additional context including class mapping and top-k values.
        """
        # Retrieve additional task-specific options
        class_index_map = kwargs.get("class_index_map")
        topk: Sequence[int] = kwargs.get("topk", (1, 5))

        scores = np.asarray(predictions)
        target = np.asarray(target)

        if target.ndim > 0 and target.size > 1:
            target_idx = int(np.argmax(target))
        else:
            target_idx = int(target)

        if class_index_map is not None:
            target_idx = int(class_index_map[target_idx])

        max_k = max(topk)
        top_idx = np.argsort(scores)[-max_k:][::-1]

        for k in topk:
            if k not in self.correct_at_k:
                self.correct_at_k[k] = 0
            if target_idx in top_idx[:k]:
                self.correct_at_k[k] += 1

        self.total += 1

    def _compute_impl(self) -> dict[str, float]:
        """Compute final accuracy metrics.

        Returns
        -------
        dict[str, float]
            Top-k accuracy values.
        """
        if self.total == 0:
            return {"top1_acc": 0.0, "top5_acc": 0.0}

        out: dict[str, float] = {}
        for k, correct in sorted(self.correct_at_k.items()):
            out[f"top{k}_acc"] = float(correct / self.total)
        return out
