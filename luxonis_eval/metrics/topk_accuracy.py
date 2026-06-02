from collections.abc import Sequence
from typing import Any

import numpy as np
from depthai_nodes import Classifications

from luxonis_eval.metrics.base_metric import BaseMetric


class TopKAccuracy(BaseMetric):
    """Top-K accuracy metric."""

    def __init__(self, topk: Sequence[int] = (1, 5), **kwargs: Any) -> None:
        """Initialize the Top-K accuracy metric.

        Parameters
        ----------
        topk : Sequence[int], optional
            Sequence of K values for top-K accuracy.
        **kwargs : Any
            Additional metric configuration.
        """
        self.topk = tuple(int(k) for k in topk)
        super().__init__(**kwargs)

    def metric_keys(self) -> list[str]:
        """Return the ground-truth keys required by the metric.

        Returns
        -------
        list[str]
            Ground-truth key names.
        """
        return ["/classification"]

    def _reset_impl(self) -> None:
        """Reset internal metric state."""
        self.correct_at_k = dict.fromkeys(self.topk, 0)
        self.total = 0

    def _update_impl(
        self,
        predictions: Classifications,
        target: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> None:
        """Update internal metric state.

        Parameters
        ----------
        predictions : Classifications
            Model predictions (logits or probabilities).
        target : dict[str, np.ndarray]
            Ground-truth labels.
        **kwargs : Any
            Additional context.
        """
        cls_target = target[self.metric_keys()[0]]
        class_index_map = kwargs.get("class_index_map")
        class_map = kwargs.get("class_map", {})
        class_map = {v: k for k, v in class_map.items()}

        topk = tuple(kwargs.get("topk", self.topk))

        pred_classes = predictions.classes
        tgt = np.asarray(cls_target)

        target_idx = (
            int(np.argmax(tgt)) if tgt.ndim > 0 and tgt.size > 1 else int(tgt)
        )
        if class_index_map is not None:
            target_idx = int(class_index_map[target_idx])

        max_k = max(topk)
        top_idx = [class_map[pred_classes[i]] for i in range(max_k)]  # type: ignore

        for k in topk:
            if k not in self.correct_at_k:
                self.correct_at_k[k] = 0
            if target_idx in top_idx[:k]:
                self.correct_at_k[k] += 1

        self.total += 1

    def _compute_impl(self) -> dict[str, float]:
        """Compute final Top-K accuracy metrics.

        Returns
        -------
        dict[str, float]
            Computed Top-K accuracy results.
        """
        if self.total == 0:
            return {"top1_acc": 0.0, "top5_acc": 0.0}
        return {
            f"top{k}_acc": float(self.correct_at_k[k] / self.total)
            for k in sorted(self.correct_at_k)
        }
