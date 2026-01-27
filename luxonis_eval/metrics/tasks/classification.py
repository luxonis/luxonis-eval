from collections.abc import Sequence

from luxonis_eval.metrics.primitives.topk_accuracy import TopKAccuracy
from luxonis_eval.metrics.tasks.base_task_metric import BaseTaskMetric


class ClassificationMetric(BaseTaskMetric):
    """Classification task metric."""

    def __init__(self, *, topk: Sequence[int] = (1, 5)) -> None:
        """Initialize the classification metric with Top-K accuracy.

        Parameters
        ----------
        topk : Sequence[int]
            Tuple of K values for Top-K accuracy.
        """
        super().__init__(metrics=[TopKAccuracy(topk=topk)])
