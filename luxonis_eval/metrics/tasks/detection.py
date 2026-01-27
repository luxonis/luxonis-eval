from luxonis_eval.metrics.primitives.bbox_map import BboxMeanAveragePrecision
from luxonis_eval.metrics.tasks.base_task_metric import BaseTaskMetric


class DetectionMetric(BaseTaskMetric):
    """Object detection task metric."""

    def __init__(self) -> None:
        """Initialize the detection metric with bounding box mean average precision."""
        super().__init__(metrics=[BboxMeanAveragePrecision()])
