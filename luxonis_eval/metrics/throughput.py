import time

from luxonis_eval.core.results import ThroughputResult


class ThroughputMetric:
    """Throughput evaluation metric."""

    _TRACKED_STAGES = (
        "inference",
        "parsing",
        "metric_update",
        "metric_compute",
    )

    def __init__(self) -> None:
        """Initialize the metric."""
        self.reset()

    def reset(self) -> None:
        """Reset the metric state."""
        self._num_updates = 0
        self._stage_elapsed = dict.fromkeys(self._TRACKED_STAGES, 0.0)

        self._t0 = time.perf_counter()

    def update(
        self,
        *,
        inference: float = 0.0,
        parsing: float = 0.0,
        metric_update: float = 0.0,
    ) -> None:
        """Update the metric with a new sample and tracked stage
        timings."""
        self._num_updates += 1
        self._stage_elapsed["inference"] += inference
        self._stage_elapsed["parsing"] += parsing
        self._stage_elapsed["metric_update"] += metric_update

    def compute(
        self, *, metric_compute: float = 0.0
    ) -> ThroughputResult:
        """Compute final throughput metrics.

        Parameters
        ----------
        metric_compute : float, optional
            Time spent in final metric aggregation after the sample loop.

        Returns
        -------
        ThroughputResult
            Computed throughput results.
        """
        self._stage_elapsed["metric_compute"] = metric_compute
        elapsed = max(time.perf_counter() - self._t0, 1e-12)
        sps = self._num_updates / elapsed
        msp = (
            (elapsed / self._num_updates) * 1000.0
            if self._num_updates
            else 0.0
        )
        tracked_elapsed = sum(self._stage_elapsed.values())
        overhead_elapsed = max(elapsed - tracked_elapsed, 0.0)

        return ThroughputResult(
            elapsed_s=float(elapsed),
            samples=int(self._num_updates),
            samples_per_s=float(sps),
            ms_per_sample=float(msp),
            overhead_ms_per_sample=float(
                (overhead_elapsed / self._num_updates) * 1000.0
                if self._num_updates
                else 0.0
            ),
            inference_ms_per_sample=float(
                (self._stage_elapsed["inference"] / self._num_updates)
                * 1000.0
                if self._num_updates
                else 0.0
            ),
            parsing_ms_per_sample=float(
                (self._stage_elapsed["parsing"] / self._num_updates)
                * 1000.0
                if self._num_updates
                else 0.0
            ),
            metric_update_ms_per_sample=float(
                (self._stage_elapsed["metric_update"] / self._num_updates)
                * 1000.0
                if self._num_updates
                else 0.0
            ),
            metric_compute_ms_per_sample=float(
                (self._stage_elapsed["metric_compute"] / self._num_updates)
                * 1000.0
                if self._num_updates
                else 0.0
            ),
        )
