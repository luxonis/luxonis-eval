import time


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
        self._stage_elapsed_s = dict.fromkeys(self._TRACKED_STAGES, 0.0)

        self._t0 = time.perf_counter()

    def update(
        self,
        *,
        inference_s: float = 0.0,
        parsing_s: float = 0.0,
        metric_update_s: float = 0.0,
    ) -> None:
        """Update the metric with a new sample and tracked stage
        timings."""
        self._num_updates += 1
        self._stage_elapsed_s["inference"] += inference_s
        self._stage_elapsed_s["parsing"] += parsing_s
        self._stage_elapsed_s["metric_update"] += metric_update_s

    def compute(
        self, *, metric_compute_s: float = 0.0
    ) -> dict[str, float | int]:
        """Compute final throughput metrics.

        Parameters
        ----------
        metric_compute_s : float, optional
            Time spent in final metric aggregation after the sample loop.

        Returns
        -------
        dict[str, float | int]
            Computed throughput results.
        """
        self._stage_elapsed_s["metric_compute"] = metric_compute_s
        elapsed = max(time.perf_counter() - self._t0, 1e-12)
        sps = self._num_updates / elapsed
        msp = (
            (elapsed / self._num_updates) * 1000.0
            if self._num_updates
            else 0.0
        )
        tracked_elapsed = sum(self._stage_elapsed_s.values())
        overhead_elapsed = max(elapsed - tracked_elapsed, 0.0)

        results: dict[str, float | int] = {
            "elapsed_s": float(elapsed),
            "samples": int(self._num_updates),
            "samples_per_s": float(sps),
            "ms_per_sample": float(msp),
            "overhead_elapsed_s": float(overhead_elapsed),
            "overhead_ms_per_sample": float(
                (overhead_elapsed / self._num_updates) * 1000.0
                if self._num_updates
                else 0.0
            ),
        }

        for stage, stage_elapsed in self._stage_elapsed_s.items():
            results[f"{stage}_elapsed_s"] = float(stage_elapsed)
            results[f"{stage}_ms_per_sample"] = float(
                (stage_elapsed / self._num_updates) * 1000.0
                if self._num_updates
                else 0.0
            )

        return results
