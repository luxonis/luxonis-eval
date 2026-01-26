import time


class ThroughputMetric:
    """Throughput evaluation metric."""

    def __init__(self) -> None:
        """Initialize the metric."""
        self.reset()

    def reset(self) -> None:
        self._num_updates = 0

        self._t0 = time.perf_counter()
        self._t_last = self._t0

    def update(self) -> None:
        self._num_updates += 1

    def compute(self) -> dict[str, float]:
        elapsed = max(time.perf_counter() - self._t0, 1e-12)
        sps = self._num_updates / elapsed
        msp = (
            (elapsed / self._num_updates) * 1000.0
            if self._num_updates
            else 0.0
        )

        return {
            "elapsed_s": float(elapsed),
            "samples": int(self._num_updates),
            "samples_per_s": float(sps),
            "ms_per_sample": float(msp),
        }
