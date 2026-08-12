from dataclasses import dataclass


MetricValues = dict[str, float]
MetricsResult = dict[str, MetricValues]


@dataclass(slots=True)
class ThroughputResult:
    """End-to-end throughput and latency measurements."""

    elapsed_s: float
    samples: int
    samples_per_s: float
    ms_per_sample: float
    overhead_ms_per_sample: float
    inference_ms_per_sample: float
    parsing_ms_per_sample: float
    metric_update_ms_per_sample: float
    metric_compute_ms_per_sample: float


@dataclass(slots=True)
class EvaluationResult:
    """Structured output returned by ``LuxonisEval.evaluate()``."""

    evaluator_name: str
    engine: str
    model_name: str
    metrics: MetricsResult
    throughput: ThroughputResult
