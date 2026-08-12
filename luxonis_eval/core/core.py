import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from luxonis_ml.data.loaders import LuxonisLoader
from luxonis_ml.typing import Params, PathType

from luxonis_eval.config import EvalConfig, EvaluatorConfig
from luxonis_eval.core.factories import (
    create_engine,
    create_loader,
    create_metrics,
    create_parser,
    create_visualizers,
)
from luxonis_eval.core.reporting import (
    RichProgressAdapter,
    TQDMProgressAdapter,
    format_evaluation_result,
    get_model_name,
)
from luxonis_eval.core.runtime import (
    build_metric_contexts,
    normalize_target,
    resolve_class_mapping,
    select_evaluator_outputs,
)
from luxonis_eval.core.validation import (
    resolve_evaluator_config,
    validate_engine_setup,
)
from luxonis_eval.engines.base_engine import BaseEngine, ModelSpec
from luxonis_eval.loaders.base_loader import BaseEvalLoader
from luxonis_eval.metrics import ThroughputMetric
from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.parsers.base_parser import BaseParser
from luxonis_eval.core.results import EvaluationResult
from luxonis_eval.visualizers.base_visualizer import BaseVisualizer


class LuxonisEval:
    """Own the evaluation lifecycle and runtime state."""

    def __init__(
        self,
        cfg: PathType | Params | EvalConfig,
        opts: Params | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Construct a new evaluator."""
        if isinstance(cfg, EvalConfig):
            self.cfg = cfg
        else:
            normalized_cfg = str(cfg) if isinstance(cfg, Path) else cfg
            self.cfg = EvalConfig.get_config(normalized_cfg, opts)

        self._is_setup = False
        self._is_closed = False
        self._clear_runtime_fields()

    def setup(self) -> None:
        """Initialize all runtime components required for evaluation."""
        if self._is_setup:
            logger.warning(
                "LuxonisEval.setup() called but the evaluator is already set up."
            )
            return

        self._clear_runtime_fields()
        self._is_closed = False
        logger.info("Setting up evaluation configuration.")

        try:
            self.evaluator_cfg = resolve_evaluator_config(
                self.cfg.pipeline.evaluators
            )
            self.engine = create_engine(self.cfg)
            self.model_spec = self.engine.setup()
            validate_engine_setup(self.model_spec)

            self.loader, self.loader_task_name = create_loader(
                self.cfg,
                self.evaluator_cfg,
                self.model_spec,
            )
            self.parser = create_parser(self.evaluator_cfg)
            self.metrics = create_metrics(self.evaluator_cfg)
            if not self.metrics:
                raise ValueError(
                    "At least one metric must be specified in the configuration."
                )
            self.throughput_metric = ThroughputMetric()
            logger.info("Throughput metric initialized.")
            self.visualizers = create_visualizers(self.evaluator_cfg)

            (
                self.ldf_class_map,
                self.class_map,
                self.class_index_map,
            ) = resolve_class_mapping(
                self.loader,
                loader_params=self.cfg.pipeline.loader.params,
                loader_task_name=self.loader_task_name,
            )
            self.metric_contexts = build_metric_contexts(
                self.evaluator_cfg,
                model_spec=self.model_spec,
                ldf_class_map=self.ldf_class_map,
                class_map=self.class_map,
                class_index_map=self.class_index_map,
            )
            self._sanity_check_pipeline()
        except Exception:
            try:
                self.close()
            except Exception:
                logger.exception(
                    "Failed to clean up after an evaluation setup error."
                )
            raise

        self._is_setup = True
        self._is_closed = False

    def evaluate(self) -> EvaluationResult:
        """Run the evaluation loop and return structured results."""
        self._require_setup()
        self._reset_runtime_metrics()
        return self._run_pipeline()

    def _run_pipeline(self) -> EvaluationResult:
        if self.cfg.pipeline.benchmark is not None:
            logger.warning(
                "pipeline.benchmark is configured, but benchmark execution is not implemented yet. Running quality evaluators only."
            )
        return self._run_evaluators()

    def _run_evaluators(self) -> EvaluationResult:
        """Run the configured quality evaluator."""
        self._require_setup()
        engine_name = self.cfg.pipeline.engine.name
        model_name = get_model_name(self.cfg.pipeline.engine.model_path)

        assert self.engine is not None
        assert self.loader is not None
        assert self.parser is not None
        assert self.throughput_metric is not None
        assert self.evaluator_cfg is not None
        assert self.model_spec is not None

        with self._progress(
            f"Running {engine_name.upper()} inference ({model_name})...",
            total=len(self.loader),
        ) as progress:
            for sample in self.loader:
                img: np.ndarray = sample[0]  # type: ignore
                target = normalize_target(
                    sample[1],
                    loader=self.loader,
                    loader_task_name=self.loader_task_name,
                )

                inference_t0 = time.perf_counter()
                raw_output = self.engine.infer_once(img)
                inference_elapsed = time.perf_counter() - inference_t0

                parsing_t0 = time.perf_counter()
                predictions = self.parser.parse(
                    select_evaluator_outputs(
                        raw_output,
                        self.evaluator_cfg.outputs,
                    ),
                    model_spec=self.model_spec,
                    class_map=self.class_map,
                    **self.evaluator_cfg.parser.params,
                )
                parsing_elapsed = time.perf_counter() - parsing_t0

                metric_update_t0 = time.perf_counter()
                for metric, metric_ctx in zip(
                    self.metrics, self.metric_contexts, strict=True
                ):
                    metric.update(
                        predictions=predictions,
                        target=target,
                        **metric_ctx,
                    )
                metric_update_elapsed = (
                    time.perf_counter() - metric_update_t0
                )

                self.throughput_metric.update(
                    inference=inference_elapsed,
                    parsing=parsing_elapsed,
                    metric_update=metric_update_elapsed,
                )

                active_visualizer_cfgs = [
                    visualizer_cfg
                    for visualizer_cfg in self.evaluator_cfg.visualizers
                    if visualizer_cfg.active
                ]
                for visualizer, visualizer_cfg in zip(
                    self.visualizers,
                    active_visualizer_cfgs,
                    strict=True,
                ):
                    visualizer.visualize(
                        predictions,
                        self.engine.vis_frame(),
                        **visualizer_cfg.params,
                    )
                progress.update(advance=1)

        metric_compute_t0 = time.perf_counter()
        results = {
            metric.__class__.__name__: metric.compute()
            for metric in self.metrics
        }
        metric_compute_elapsed = time.perf_counter() - metric_compute_t0
        throughput = self.throughput_metric.compute(
            metric_compute=metric_compute_elapsed
        )

        result = EvaluationResult(
            evaluator_name=self.evaluator_cfg.name,
            engine=engine_name,
            model_name=model_name,
            metrics=results,
            throughput=throughput,
        )
        logger.warning(
            "Throughput values are end-to-end pipeline measurements and not isolated model-only benchmarks. Lower numbers than modelconverter benchmark results are expected."
        )
        logger.info(f"\n{format_evaluation_result(result)}")

        return result

    def close(self) -> None:
        """Release owned runtime resources."""
        if self._is_closed:
            logger.warning(
                "LuxonisEval.close() called but the evaluator is already closed."
            )
            return

        try:
            if self.engine is not None:
                self.engine.close()
        finally:
            self._clear_runtime_fields()
            self._is_setup = False
            self._is_closed = True

    def _sanity_check_pipeline(self) -> None:
        assert self.loader is not None
        assert self.engine is not None
        assert self.parser is not None
        assert self.evaluator_cfg is not None
        assert self.model_spec is not None

        if len(self.loader) == 0:
            raise ValueError(
                "Evaluation loader is empty. Pipeline sanity check "
                "requires at least one sample."
            )

        logger.info("Running pipeline sanity check on one real sample.")

        img, target = self.loader[0]
        target = normalize_target(
            target,
            loader=self.loader,
            loader_task_name=self.loader_task_name,
        )
        raw_output = self.engine.infer_once(img)
        predictions = self.parser.parse(
            select_evaluator_outputs(raw_output, self.evaluator_cfg.outputs),
            model_spec=self.model_spec,
            class_map=self.class_map,
            **self.evaluator_cfg.parser.params,
        )

        for metric, metric_ctx in zip(
            self.metrics, self.metric_contexts, strict=True
        ):
            missing = set(metric.required_target_keys()) - set(target)
            if missing:
                raise ValueError(
                    "Target is missing required keys for "
                    f"{metric.__class__.__name__}: {sorted(missing)}. "
                    f"Got keys: {sorted(target.keys())}."
                )

            metric.update(
                predictions=predictions,
                target=target,
                **metric_ctx,
            )
            metric.compute()
            metric.reset()

    def _clear_runtime_fields(self) -> None:
        self.engine: BaseEngine | None = None
        self.loader: BaseEvalLoader | LuxonisLoader | None = None
        self.parser: BaseParser | None = None
        self.metrics: list[BaseMetric] = []
        self.throughput_metric: ThroughputMetric | None = None
        self.visualizers: list[BaseVisualizer] = []
        self.evaluator_cfg: EvaluatorConfig | None = None

        self.model_spec: ModelSpec | None = None
        self.loader_task_name: str | None = None

        self.ldf_class_map: dict[int, str] = {}
        self.class_map: dict[int, str] = {}
        self.class_index_map: dict[int, int] | None = None
        self.metric_contexts: list[dict[str, Any]] = []

    def _require_setup(self) -> None:
        if not self._is_setup:
            raise RuntimeError(
                "LuxonisEval.setup() must be called before evaluate()."
            )

    def _reset_runtime_metrics(self) -> None:
        for metric in self.metrics:
            metric.reset()

        if self.throughput_metric is None:
            raise RuntimeError(
                "Throughput metric is unavailable before setup."
            )
        self.throughput_metric.reset()

    def _progress(
        self, description: str, total: int
    ) -> TQDMProgressAdapter | RichProgressAdapter:
        if self.cfg.runtime.logging.use_rich:
            return RichProgressAdapter(description, total)
        return TQDMProgressAdapter(description, total)
