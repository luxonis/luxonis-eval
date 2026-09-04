import time
from pathlib import Path

import numpy as np
from loguru import logger
from luxonis_ml.data.loaders import LuxonisLoader
from luxonis_ml.typing import Params, PathType

from luxonis_eval.config import EvalConfig, EvaluatorConfig
from luxonis_eval.core.context import EvalContext
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
from luxonis_eval.core.results import EvaluationResult
from luxonis_eval.core.runtime import (
    build_eval_context,
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
from luxonis_eval.parsers.yolo import clear_prediction_metadata
from luxonis_eval.visualizers.base_visualizer import BaseVisualizer
from luxonis_eval.visualizers.utils import prepare_visualization_frame


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
            self.visualizers = create_visualizers(self.evaluator_cfg)
            if not self.metrics and not self.visualizers:
                raise ValueError(
                    "At least one metric or one active visualizer must be "
                    "specified in the configuration."
                )
            self.throughput_metric = ThroughputMetric()
            logger.info("Throughput metric initialized.")

            (
                self.ldf_class_map,
                self.class_map,
                self.class_index_map,
            ) = resolve_class_mapping(
                self.loader,
                loader_params=self.cfg.pipeline.loader.params,
                loader_task_name=self.loader_task_name,
            )
            self.eval_context = build_eval_context(
                model_spec=self.model_spec,
                ldf_class_map=self.ldf_class_map,
                class_map=self.class_map,
                class_index_map=self.class_index_map,
            )
            self._attach_eval_context()
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
        self._reset_runtime_state()
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
                    )
                )
                parsing_elapsed = time.perf_counter() - parsing_t0

                try:
                    metric_update_t0 = time.perf_counter()
                    for metric in self.metrics:
                        metric.update(
                            predictions=predictions,
                            target=target,
                        )
                    metric_update_elapsed = (
                        time.perf_counter() - metric_update_t0
                    )

                    self.throughput_metric.update(
                        inference=inference_elapsed,
                        parsing=parsing_elapsed,
                        metric_update=metric_update_elapsed,
                    )

                    if self.visualizers:
                        normalize_cfg = (
                            self.cfg.pipeline.loader.preprocessing.normalize
                        )
                        normalization_params = (
                            normalize_cfg.params
                            if normalize_cfg.active and engine_name != "depthai"
                            else {}
                        )
                        vis_frame = prepare_visualization_frame(
                            self.engine.vis_frame(),
                            mean=normalization_params.get("mean"),  # type: ignore[arg-type]
                            std=normalization_params.get("std"),  # type: ignore[arg-type]
                        )
                        for visualizer in self.visualizers:
                            visualizer.run(predictions, target, vis_frame)
                finally:
                    clear_prediction_metadata(predictions)
                progress.update(advance=1)

        metric_compute_t0 = time.perf_counter()
        results = [
            (metric.__class__.__name__, metric.compute())
            for metric in self.metrics
        ]
        metric_compute_elapsed = time.perf_counter() - metric_compute_t0
        throughput = self.throughput_metric.compute(
            metric_compute=metric_compute_elapsed
        )
        evaluator_name = (
            self.evaluator_cfg.name or self.evaluator_cfg.task_name or "task_0"
        )

        result = EvaluationResult(
            evaluator_name=evaluator_name,
            engine=engine_name,
            model_name=model_name,
            metrics=results,
            throughput=throughput,
        )
        logger.warning(
            "Throughput values are end-to-end pipeline measurements and not isolated model-only benchmarks. Lower numbers than modelconverter benchmark results are expected."
        )
        logger.info(f"\n{format_evaluation_result(result)}")
        self._log_saved_visualizations(self.visualizers)

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
            for visualizer in self.visualizers:
                visualizer.close()
            self._clear_runtime_fields()
            self._is_setup = False
            self._is_closed = True

    def _sanity_check_pipeline(self) -> None:
        assert self.loader is not None
        assert self.engine is not None
        assert self.parser is not None
        assert self.evaluator_cfg is not None

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
        raw_output = self.engine.infer_once(img)  # type: ignore[arg-type]
        predictions = self.parser.parse(
            select_evaluator_outputs(raw_output, self.evaluator_cfg.outputs)
        )

        try:
            for metric in self.metrics:
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
                )
                metric.compute()
                metric.reset()
            for visualizer in self.visualizers:
                missing = set(visualizer.required_target_keys()) - set(target)
                if missing:
                    raise ValueError(
                        "Target is missing required keys for "
                        f"{visualizer.__class__.__name__}: {sorted(missing)}. "
                        f"Got keys: {sorted(target.keys())}."
                    )
        finally:
            clear_prediction_metadata(predictions)

    def _clear_runtime_fields(self) -> None:
        self.engine: BaseEngine | None = None
        self.loader: BaseEvalLoader | LuxonisLoader | None = None
        self.parser: BaseParser | None = None
        self.metrics: list[BaseMetric] = []
        self.throughput_metric: ThroughputMetric | None = None
        self.visualizers: list[BaseVisualizer] = []
        self.evaluator_cfg: EvaluatorConfig | None = None

        self.eval_context: EvalContext | None = None
        self.model_spec: ModelSpec | None = None
        self.loader_task_name: str | None = None

        self.ldf_class_map: dict[int, str] = {}
        self.class_map: dict[int, str] = {}
        self.class_index_map: dict[int, int] | None = None

    def _attach_eval_context(self) -> None:
        if self.eval_context is None:
            raise RuntimeError(
                "Evaluation context is unavailable before setup."
            )
        if self.parser is None:
            raise RuntimeError("Parser is unavailable before setup.")

        self.parser.attach_context(self.eval_context)
        for metric in self.metrics:
            metric.attach_context(self.eval_context)
        for visualizer in self.visualizers:
            visualizer.attach_context(self.eval_context)

    def _require_setup(self) -> None:
        if not self._is_setup:
            raise RuntimeError(
                "LuxonisEval.setup() must be called before evaluate()."
            )

    def _reset_runtime_state(self) -> None:
        for metric in self.metrics:
            metric.reset()
        for visualizer in self.visualizers:
            visualizer.reset()

        if self.throughput_metric is None:
            raise RuntimeError(
                "Throughput metric is unavailable before setup."
            )
        self.throughput_metric.reset()

    @staticmethod
    def _log_saved_visualizations(
        visualizers: list[BaseVisualizer],
    ) -> None:
        save_dirs = list(
            dict.fromkeys(
                visualizer.save_dir.resolve()
                for visualizer in visualizers
                if visualizer.save
            )
        )
        if not save_dirs:
            return
        destinations = ", ".join(f"'{directory}'" for directory in save_dirs)
        logger.info(f"Visualizations saved to {destinations}.")

    def _progress(
        self, description: str, total: int
    ) -> TQDMProgressAdapter | RichProgressAdapter:
        if self.cfg.runtime.logging.use_rich:
            return RichProgressAdapter(description, total)
        return TQDMProgressAdapter(description, total)
