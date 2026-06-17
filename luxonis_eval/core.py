import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.loaders import LuxonisLoader
from luxonis_ml.typing import Params, PathType
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from luxonis_eval.engines.base_engine import BaseEngine
from luxonis_eval.loaders.base_loader import BaseEvalLoader
from luxonis_eval.metrics import ThroughputMetric
from luxonis_eval.metrics.base_metric import BaseMetric
from luxonis_eval.parsers.base_parser import BaseParser
from luxonis_eval.registry import (
    DATALOADERS_REGISTRY,
    ENGINES_REGISTRY,
    METRICS_REGISTRY,
    PARSERS_REGISTRY,
    VISUALIZERS_REGISTRY,
    from_registry,
)
from luxonis_eval.utils.config import EvalConfig, EvaluatorConfig
from luxonis_eval.utils.utils import (
    get_metric_ctx,
    get_model_name,
    make_report_table,
    normalize_luxonis_task_labels,
    resolve_luxonis_loader_class_mapping,
    resolve_luxonis_task_name,
)
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
            self.evaluator_cfg = self._resolve_evaluator_config()
            self.engine = self._create_engine()
            self.engine.setup()
            self._validate_engine_setup()
            self.backend = self.cfg.pipeline.engine.name
            self.model_name = get_model_name(
                self.cfg.pipeline.engine.model_path
            )

            self.loader = self._create_loader()
            self.parser = self._create_parser()
            self.metrics = self._create_metrics()
            if not self.metrics:
                raise ValueError(
                    "At least one metric must be specified in the configuration."
                )
            self.throughput_metric = ThroughputMetric()
            logger.info("Throughput metric initialized.")
            self.visualizers = self._create_visualizers()

            self._resolve_class_mapping()
            self.metric_contexts = self._build_metric_contexts()
            self._run_static_compatibility_warnings()
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

    def evaluate(self) -> dict[str, Any]:
        """Run the evaluation loop and return structured results."""
        self._require_setup()
        self._reset_runtime_metrics()
        return self._run_pipeline()

    def _run_pipeline(self) -> dict[str, Any]:
        if self.cfg.pipeline.benchmark is not None:
            logger.warning(
                "pipeline.benchmark is configured, but benchmark execution is not implemented yet. Running quality evaluators only."
            )
        return self._run_evaluators()

    def _run_evaluators(self) -> dict[str, Any]:
        """Run the configured quality evaluator."""
        self._require_setup()

        assert self.engine is not None
        assert self.loader is not None
        assert self.parser is not None
        assert self.throughput_metric is not None
        assert self.backend is not None
        assert self.evaluator_cfg is not None
        assert self.model_name is not None

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            ptask = progress.add_task(
                f"Running {self.backend.upper()} inference ({self.model_name})...",
                total=len(self.loader),
            )

            for sample in self.loader:
                img: np.ndarray = sample[0]  # type: ignore
                target = self._normalize_target(sample[1])

                inference_t0 = time.perf_counter()
                raw_output = self.engine.infer_once(img)
                inference_elapsed = time.perf_counter() - inference_t0

                parsing_t0 = time.perf_counter()
                predictions = self.parser.parse(
                    self._select_evaluator_outputs(raw_output),
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
                metric_update_elapsed = time.perf_counter() - metric_update_t0

                self.throughput_metric.update(
                    inference=inference_elapsed,
                    parsing=parsing_elapsed,
                    metric_update=metric_update_elapsed,
                )

                active_visualizer_cfgs = [
                    visualizer_cfg
                    for visualizer_cfg in self.evaluator_cfg.visualizers
                    if visualizer_cfg.visualize
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

                progress.update(ptask, advance=1)

        metric_compute_t0 = time.perf_counter()
        results = [
            (metric.__class__.__name__, metric.compute())
            for metric in self.metrics
        ]
        metric_compute_elapsed = time.perf_counter() - metric_compute_t0
        throughput = self.throughput_metric.compute(
            metric_compute=metric_compute_elapsed
        )

        device = self.engine.platform_name
        if device is None:
            raise RuntimeError(
                "Engine platform name is unavailable after setup."
            )

        report = make_report_table(
            backend=self.backend,
            model_name=self.model_name,
            device=device,
            tp=throughput,
            results=results,
        )

        logger.warning(
            "Throughput values are end-to-end pipeline measurements and not isolated model-only benchmarks. Lower numbers than modelconverter benchmark results are expected."
        )
        logger.info(f"\n{report}")

        return {
            "evaluator_name": self.evaluator_cfg.name,
            "backend": self.backend,
            "model_name": self.model_name,
            "device": device,
            "metrics": results,
            "throughput": throughput,
            "report": report,
        }

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

    def _create_engine(self) -> BaseEngine:
        try:
            engine = from_registry(
                ENGINES_REGISTRY,
                self.cfg.pipeline.engine.name,
                self.cfg.pipeline.engine.model_path,
                **self.cfg.pipeline.engine.params,
            )
        except KeyError as e:
            raise ValueError(
                f"Unknown engine: {self.cfg.pipeline.engine.name}. "
                f"Available engines: {list(ENGINES_REGISTRY._module_dict)}"
            ) from e

        if not isinstance(engine, BaseEngine):
            raise TypeError(
                f"{self.cfg.pipeline.engine.name} engine must be an instance of BaseEngine."
            )

        logger.info(
            f"{self.cfg.pipeline.engine.name} inference engine initialized."
        )
        return engine

    def _validate_engine_setup(self) -> None:
        assert self.engine is not None
        if self.engine.width is None or self.engine.height is None:
            raise RuntimeError(
                "Engine setup did not populate the input shape."
            )
        if self.engine.platform_name is None:
            raise RuntimeError(
                "Engine setup did not populate the platform name."
            )

    def _create_loader(self) -> BaseEvalLoader | LuxonisLoader:
        assert self.engine is not None
        assert self.evaluator_cfg is not None
        if self.engine.width is None or self.engine.height is None:
            raise RuntimeError(
                "Engine input shape is unavailable after setup."
            )

        try:
            if self.cfg.pipeline.loader.name == "LuxonisLoader":
                loader_params = dict(self.cfg.pipeline.loader.params)
                loader_params.pop("filter_task_names", None)
                dataset_name: str = loader_params.pop("dataset_name")  # type: ignore
                dataset_kwargs = {}
                for key in (
                    "team_id",
                    "bucket_type",
                    "bucket_storage",
                    "delete_local",
                    "delete_remote",
                ):
                    if key in loader_params:
                        dataset_kwargs[key] = loader_params.pop(key)

                dataset = LuxonisDataset(dataset_name, **dataset_kwargs)
                selected_task_name = resolve_luxonis_task_name(
                    dataset_name,
                    dataset.get_classes(),
                    task_name=self.evaluator_cfg.task_name,
                )
                loader_params["filter_task_names"] = [selected_task_name]
                self.loader_task_name = selected_task_name

                augmentation_config = []
                if self.cfg.pipeline.loader.preprocessing.normalize.active:
                    augmentation_config.append(
                        {
                            "name": "Normalize",
                            "params": self.cfg.pipeline.loader.preprocessing.normalize.params,
                        }
                    )
                dataloader = LuxonisLoader(
                    dataset,
                    view=loader_params.pop("view", ["val"]),  # type: ignore
                    augmentation_config=augmentation_config,
                    height=self.engine.height,
                    width=self.engine.width,
                    keep_aspect_ratio=self.cfg.pipeline.loader.preprocessing.keep_aspect_ratio,
                    color_space=self.cfg.pipeline.loader.preprocessing.color_space,
                    **loader_params,  # type: ignore
                )
            else:
                dataloader = from_registry(
                    DATALOADERS_REGISTRY,
                    self.cfg.pipeline.loader.name,
                    **self.cfg.pipeline.loader.params,
                )
                if not isinstance(dataloader, BaseEvalLoader):
                    raise TypeError(
                        f"{self.cfg.pipeline.loader.name} custom dataloader must be an instance of BaseEvalLoader."
                    )
        except KeyError as e:
            raise ValueError(
                f"Unknown loader: {self.cfg.pipeline.loader.name}. "
                f"Available loaders: {list(DATALOADERS_REGISTRY._module_dict)}"
            ) from e

        if self.cfg.pipeline.loader.name == "LuxonisLoader":
            logger.info(
                f"LuxonisLoader dataloader initialized with task_name={self.loader_task_name!r}."
            )
        else:
            logger.info(
                f"{self.cfg.pipeline.loader.name} dataloader initialized."
            )
        logger.info(
            f"Dataset loaded with {len(dataloader)} samples and images of shape {self.engine.height}x{self.engine.width}."
        )
        return dataloader

    def _normalize_target(
        self, target: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        if (
            isinstance(self.loader, LuxonisLoader)
            and self.loader_task_name is not None
        ):
            return normalize_luxonis_task_labels(
                target,
                self.loader_task_name,
            )
        return target

    def _create_parser(self) -> BaseParser:
        assert self.evaluator_cfg is not None
        try:
            parser = from_registry(
                PARSERS_REGISTRY,
                self.evaluator_cfg.parser.name,
                **self.evaluator_cfg.parser.params,
            )
        except KeyError as e:
            raise ValueError(
                f"Unknown parser: {self.evaluator_cfg.parser.name}. "
                f"Available parsers: {list(PARSERS_REGISTRY._module_dict)}"
            ) from e

        if not isinstance(parser, BaseParser):
            raise TypeError(
                f"{self.evaluator_cfg.parser.name} parser must be an instance of BaseParser."
            )

        logger.info(f"{self.evaluator_cfg.parser.name} parser initialized.")
        return parser

    def _create_metrics(self) -> list[BaseMetric]:
        assert self.evaluator_cfg is not None
        metrics: list[BaseMetric] = []
        for metric_cfg in self.evaluator_cfg.metrics:
            try:
                metric = from_registry(
                    METRICS_REGISTRY,
                    metric_cfg.name,
                    **metric_cfg.params,
                )
            except KeyError as e:
                raise ValueError(
                    f"Unknown metric: {metric_cfg.name}. "
                    f"Available metrics: {list(METRICS_REGISTRY._module_dict)}"
                ) from e

            if not isinstance(metric, BaseMetric):
                raise TypeError(
                    f"{metric_cfg.name} metric must be an instance of BaseMetric."
                )

            logger.info(f"{metric_cfg.name} metric initialized.")
            metrics.append(metric)

        return metrics

    def _create_visualizers(self) -> list[BaseVisualizer]:
        assert self.evaluator_cfg is not None
        enabled_visualizers = [
            visualizer_cfg
            for visualizer_cfg in self.evaluator_cfg.visualizers
            if visualizer_cfg.visualize
        ]
        if not enabled_visualizers:
            logger.info("Visualization is disabled.")
            return []

        visualizers: list[BaseVisualizer] = []
        for visualizer_cfg in enabled_visualizers:
            try:
                visualizer = from_registry(
                    VISUALIZERS_REGISTRY,
                    visualizer_cfg.name,
                    **visualizer_cfg.params,
                )
            except KeyError as e:
                raise ValueError(
                    f"Unknown visualizer: {visualizer_cfg.name}. "
                    f"Available visualizers: {list(VISUALIZERS_REGISTRY._module_dict)}"
                ) from e

            if not isinstance(visualizer, BaseVisualizer):
                raise TypeError(
                    f"{visualizer_cfg.name} visualizer must be an instance of BaseVisualizer."
                )

            logger.info(f"{visualizer_cfg.name} visualizer initialized.")
            visualizers.append(visualizer)

        return visualizers

    def _resolve_class_mapping(self) -> None:
        if self.loader is None:
            raise RuntimeError("Loader is unavailable before setup completes.")

        if isinstance(self.loader, LuxonisLoader):
            (
                self.ldf_class_map,
                self.class_map,
                self.class_index_map,
            ) = resolve_luxonis_loader_class_mapping(
                self.loader,
                **self._loader_class_mapping_params(),
            )
            return

        (
            self.ldf_class_map,
            self.class_map,
            self.class_index_map,
        ) = self.loader.get_class_mapping(**self.cfg.pipeline.loader.params)

    def _build_metric_contexts(self) -> list[dict[str, Any]]:
        assert self.engine is not None
        assert self.evaluator_cfg is not None
        if self.engine.width is None or self.engine.height is None:
            raise RuntimeError(
                "Engine input shape is unavailable after setup."
            )

        return [
            get_metric_ctx(
                base_ctx=metric_cfg.params,
                width=self.engine.width,
                height=self.engine.height,
                ldf_class_map=self.ldf_class_map,
                class_map=self.class_map,
                class_index_map=self.class_index_map,
            )
            for metric_cfg in self.evaluator_cfg.metrics
        ]

    def _run_static_compatibility_warnings(self) -> None:
        if (
            self.cfg.pipeline.engine.name == "depthai"
            and self.cfg.pipeline.loader.preprocessing.normalize.active
        ):
            logger.warning(
                "Normalization is usually part of the model's preprocessing pipeline in DepthAI. Consider disabling normalization in the dataset config."
            )
        if (
            self.cfg.pipeline.engine.name == "depthai"
            and self.cfg.pipeline.loader.preprocessing.color_space == "RGB"
        ):
            logger.warning(
                "Color space is set to RGB in the dataset config. DepthAI expects BGR color space."
            )

    def _sanity_check_pipeline(self) -> None:
        if self.loader is None or self.engine is None or self.parser is None:
            raise RuntimeError(
                "Pipeline components are unavailable before sanity check."
            )

        if len(self.loader) == 0:
            raise ValueError(
                "Evaluation loader is empty. Pipeline sanity check requires at least one sample."
            )

        logger.info("Running pipeline sanity check on one real sample.")

        img, target = self.loader[0]
        target = self._normalize_target(target)
        raw_output = self.engine.infer_once(img)
        assert self.evaluator_cfg is not None
        predictions = self.parser.parse(
            self._select_evaluator_outputs(raw_output),
            class_map=self.class_map,
            **self.evaluator_cfg.parser.params,
        )

        for metric, metric_ctx in zip(
            self.metrics, self.metric_contexts, strict=True
        ):
            missing = set(metric.required_target_keys()) - set(target)
            if missing:
                raise ValueError(
                    f"Target is missing required keys for {metric.__class__.__name__}: "
                    f"{sorted(missing)}. Got keys: {sorted(target.keys())}."
                )

            metric.update(
                predictions=predictions,
                target=target,
                **metric_ctx,
            )
            metric.compute()
            metric.reset()

    def _resolve_evaluator_config(self) -> EvaluatorConfig:
        evaluators = self.cfg.pipeline.evaluators
        if not evaluators:
            raise NotImplementedError(
                "pipeline.evaluators is required for quality evaluation. Benchmark-only pipeline execution is not implemented yet."
            )
        if len(evaluators) > 1:
            raise NotImplementedError(
                "Multiple pipeline.evaluators are not implemented yet."
            )
        return evaluators[0]

    def _loader_class_mapping_params(self) -> Params:
        params = dict(self.cfg.pipeline.loader.params)
        params.pop("filter_task_names", None)
        if self.loader_task_name is not None:
            params["selected_task_name"] = self.loader_task_name
        return params

    def _select_evaluator_outputs(self, raw_output: Any) -> Any:
        assert self.evaluator_cfg is not None
        if not self.evaluator_cfg.outputs:
            return raw_output

        if isinstance(raw_output, list):
            selected_outputs = []
            for output_name in self.evaluator_cfg.outputs:
                match = re.fullmatch(r"output_?(?P<index>\d+)", output_name)
                if match is None:
                    raise ValueError(
                        "List-based evaluator output selection currently supports only names like 'output0' or 'output_0'. "
                        f"Received {output_name!r}."
                    )
                index = int(match.group("index"))
                if index >= len(raw_output):
                    raise ValueError(
                        f"Evaluator requested {output_name!r}, but the engine produced only {len(raw_output)} outputs."
                    )
                selected_outputs.append(raw_output[index])
            return selected_outputs

        logger.warning(
            "Evaluator outputs filtering is not implemented for this engine output type yet. Using all raw outputs."
        )
        return raw_output

    def _clear_runtime_fields(self) -> None:
        self.engine: BaseEngine | None = None
        self.loader: BaseEvalLoader | LuxonisLoader | None = None
        self.parser: BaseParser | None = None
        self.metrics: list[BaseMetric] = []
        self.throughput_metric: ThroughputMetric | None = None
        self.visualizers: list[BaseVisualizer] = []
        self.evaluator_cfg: EvaluatorConfig | None = None

        self.backend: str | None = None
        self.model_name: str | None = None
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
