from typing import Any

from luxonis_ml.typing import Params

from luxonis_eval.config.nn_archive import (
    NNArchiveConfig,
    get_archive_head,
    resolve_archive_color_space,
    resolve_archive_normalization,
)
from luxonis_eval.config.resolved import (
    DataLoaderConfig,
    EvaluatorConfig,
    NormalizeAugmentationConfig,
    ParserConfig,
    PipelineConfig,
    PreProcessingConfig,
    ResolvedEvalConfig,
)
from luxonis_eval.config.shared import DEFAULT_NORMALIZE_PARAMS
from luxonis_eval.config.source import (
    SourceDataLoaderConfig,
    SourceEvalConfig,
    SourceEvaluatorConfig,
    SourceNormalizeAugmentationConfig,
)
from luxonis_eval.registry import PARSERS_REGISTRY


class EvalConfigResolver:
    def __init__(
        self, config_cls: type[ResolvedEvalConfig] = ResolvedEvalConfig
    ) -> None:
        self._config_cls = config_cls

    def resolve(
        self,
        source: SourceEvalConfig,
        nn_archive_cfg: NNArchiveConfig | None,
    ) -> ResolvedEvalConfig:
        prefer_archive = source.runtime.nn_archive_params_override
        pipeline = PipelineConfig(
            loader=self._resolve_loader(
                source.pipeline.loader,
                nn_archive_cfg,
                prefer_archive=prefer_archive,
            ),
            engine=source.pipeline.engine.model_copy(deep=True),
            evaluators=self._resolve_evaluators(
                source.pipeline.evaluators,
                nn_archive_cfg,
                prefer_archive=prefer_archive,
            ),
            benchmark=source.pipeline.benchmark,
        )
        resolved = self._config_cls(
            version=source.version,
            runtime=source.runtime.model_copy(deep=True),
            pipeline=pipeline,
        )
        resolved._set_nn_archive_cfg(nn_archive_cfg)
        return resolved

    def _resolve_loader(
        self,
        source_loader: SourceDataLoaderConfig,
        nn_archive_cfg: NNArchiveConfig | None,
        *,
        prefer_archive: bool,
    ) -> DataLoaderConfig:
        archive_color_space = resolve_archive_color_space(nn_archive_cfg)
        source_color_space = source_loader.preprocessing.color_space
        color_space = self._pick_preferred_value(
            source_value=source_color_space,
            archive_value=archive_color_space,
            prefer_archive=prefer_archive,
            default="RGB",
        )
        keep_aspect_ratio = self._resolve_keep_aspect_ratio(source_loader)
        normalize = self._resolve_normalize_config(
            loader_name=source_loader.name,
            source_normalize=source_loader.preprocessing.normalize,
            nn_archive_cfg=nn_archive_cfg,
            prefer_archive=prefer_archive,
        )
        return DataLoaderConfig(
            name=source_loader.name,
            params=dict(source_loader.params),
            preprocessing=PreProcessingConfig(
                normalize=normalize,
                color_space=color_space,
                keep_aspect_ratio=keep_aspect_ratio,
            ),
        )

    def _resolve_keep_aspect_ratio(
        self, source_loader: SourceDataLoaderConfig
    ) -> bool:
        keep_aspect_ratio = source_loader.preprocessing.keep_aspect_ratio
        if keep_aspect_ratio is not None:
            return keep_aspect_ratio

        if source_loader.name == "LuxonisLoader":
            return True

        return False

    def _resolve_normalize_config(
        self,
        *,
        loader_name: str,
        source_normalize: SourceNormalizeAugmentationConfig | None,
        nn_archive_cfg: NNArchiveConfig | None,
        prefer_archive: bool,
    ) -> NormalizeAugmentationConfig:
        explicit_active = (
            source_normalize.active if source_normalize is not None else None
        )
        explicit_params = (
            dict(source_normalize.params)
            if source_normalize is not None
            and source_normalize.params is not None
            else None
        )
        archive_active, archive_params = (None, None)
        if loader_name == "LuxonisLoader":
            archive_active, archive_params = (
                self._resolve_archive_normalize_state(nn_archive_cfg)
            )

        if prefer_archive:
            if archive_active is not None:
                active = archive_active
            elif explicit_active is not None:
                active = explicit_active
            elif explicit_params is not None:
                active = True
            else:
                active = False
        else:  # noqa: PLR5501
            if explicit_active is False:
                active = False
            elif explicit_active is True or explicit_params is not None:
                active = True
            elif archive_active is not None:
                active = archive_active
            else:
                active = False

        if not active:
            params = {}
        elif prefer_archive:
            params = (
                archive_params
                or explicit_params
                or dict(DEFAULT_NORMALIZE_PARAMS)
            )
        else:
            params = (
                explicit_params
                or archive_params
                or dict(DEFAULT_NORMALIZE_PARAMS)
            )

        return NormalizeAugmentationConfig(active=active, params=params)

    def _pick_preferred_value(
        self,
        *,
        source_value: Any,
        archive_value: Any,
        prefer_archive: bool,
        default: Any = None,
    ) -> Any:
        primary = archive_value if prefer_archive else source_value
        secondary = source_value if prefer_archive else archive_value
        return primary or secondary or default

    def _resolve_archive_normalize_state(
        self, nn_archive_cfg: NNArchiveConfig | None
    ) -> tuple[bool | None, Params | None]:
        if nn_archive_cfg is None:
            return None, None

        mean, scale = resolve_archive_normalization(nn_archive_cfg)
        if mean is None and scale is None:
            return False, None
        if mean is None or scale is None:
            raise ValueError(
                "NNArchive preprocessing normalization must define both "
                "`mean` and `scale`, or neither."
            )
        return True, {"mean": list(mean), "std": list(scale)}

    def _resolve_evaluators(
        self,
        source_evaluators: list[SourceEvaluatorConfig] | None,
        nn_archive_cfg: NNArchiveConfig | None,
        *,
        prefer_archive: bool,
    ) -> list[EvaluatorConfig] | None:
        if source_evaluators is None:
            return None
        return [
            self._resolve_evaluator(
                source_evaluators[0],
                nn_archive_cfg,
                prefer_archive=prefer_archive,
            )
        ]

    def _resolve_evaluator(
        self,
        source_evaluator: SourceEvaluatorConfig,
        nn_archive_cfg: NNArchiveConfig | None,
        *,
        prefer_archive: bool,
    ) -> EvaluatorConfig:
        archive_parser = self._resolve_archive_parser(nn_archive_cfg)
        parser_name = (
            source_evaluator.parser.name
            if source_evaluator.parser is not None
            else archive_parser.name
            if archive_parser is not None
            else None
        )
        if parser_name is None:
            raise ValueError(
                "pipeline.evaluators[0].parser is required unless it can be inferred from the NNArchive head metadata."
            )

        archive_params = (
            dict(archive_parser.params) if archive_parser is not None else {}
        )
        source_params = (
            dict(source_evaluator.parser.params)
            if source_evaluator.parser is not None
            else {}
        )
        parser_params = (
            {**source_params, **archive_params}
            if prefer_archive
            else {**archive_params, **source_params}
        )
        archive_outputs = self._resolve_archive_outputs(nn_archive_cfg)
        outputs = self._pick_preferred_value(
            source_value=source_evaluator.outputs,
            archive_value=archive_outputs,
            prefer_archive=prefer_archive,
        )

        return EvaluatorConfig(
            name=source_evaluator.name,
            task_name=source_evaluator.task_name,
            outputs=outputs,
            parser=ParserConfig(
                name=parser_name,
                params=parser_params,
            ),
            metrics=[
                metric.model_copy(deep=True)
                for metric in source_evaluator.metrics
            ],
            visualizers=[
                visualizer.model_copy(deep=True)
                for visualizer in source_evaluator.visualizers
            ],
        )

    def _resolve_archive_outputs(
        self, nn_archive_cfg: NNArchiveConfig | None
    ) -> list[str] | None:
        archive_head = (
            get_archive_head(nn_archive_cfg) if nn_archive_cfg else None
        )
        if archive_head is None or archive_head.outputs is None:
            return None
        return list(archive_head.outputs)

    def _resolve_archive_parser(
        self, nn_archive_cfg: NNArchiveConfig | None
    ) -> ParserConfig | None:
        archive_head = (
            get_archive_head(nn_archive_cfg) if nn_archive_cfg else None
        )
        if archive_head is None:
            return None

        if archive_head.parser in {"YOLO", "YOLOExtendedParser"}:
            return resolve_yolo_archive_parser(archive_head)

        parser_name = self._map_archive_parser_name(archive_head.parser)
        return ParserConfig(name=parser_name, params={})

    def _map_archive_parser_name(self, parser_name: str) -> str:
        mapping = {
            "Classification": "ClassificationParser",
            "ClassificationParser": "ClassificationParser",
            "SemanticSegmentation": "SemanticSegmentationParser",
            "SemanticSegmentationParser": "SemanticSegmentationParser",
            "SegmentationParser": "SemanticSegmentationParser",
            "Segmentation": "SemanticSegmentationParser",
        }
        resolved_name = mapping.get(parser_name, parser_name)
        if resolved_name not in PARSERS_REGISTRY:
            raise NotImplementedError(
                f"NNArchive parser '{parser_name}' is not supported by luxonis-eval yet."
            )
        return resolved_name


def resolve_yolo_archive_parser(head: Any) -> ParserConfig:
    metadata = resolve_head_metadata(head)
    if "n_keypoints" in metadata:
        parser_name = "YOLOKeypointDetectionParser"
    elif "n_prototypes" in metadata or "mask_outputs" in metadata:
        parser_name = "YOLOInstanceSegmentationParser"
    else:
        parser_name = "YOLODetectionParser"

    params = {}
    for key in (
        "subtype",
        "n_classes",
        "anchors",
        "conf_threshold",
        "iou_threshold",
        "max_det",
    ):
        if key in metadata:
            params[key] = metadata[key]

    return ParserConfig(name=parser_name, params=params)


def resolve_head_metadata(head: Any) -> Params:
    metadata = getattr(head, "metadata", None)
    if metadata is None:
        raise ValueError("NNArchive head metadata is missing.")
    if hasattr(metadata, "model_dump"):
        return metadata.model_dump(exclude_none=True)
    if isinstance(metadata, dict):
        return dict(metadata)
    raise TypeError(
        f"Unsupported NNArchive head metadata type: {type(metadata)}."
    )
