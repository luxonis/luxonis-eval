from pathlib import Path

import yaml
from loguru import logger
from luxonis_ml.typing import Params, PathType
from luxonis_ml.utils.config import LuxonisConfig
from luxonis_ml.utils.filesystem import LuxonisFileSystem

from luxonis_eval.config.nn_archive import load_nn_archive_from_source_config
from luxonis_eval.config.resolved import ResolvedEvalConfig
from luxonis_eval.config.resolver import EvalConfigResolver
from luxonis_eval.config.source import SourceEvalConfig


class EvalConfig(ResolvedEvalConfig):
    """Public runtime evaluation config."""

    @classmethod
    def get_config(
        cls,
        cfg: PathType | Params | None = None,
        overrides: Params | list[str] | tuple[str, ...] | None = None,
    ) -> "EvalConfig":
        raw_data = _load_raw_config_data(cfg)
        overrides_dict = _normalize_overrides(overrides)
        LuxonisConfig._merge_overrides(raw_data, overrides_dict)

        source = SourceEvalConfig(**raw_data)  # type: ignore

        logger.debug(f"Source config:\n{source}")

        nn_archive_cfg = load_nn_archive_from_source_config(source)
        logger.debug(f"NNArchive config:\n{nn_archive_cfg}")

        resolved_config = EvalConfigResolver(config_cls=cls).resolve(
            source, nn_archive_cfg
        )
        logger.debug(f"Resolved config:\n{resolved_config}")

        return resolved_config  # pyright: ignore[reportReturnType]


def _load_raw_config_data(cfg: PathType | Params | None) -> Params:
    if cfg is None:
        return {}

    if isinstance(cfg, Path):
        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    elif isinstance(cfg, str):
        filesystem = LuxonisFileSystem(cfg)
        buffer = filesystem.read_to_byte_buffer()
        data = yaml.safe_load(buffer)
    else:
        data = cfg

    return data or {}


def _normalize_overrides(
    overrides: Params | list[str] | tuple[str, ...] | None,
) -> Params:
    if overrides is None:
        return {}

    if isinstance(overrides, list | tuple):
        if len(overrides) % 2 != 0:
            raise ValueError(
                "Override options should be a list of key-value pairs but it's length is not divisible by 2."
            )
        return dict(zip(overrides[::2], overrides[1::2], strict=True))

    return dict(overrides)
