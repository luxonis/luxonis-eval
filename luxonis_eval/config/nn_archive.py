import json
import tarfile
from pathlib import Path, PurePosixPath

from luxonis_ml.nn_archive.config import Config as NNArchiveConfig
from luxonis_ml.nn_archive.config_building_blocks import HeadType, Input
from luxonis_ml.nn_archive.utils import is_nn_archive
from luxonis_ml.typing import Params

from luxonis_eval.config.source import SourceEvalConfig

_ARCHIVE_NORMALIZATION_SCALE = 255.0


def load_nn_archive_from_source_config(
    source: SourceEvalConfig,
) -> NNArchiveConfig | None:
    model_path = source.pipeline.engine.model_path
    if not is_nn_archive(model_path):
        return None
    return load_nn_archive_config(model_path)


def load_nn_archive_config(model_path: str) -> NNArchiveConfig:
    with tarfile.open(model_path, "r:xz") as archive:
        config_member = next(
            (
                member
                for member in archive.getmembers()
                if Path(member.name).name == "config.json"
            ),
            None,
        )
        if config_member is None:
            raise ValueError(
                f"NNArchive model '{model_path}' does not contain config.json."
            )

        config_file = archive.extractfile(config_member)
        if config_file is None:
            raise ValueError(
                f"NNArchive model '{model_path}' contains an unreadable config.json."
            )
        config_data = json.load(config_file)

    nn_archive_cfg = NNArchiveConfig(**config_data)
    validate_archive_scope(nn_archive_cfg, model_path=model_path)
    return nn_archive_cfg


def validate_archive_scope(
    nn_archive_cfg: NNArchiveConfig, *, model_path: str
) -> None:
    inputs = nn_archive_cfg.model.inputs
    if len(inputs) > 1:
        raise NotImplementedError(
            f"NNArchive model '{model_path}' exposes {len(inputs)} inputs. Only single-input archives are supported."
        )

    heads = nn_archive_cfg.model.heads or []
    if len(heads) > 1:
        raise NotImplementedError(
            f"NNArchive model '{model_path}' exposes {len(heads)} heads. Only single-head archives are supported."
        )


def get_archive_input(nn_archive_cfg: NNArchiveConfig) -> Input | None:
    inputs = nn_archive_cfg.model.inputs
    if not inputs:
        return None
    return inputs[0]


def get_archive_head(nn_archive_cfg: NNArchiveConfig) -> HeadType | None:
    heads = nn_archive_cfg.model.heads or []
    if not heads:
        return None
    return heads[0]


def resolve_archive_color_space(
    nn_archive_cfg: NNArchiveConfig | None,
) -> str | None:
    if nn_archive_cfg is None:
        return None

    archive_input = get_archive_input(nn_archive_cfg)
    if archive_input is None:
        return None

    preprocessing = archive_input.preprocessing
    dai_type = (
        preprocessing.dai_type.upper() if preprocessing.dai_type else None
    )
    if dai_type is not None:
        if "GRAY" in dai_type:
            return "GRAY"
        if "RGB" in dai_type:
            return "RGB"
        if "BGR" in dai_type:
            return "BGR"

    if preprocessing.reverse_channels is True:
        return "RGB"
    if preprocessing.reverse_channels is False:
        return "BGR"

    return None


def resolve_archive_normalization(
    nn_archive_cfg: NNArchiveConfig | None,
) -> tuple[list[float] | None, list[float] | None]:
    if nn_archive_cfg is None:
        return None, None

    archive_input = get_archive_input(nn_archive_cfg)
    if archive_input is None:
        return None, None

    mean = archive_input.preprocessing.mean
    scale = archive_input.preprocessing.scale
    if mean is None or scale is None:
        return mean, scale

    return _normalize_archive_values(mean), _normalize_archive_values(scale)


def resolve_archive_head_metadata(
    nn_archive_cfg: NNArchiveConfig | None,
) -> Params | None:
    if nn_archive_cfg is None:
        return None

    archive_head = get_archive_head(nn_archive_cfg)
    if archive_head is None:
        return None

    return archive_head.metadata.model_dump(exclude_none=True)


def load_onnx_bytes_from_nn_archive(model_path: str | Path) -> bytes:
    model_path = str(model_path)
    nn_archive_cfg = load_nn_archive_config(model_path)

    with tarfile.open(model_path, "r:xz") as archive:
        onnx_member = _resolve_onnx_member(
            archive,
            model_path=model_path,
            nn_archive_cfg=nn_archive_cfg,
        )
        onnx_file = archive.extractfile(onnx_member)
        if onnx_file is None:
            raise ValueError(
                f"NNArchive model '{model_path}' contains an unreadable ONNX payload '{onnx_member.name}'."
            )
        return onnx_file.read()


def _resolve_onnx_member(
    archive: tarfile.TarFile,
    *,
    model_path: str,
    nn_archive_cfg: NNArchiveConfig,
) -> tarfile.TarInfo:
    declared_path = PurePosixPath(
        nn_archive_cfg.model.metadata.path
    ).as_posix()
    matches = [
        member
        for member in archive.getmembers()
        if member.isfile()
        and PurePosixPath(member.name).as_posix() == declared_path
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"NNArchive model '{model_path}' declares ONNX payload '{declared_path}', but the archive contains multiple matches for that path."
        )
    raise ValueError(
        f"NNArchive model '{model_path}' declares ONNX payload '{declared_path}' in config.json, but that file is missing from the archive."
    )


def _normalize_archive_values(values: list[float]) -> list[float]:
    return [value / _ARCHIVE_NORMALIZATION_SCALE for value in values]
