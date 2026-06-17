import json
import re
from importlib.resources import files
from typing import Any

import numpy as np
from loguru import logger
from luxonis_ml.data.loaders import LuxonisLoader
from luxonis_ml.data.utils import split_task
from luxonis_ml.typing import Params

from luxonis_eval.config import EvaluatorConfig
from luxonis_eval.loaders.base_loader import BaseEvalLoader
from luxonis_eval.metrics.metrics_utils import normalized_xywh_to_coco_xywh


def normalize_target(
    target: dict[str, np.ndarray],
    *,
    loader: BaseEvalLoader | LuxonisLoader | None,
    loader_task_name: str | None,
) -> dict[str, np.ndarray]:
    if isinstance(loader, LuxonisLoader) and loader_task_name is not None:
        return normalize_luxonis_task_labels(target, loader_task_name)
    return target


def resolve_class_mapping(
    loader: BaseEvalLoader | LuxonisLoader,
    *,
    loader_params: Params,
    loader_task_name: str | None,
) -> tuple[dict[int, str], dict[int, str], dict[int, int] | None]:
    if isinstance(loader, LuxonisLoader):
        return resolve_luxonis_loader_class_mapping(
            loader,
            **loader_class_mapping_params(
                loader_params=loader_params,
                loader_task_name=loader_task_name,
            ),
        )

    return loader.get_class_mapping(**loader_params)


def build_metric_contexts(
    evaluator_cfg: EvaluatorConfig,
    *,
    width: int | None,
    height: int | None,
    ldf_class_map: dict[int, str],
    class_map: dict[int, str],
    class_index_map: dict[int, int] | None,
) -> list[dict[str, Any]]:
    if width is None or height is None:
        raise RuntimeError("Engine input shape is unavailable after setup.")

    return [
        get_metric_ctx(
            base_ctx=metric_cfg.params,
            width=width,
            height=height,
            ldf_class_map=ldf_class_map,
            class_map=class_map,
            class_index_map=class_index_map,
        )
        for metric_cfg in evaluator_cfg.metrics
    ]


def loader_class_mapping_params(
    *,
    loader_params: Params,
    loader_task_name: str | None,
) -> Params:
    params = dict(loader_params)
    params.pop("filter_task_names", None)
    if loader_task_name is not None:
        params["selected_task_name"] = loader_task_name
    return params


def select_evaluator_outputs(
    raw_output: Any,
    outputs: list[str] | None,
) -> Any:
    if not outputs:
        return raw_output

    if isinstance(raw_output, list):
        selected_outputs = []
        for output_name in outputs:
            match = re.fullmatch(r"output_?(?P<index>\d+)", output_name)
            if match is None:
                raise ValueError(
                    "List-based evaluator output selection currently supports "
                    "only names like 'output0' or 'output_0'. "
                    f"Received {output_name!r}."
                )
            index = int(match.group("index"))
            if index >= len(raw_output):
                raise ValueError(
                    f"Evaluator requested {output_name!r}, but the engine "
                    f"produced only {len(raw_output)} outputs."
                )
            selected_outputs.append(raw_output[index])
        return selected_outputs

    logger.warning(
        "Evaluator outputs filtering is not implemented for this engine "
        "output type yet. Using all raw outputs."
    )
    return raw_output


def resolve_luxonis_task_name(
    dataset_name: str,
    dataset_classes: dict[str, dict[str, int]],
    *,
    task_name: str | None = None,
) -> str:
    available_tasks = list(dataset_classes.keys())

    if task_name is not None:
        selected_task = task_name
    elif "" in dataset_classes:
        selected_task = ""
    elif len(available_tasks) == 1:
        selected_task = available_tasks[0]
    else:
        available = ", ".join(repr(task) for task in available_tasks)
        raise ValueError(
            "Dataset exposes multiple Luxonis tasks "
            f"({available}). Set pipeline.evaluators[*].task_name explicitly."
        )

    if selected_task not in dataset_classes:
        raise ValueError(
            f"Task {selected_task!r} was requested but is not present in "
            f"dataset {dataset_name!r}."
        )

    return selected_task


def normalize_luxonis_task_labels(
    labels: dict[str, np.ndarray],
    selected_task_name: str,
) -> dict[str, np.ndarray]:
    normalized_labels: dict[str, np.ndarray] = {}

    for task, array in labels.items():
        task_name, task_type = split_task(task)
        normalized_task = task
        if task_name == selected_task_name:
            normalized_task = f"/{task_type}"
        normalized_labels[normalized_task] = array

    return normalized_labels


def resolve_luxonis_loader_class_mapping(
    dataloader: LuxonisLoader,
    **kwargs: Any,
) -> tuple[dict[int, str], dict[int, str], dict[int, int] | None]:
    if not isinstance(dataloader, LuxonisLoader):
        raise NotImplementedError(
            "Built-in class mapping resolution is only implemented for "
            "`LuxonisLoader`. Please provide a custom `get_class_mapping()` "
            "implementation for other loader types inheriting from "
            "`BaseEvalLoader`."
        )

    dataset_classes = dataloader.dataset.get_classes()
    selected_task = resolve_luxonis_task_name(
        dataloader.dataset.dataset_name,
        dataset_classes,
        task_name=kwargs.get("selected_task_name"),
    )
    ldf_class_map = {v: k for k, v in dataset_classes[selected_task].items()}

    if "imagenet" in dataloader.dataset.dataset_name:
        native_class_map = get_dataset_class_mapping("imagenet")
    elif "coco" in dataloader.dataset.dataset_name:
        native_class_map = get_dataset_class_mapping("coco")
    else:
        native_class_map = kwargs.get("class_mapping")
        if native_class_map:
            logger.info(
                f"Dataset '{dataloader.dataset.dataset_name}' does not match "
                "known datasets for automatic class mapping. Using the "
                "provided 'loader.params.class_mapping' argument."
            )
        else:
            logger.warning(
                f"Dataset '{dataloader.dataset.dataset_name}' does not match "
                "known datasets for automatic class mapping and no "
                "'loader.params.class_mapping' was provided. Falling back to "
                "the dataset's LDF class order as the native class mapping."
            )
            native_class_map = ldf_class_map.copy()

    class_index_map = get_class_index_mapping(ldf_class_map, native_class_map)
    return ldf_class_map, native_class_map, class_index_map


def get_dataset_class_mapping(dataset_name: str) -> dict[int, str]:
    supported = {"coco", "imagenet"}
    if dataset_name not in supported:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported values are "
            f"{supported}."
        )

    mapping_file = files("luxonis_eval.metadata").joinpath(
        f"{dataset_name}_class_mappings.json"
    )
    with mapping_file.open() as f:
        class_mapping = json.load(f)
    return {int(k): v for k, v in class_mapping.items()}


def get_class_index_mapping(
    ldf_class_map: dict[int, str],
    native_class_map: dict[int, str],
) -> dict[int, int]:
    ldf_to_native_index_map: dict[int, int] = {}
    for k, v in ldf_class_map.items():
        if k == 0 and v == "background":
            continue

        for key, value in native_class_map.items():
            values = value.split(", ")
            if v in values:
                ldf_to_native_index_map[k] = key
                break

        if k not in ldf_to_native_index_map:
            raise ValueError(f"Label {v} not found in native class map.")

    return ldf_to_native_index_map


def get_metric_ctx(base_ctx: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    class_index_map = kwargs.get("class_index_map")
    class_map = kwargs.get("class_map") or {}
    ldf_class_map = kwargs.get("ldf_class_map") or {}
    width = kwargs.get("width", -1)
    height = kwargs.get("height", -1)

    ldf_name_to_idx = {v: k for k, v in ldf_class_map.items()}

    return {
        **base_ctx,
        "class_map": class_map,
        "class_index_map": class_index_map,
        "width": width,
        "height": height,
        "category_ids": sorted(class_map.keys()),
        "target_converter": normalized_xywh_to_coco_xywh,
        "target_bg": ldf_name_to_idx.get("background"),
        "target_class_map": ldf_class_map,
    }
