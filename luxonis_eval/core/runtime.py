import json
from importlib.resources import files

import numpy as np
from loguru import logger
from luxonis_ml.data.loaders import LuxonisLoader
from luxonis_ml.data.utils import split_task
from luxonis_ml.typing import Params

from luxonis_eval.core.context import EvalContext
from luxonis_eval.core.targets import prepare_target
from luxonis_eval.engines.base_engine import ModelSpec
from luxonis_eval.engines.io import EngineOutput
from luxonis_eval.loaders.base_loader import BaseEvalLoader


def normalize_target(
    target: dict[str, np.ndarray],
    loader: BaseEvalLoader | LuxonisLoader | None,
    loader_task_name: str | None,
    model_spec: ModelSpec | None = None,
) -> dict[str, np.ndarray]:
    normalized_target = target
    if isinstance(loader, LuxonisLoader) and loader_task_name is not None:
        normalized_target = normalize_luxonis_task_labels(
            target, loader_task_name
        )

    if model_spec is None:
        return normalized_target

    return prepare_target(
        normalized_target,
        width=model_spec.width,
        height=model_spec.height,
    )


def resolve_class_mapping(
    loader: BaseEvalLoader | LuxonisLoader,
    loader_params: Params,
    loader_task_name: str | None,
) -> tuple[dict[int, str], dict[int, str], dict[int, int] | None]:
    if isinstance(loader, LuxonisLoader):
        return resolve_luxonis_loader_class_mapping(
            loader,
            loader_task_name=loader_task_name,
            class_mapping=loader_params.get("class_mapping", None),  # type: ignore
        )

    return loader.get_class_mapping(**loader_params)


def build_eval_context(
    model_spec: ModelSpec,
    ldf_class_map: dict[int, str],
    class_map: dict[int, str],
    class_index_map: dict[int, int] | None,
) -> EvalContext:
    ldf_name_to_idx = {v: k for k, v in ldf_class_map.items()}
    return EvalContext(
        model_spec=model_spec,
        class_map=class_map,
        target_class_map=ldf_class_map,
        class_index_map=class_index_map,
        category_ids=tuple(sorted(class_map.keys())),
        target_background_index=ldf_name_to_idx.get("background"),
    )


def select_evaluator_outputs(
    raw_output: EngineOutput,
    outputs: list[str] | None,
) -> EngineOutput:
    if not outputs:
        return raw_output

    return raw_output.select(outputs)


def resolve_luxonis_task_name(
    dataset_name: str,
    dataset_classes: dict[str, dict[str, int]],
    task_name: str | None = None,
) -> str:
    available_tasks = list(dataset_classes.keys())

    if task_name is not None:
        resolved_task = task_name
    elif "" in dataset_classes:
        resolved_task = ""
    elif len(available_tasks) == 1:
        resolved_task = available_tasks[0]
    else:
        available = ", ".join(repr(task) for task in available_tasks)
        raise ValueError(
            "Dataset exposes multiple Luxonis tasks "
            f"({available}). Set pipeline.evaluators[*].task_name explicitly."
        )

    if resolved_task not in dataset_classes:
        raise ValueError(
            f"Task {resolved_task!r} was requested but is not present in "
            f"dataset {dataset_name!r}."
        )

    return resolved_task


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
    loader_task_name: str | None,
    class_mapping: dict[int, str] | None = None,
) -> tuple[dict[int, str], dict[int, str], dict[int, int] | None]:
    if not isinstance(dataloader, LuxonisLoader):
        raise NotImplementedError(
            "Built-in class mapping resolution is only implemented for "
            "`LuxonisLoader`. Please provide a custom `get_class_mapping()` "
            "implementation for other loader types inheriting from "
            "`BaseEvalLoader`."
        )
    loader_task_name = loader_task_name or ""
    dataset_classes = dataloader.dataset.get_classes()
    ldf_class_map = {
        v: k for k, v in dataset_classes[loader_task_name].items()
    }

    if "imagenet" in dataloader.dataset.dataset_name:
        native_class_map = get_dataset_class_mapping("imagenet")
    elif "coco" in dataloader.dataset.dataset_name:
        native_class_map = get_dataset_class_mapping("coco")
    else:
        native_class_map = class_mapping
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
