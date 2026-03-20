import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
from loguru import logger
from luxonis_ml.data.loaders import LuxonisLoader
from tabulate import tabulate

from luxonis_eval.metrics.metrics_utils import yolo_norm_to_coco_xywh


def section(
    title: str, width: int = 35, line_char: str = "═"
) -> list[list[str]]:
    """Create a section header row for a tabulated report.

    Parameters
    ----------
    title : str
        Section title.
    width : int, optional
        Total header width.
    line_char : str, optional
        Character to use for the line.

    Returns
    -------
    list[list[str]]
        Rows suitable for appending to a tabulate row list.
    """
    label = f" {title} "
    centered = label.center(width, line_char)
    return [[centered, ""]]


def make_report_table(
    *,
    backend: str,
    task_name: str,
    device: str,
    tp: dict[str, float],
    results: list[dict[str, Any]],
) -> str:
    """Build a formatted report table.

    Parameters
    ----------
    backend : str
        Backend identifier.
    task_name : str
        Inference task name.
    device : str
        Inference device descriptor.
    tp : dict[str, float]
        Throughput and latency related metrics.
    results : list[dict[str, Any]]
        Quality metric results.

    Returns
    -------
    str
        Rendered table as text.
    """
    rows: list[list[str]] = []

    rows += section("SETTINGS")
    rows += [
        ["Backend", str(backend).upper()],
        ["Task", task_name],
        ["Device", str(device)],
    ]

    rows += section("PERFORMANCE")
    rows += [
        ["Throughput", f"{tp['samples_per_s']:.2f} samples/s"],
        ["Latency", f"{tp['ms_per_sample']:.2f} ms/sample"],
    ]

    rows += section("QUALITY")
    for result in results:
        metric_name = result.pop("metric")
        rows += section(metric_name, line_char="-")
        for k, v in result.items():
            val = f"{v * 100:.2f}%" if isinstance(v, float) else str(v)
            rows.append([str(k), val])

    return tabulate(
        rows,
        headers=["Metric", "Value"],
        tablefmt="rounded_outline",
        colalign=("left", "right"),
        disable_numparse=True,
    )


def check_loader_output(output: object) -> None:
    """
    Validates the output of a loader.

    Parameters
    ----------
    output : object
        The output to validate.

    Raises
    ------
    TypeError
        If the output is not of type 'luxonis_ml.typing.LoaderOutput': a tuple containing either a single image as a 'np.ndarray or a dictionary mapping image names to 'np.ndarray', along with a dictionary of task group names and their annotations.
    """
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError(
            f"LoaderOutput must be a tuple of length 2, got {type(output)}"
        )

    images, labels = output

    if isinstance(images, np.ndarray):
        pass  # LoaderSingleOutput
    elif isinstance(images, dict) and all(
        isinstance(k, str) and isinstance(v, np.ndarray)
        for k, v in images.items()
    ):  # LoaderMultiOutput is not yet supported.
        raise TypeError(
            "Multi-image loader output (LoaderMultiOutput) is not yet supported."
        )
    else:
        raise TypeError(
            f"First element must be np.ndarray or dict[str, np.ndarray], got {type(images)}"
        )

    if not isinstance(labels, dict) or not all(
        isinstance(k, str) and isinstance(v, np.ndarray)
        for k, v in labels.items()
    ):
        raise TypeError("Labels must be dict[str, np.ndarray]")


def check_loader_classes(classes: dict[str, int]) -> None:
    """
    Validates the output of a loader's load_classes method.

    Parameters
    ----------
    classes : dict[str, int]
        The class mapping to validate.

    Raises
    ------
    TypeError
        If the classes are not a dict[str, int].
    """
    if not isinstance(classes, dict):
        raise TypeError(
            f"`load_classes()` must return a "
            f"`dict[str, int]`, got {type(classes).__name__}."
        )

    invalid = {
        k: v
        for k, v in classes.items()
        if not isinstance(k, str) or not isinstance(v, int)
    }
    if invalid:
        raise TypeError(
            f"`load_classes()` must return a "
            f"`dict[str, int]` (str name -> int index). "
            f"Found invalid entries: {invalid}"
        )


def get_onnx_input_info(onnx_path: Path | None) -> dict[str, Any]:
    """Retrieve ONNX model input information.

    Parameters
    ----------
    onnx_path : Path
        Path to the ONNX model file.

    Returns
    -------
    dict[str, Any]
        Dictionary containing input information.
    """
    if onnx_path is None:
        raise ValueError("ONNX model path must be provided.")

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )

    input = session.get_inputs()[0]
    return {
        "shape": input.shape,
        "name": input.name,
    }


def get_class_mapping(
    dataloader: LuxonisLoader,
    **kwargs,
) -> tuple[dict, dict, dict | None]:
    """Get native class map and optional class index mapping.

    Parameters
    ----------
    dataloader : LuxonisLoader
        Dataloader to extract class mappings from.
    **kwargs
        Additional dataset-specific parameters.

    Returns
    -------
    tuple[dict, dict, dict | None]
        LDF class map, native class map and class index map (if available).
    """

    if isinstance(dataloader, LuxonisLoader):
        ldf_class_map = dataloader.classes[""]
        ldf_class_map = {v: k for k, v in ldf_class_map.items()}
    else:
        raise NotImplementedError(
            "Built-in `get_class_mapping` is only implemented for `LuxonisLoader`. Please provide a custom implementation for other loader types inheriting from `BaseEvalLoader`."
        )

    if "imagenet" in dataloader.dataset.dataset_name:
        native_class_map = get_dataset_class_mapping("imagenet")
    elif "coco" in dataloader.dataset.dataset_name:
        native_class_map = get_dataset_class_mapping("coco")
    else:
        logger.info(
            f"Dataset '{dataloader.dataset.dataset_name}' does not match known datasets for automatic class mapping. Attempting to use provided class mapping from the 'dataloader_cfg.params.class_mapping' argument."
        )
        native_class_map = kwargs.get("class_mapping", {})

    class_index_map = None
    if native_class_map:
        class_index_map = get_class_index_mapping(
            ldf_class_map, native_class_map
        )
    else:
        logger.warning(
            "No native class map found. Class index mapping will not be available, which may affect metric calculations that require the mapping of LDF class indices to native dataset indices."
        )

    return ldf_class_map, native_class_map, class_index_map


def get_dataset_class_mapping(
    dataset_name: Literal["coco", "imagenet"],
) -> dict[int, str]:
    """Load class index-to-name mapping for a dataset.

    Parameters
    ----------
    dataset_name : Literal["coco", "imagenet"]
        Dataset identifier.

    Returns
    -------
    dict[int, str]
        Mapping from class index to class name.
    """
    if dataset_name == "coco":
        mapping_path = (
            Path(__file__).parent.parent
            / "metadata"
            / "coco_class_mappings.json"
        )
        with open(mapping_path) as f:
            class_mapping = json.load(f)
        return {int(k): v for k, v in class_mapping.items()}

    if dataset_name == "imagenet":
        mapping_path = (
            Path(__file__).parent.parent
            / "metadata"
            / "imagenet_class_mappings.json"
        )
        with open(mapping_path) as f:
            class_mapping = json.load(f)
        return {int(k): v for k, v in class_mapping.items()}

    raise ValueError(
        f"Unsupported dataset '{dataset_name}'. Supported values are 'coco' and 'imagenet'."
    )


def get_class_index_mapping(
    ldf_class_map: dict[int, str], native_class_map: dict[int, str]
) -> dict[int, int]:
    """Map LDF class indices to native dataset indices.

    Parameters
    ----------
    ldf_class_map : dict[int, str]
        LDF class index to class name mapping.
    native_class_map : dict[int, str]
        Native class index to class name mapping.

    Returns
    -------
    dict[int, int]
        Mapping from LDF class index to native class index.
    """
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
    """Get additional context for metric updates.

    Parameters
    ----------
    base_ctx : dict[str, Any]
        Base context dictionary.
    **kwargs : Any
        Additional context parameters.

    Returns
    -------
    dict[str, Any]
        Context dictionary to pass to metric updates.
    """
    class_index_map = kwargs.get("class_index_map", {})
    class_map = kwargs.get("class_map", {})
    ldf_class_map = kwargs.get("ldf_class_map", {})
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
        "target_converter": yolo_norm_to_coco_xywh,
        "target_bg": ldf_name_to_idx.get("background"),
        "target_class_map": ldf_class_map,
    }
