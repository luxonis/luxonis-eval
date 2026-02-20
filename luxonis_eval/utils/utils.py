import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import onnxruntime as ort
from tabulate import tabulate


@contextmanager
def suppress_stdout() -> Iterator[None]:
    """Suppress stdout within a context."""
    fd = sys.stdout.fileno()
    saved_fd = os.dup(fd)

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), fd)
        yield
    finally:
        os.dup2(saved_fd, fd)
        os.close(saved_fd)


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
    # TODO: Find a better way to set the type of the dataset parameter to be dynamic and extensible to other datasets may be intorduced in the future.
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
