from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def check_loader_output(output: object) -> None:
    """Validates the output of a loader.

    Parameters
    ----------
    output : object
        The output to validate.

    Raises
    ------
    TypeError
        If the output is not of type 'luxonis_ml.typing.LoaderOutput': a
        tuple containing either a single image as an `np.ndarray` or a
        dictionary mapping image names to `np.ndarray`, along with a
        dictionary of task group names and their annotations.
    """

    def _validate_image_array(image: np.ndarray) -> None:
        if image.ndim not in (2, 3):
            raise TypeError(
                "Image arrays must be `HW` (grayscale) or `HWC` (color), "
                f"got shape {image.shape}."
            )

    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError(
            f"LoaderOutput must be a tuple of length 2, got {type(output)}"
        )

    images, labels = output

    if isinstance(images, np.ndarray):
        _validate_image_array(images)
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
    """Validates the output of a loader's load_classes method.

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
