from importlib.metadata import version
from pathlib import Path
from typing import Literal

from cyclopts import App, Group
from loguru import logger
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.loaders import LuxonisLoader

from luxonis_eval.inferer import Inferer

app = App(
    help="Luxonis Eval CLI",
    version=lambda: f"LuxonisEval v{version('luxonis_eval')}",
)
app.meta.group_parameters = Group("Global Parameters", sort_key=0)
app["--help"].group = app.meta.group_parameters
app["--version"].group = app.meta.group_parameters


@app.command()
def eval(
    dataset_name: str,
    nn_archive: str | None = None,
    onnx: str | None = None,
    backend: Literal["depthai", "onnx", "all"] = "depthai",
    device_ip: str | None = None,
):
    """Run evaluation on a dataset using a specified neural network.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to evaluate on.
    nn_archive : str | None, optional
        Path to the neural network NNArchive file. Required if backend is set to 'depthai' or 'all'.
    onnx : str | None, optional
        Path to the ONNX model file, required if backend is 'onnx' or 'all'. Required if backend is set to 'onnx' or 'all'.
    backend : str, optional
        Backend to use for inference. If 'all', runs inference on all available backends.
    device_ip : str | None, optional
        IP address of the device to connect to. Only applicable for RVC4 devices.
    """
    if not nn_archive and not onnx:
        raise ValueError(
            "At least one of nn-archive or onnx must be provided."
        )

    if backend == "all" and not (nn_archive and onnx):
        raise ValueError(
            "Both nn-archive and onnx must be provided when backend is 'all'."
        )

    if nn_archive and backend not in ["depthai", "all"]:
        raise ValueError(
            "NNArchive can only be used with DepthAI backend enabled."
        )
    if onnx and backend not in ["onnx", "all"]:
        raise ValueError("ONNX can only be used with ONNX backend enabled.")

    if nn_archive and not Path(nn_archive).exists():
        raise ValueError(f"NNArchive file '{nn_archive}' does not exist.")
    if onnx and not Path(onnx).exists():
        raise ValueError(f"ONNX model file '{onnx}' does not exist.")

    if not LuxonisDataset.exists(dataset_name):
        raise ValueError(f"Dataset '{dataset_name}' does not exist.")

    # TODO: This code is placeholder, we need to implement a proper way to handle different datasets. The datasets should always inherit from BaseDataset (from luxonis_ml).
    dataset = LuxonisDataset(dataset_name)

    inferer = Inferer(
        nn_archive_path=Path(nn_archive) if nn_archive else None,
        onnx_path=Path(onnx) if onnx else None,
        backend=backend,
        device_ip=device_ip,
    )

    # TODO: This code is placeholder, we need to implement a proper way to handle different loaders based on the dataset and model type. The loaders should always inherit from BaseLoader (from luxonis_ml).
    loader = LuxonisLoader(
        dataset,
        view=["val"],
        height=inferer.height,
        width=inferer.width,
        keep_aspect_ratio=True,
        color_space="RGB" if backend == "onnx" else "BGR",
    )
    logger.info(
        f"Dataset loaded with {len(loader)} samples with images of size {inferer.height}x{inferer.width}."
    )

    # TODO: Load task_cfg, metric_cfg, onnx_cfg from a config file or CLI arguments

    # Config for imagenet-based classification task
    task_cfg = {
        "name": "ClassificationTask",
    }
    parser_cfg = {
        "name": "ClassificationParser",
        # "apply_softmax": True,
    }
    metric_cfg = {
        "metrics": [
            {"name": "TopKAccuracy"},
        ]
    }
    onnx_cfg = {}

    # Config for COCO-based detection task
    task_cfg = {
        "name": "DetectionTask",
    }
    parser_cfg = {
        "name": "YOLODetectionParser",
    }
    metric_cfg = {
        "metrics": [
            {"name": "BboxMeanAveragePrecision"},
        ]
    }
    onnx_cfg = {
        "mean": 0.0,
        "std": 255.0,
    }

    # Config for COCO-based instance segmentation task
    task_cfg = {
        "name": "InstanceSegmentationTask",
    }
    parser_cfg = {
        "name": "YOLOInstanceSegmentationParser",
    }
    metric_cfg = {
        "metrics": [
            {"name": "BboxMeanAveragePrecision"},
            {"name": "MaskMeanAveragePrecision"},
        ],
    }
    onnx_cfg = {
        "mean": 0.0,
        "std": 255.0,
    }

    # TODO: task_name should be determined based on the model type. Check if there is any way to do that automatically, otherwise add it as a parameter to the CLI or config file.
    inferer.infer(
        loader,
        task_cfg=task_cfg,
        parser_cfg=parser_cfg,
        metric_cfg=metric_cfg,
        onnx_cfg=onnx_cfg,
    )


if __name__ == "__main__":
    app()
