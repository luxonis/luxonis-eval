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
    nn_archive: str,
    backend: Literal["depthai", "onnx", "all"] = "depthai",
    onnx: str | None = None,
    device_ip: str | None = None,
):
    """Run evaluation on a dataset using a specified neural network.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to evaluate on.
    nn_archive : str
        Path to the neural network NNArchive file.
    backend : str, optional
        Backend to use for inference. If 'all', runs inference on all available backends.
    onnx : str | None, optional
        Path to the ONNX model file, required if backend is 'onnx' or 'all'.
    device_ip : str | None, optional
        IP address of the device to connect to. Only applicable for RVC4 devices.
    """
    if Path(nn_archive).exists() is False:
        raise ValueError(f"NNArchive file '{nn_archive}' does not exist.")

    if onnx is not None and Path(onnx).exists() is False:
        raise ValueError(f"ONNX model file '{onnx}' does not exist.")

    if backend in ["onnx", "all"] and onnx is None:
        raise ValueError(
            "ONNX model path must be provided when using ONNX backend."
        )

    if not LuxonisDataset.exists(dataset_name):
        raise ValueError(f"Dataset '{dataset_name}' does not exist.")

    # TODO: This code is placeholder, we need to implement a proper way to handle different datasets. The datasets should always inherit from BaseDataset (from luxonis_ml).
    dataset = LuxonisDataset(dataset_name)

    inferer = Inferer(
        nn_archive_path=Path(nn_archive),
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
    task_cfg = {}
    metric_cfg = {}
    onnx_cfg = {}
    # TODO: task_name should be determined based on the model type. Check if there is any way to do that automatically, otherwise add it as a parameter to the CLI or config file.
    inferer.infer(
        loader,
        task_name="DetectionTask",
        task_cfg=task_cfg,
        metric_cfg=metric_cfg,
        onnx_cfg=onnx_cfg,
    )


if __name__ == "__main__":
    app()
