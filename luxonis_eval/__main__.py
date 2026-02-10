from importlib.metadata import version
from pathlib import Path
from typing import Literal

from cyclopts import App, Group
from loguru import logger
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.loaders import LuxonisLoader

from luxonis_eval.inferer import Inferer
from luxonis_eval.utils.config import EvalConfig

app = App(
    help="Luxonis Eval CLI",
    version=lambda: f"LuxonisEval v{version('luxonis_eval')}",
)
app.meta.group_parameters = Group("Global Parameters", sort_key=0)
app["--help"].group = app.meta.group_parameters
app["--version"].group = app.meta.group_parameters


@app.command()
def eval(
    dataset_name: str | None = None,
    nn_archive: str | None = None,
    onnx: str | None = None,
    backend: Literal["depthai", "onnx", "all"] | None = None,
    device_ip: str | None = None,
    config: str | None = None,
):
    """Run evaluation on a dataset using a specified neural network.

    Parameters
    ----------
    dataset_name : str | None, optional
        Name of the dataset to evaluate on.
    nn_archive : str | None, optional
        Path to the neural network NNArchive file. Required if backend is set to 'depthai' or 'all'.
    onnx : str | None, optional
        Path to the ONNX model file, required if backend is 'onnx' or 'all'. Required if backend is set to 'onnx' or 'all'.
    backend : Literal["depthai", "onnx", "all"] | None, optional
        Backend to use for inference. If 'all', runs inference on all available backends.
    device_ip : str | None, optional
        IP address of the device to connect to. Only applicable for RVC4 devices.
    config : str | None, optional
        Path to the evaluation configuration file in YAML format.
    """
    overrides = {}
    if dataset_name is not None:
        overrides["dataset_name"] = dataset_name
    if nn_archive is not None:
        overrides["nn_archive"] = nn_archive
    if onnx is not None:
        overrides["onnx"] = onnx
    if backend is not None:
        overrides["backend"] = backend
    if device_ip is not None:
        overrides["device_ip"] = device_ip

    cfg = EvalConfig.get_config(cfg=config, overrides=overrides)

    # TODO: This code is placeholder, we need to implement a proper way to handle different datasets. The datasets should always inherit from BaseDataset (from luxonis_ml).
    dataset = LuxonisDataset(cfg.dataset_name)

    inferer = Inferer(
        nn_archive_path=Path(cfg.nn_archive) if cfg.nn_archive else None,
        onnx_path=Path(cfg.onnx) if cfg.onnx else None,
        backend=cfg.backend,
        device_ip=cfg.device_ip,
    )

    # TODO: This code is placeholder, we need to implement a proper way to handle different loaders based on the dataset and model type. The loaders should always inherit from BaseLoader (from luxonis_ml).
    loader = LuxonisLoader(
        dataset,
        view=["val"],
        height=inferer.height,
        width=inferer.width,
        keep_aspect_ratio=True,
        color_space="RGB" if cfg.backend == "onnx" else "BGR",
    )
    logger.info(
        f"Dataset loaded with {len(loader)} samples with images of size {inferer.height}x{inferer.width}."
    )

    # TODO: task_name should be determined based on the model type. Check if there is any way to do that automatically, otherwise add it as a parameter to the CLI or config file.
    inferer.infer(
        loader,
        task_cfg=cfg.task_cfg.model_dump(),
        parser_cfg=cfg.parser_cfg.model_dump(),
        metric_cfg=cfg.metrics_cfg.model_dump(),
        onnx_cfg=cfg.onnx_cfg.model_dump() if cfg.onnx_cfg else None,
    )


if __name__ == "__main__":
    app()
