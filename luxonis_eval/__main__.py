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
    backend: Literal["depthai", "onnx"] | None = None,
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
    backend : Literal["depthai", "onnx"] | None, optional
        Backend to use for inference.
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
    dataset = LuxonisDataset(cfg.dataset_cfg.name)

    inferer = Inferer(
        model_path=Path(cfg.engine_cfg.model_path),
        backend=cfg.engine_cfg.name,
        device_ip=cfg.engine_cfg.params.get("device_ip"),  # type: ignore
    )

    augmentation_config = []
    if cfg.dataset_cfg.preprocessing.normalize.active:
        augmentation_config.append(
            {
                "name": "Normalize",
                "params": cfg.dataset_cfg.preprocessing.normalize.params,
            }
        )

    # TODO: This code is placeholder, we need to implement a proper way to handle different loaders based on the dataset and model type. The loaders should always inherit from BaseLoader (from luxonis_ml).
    loader = LuxonisLoader(
        dataset,
        view=cfg.dataset_cfg.params.get("view", ["val"]),  # type: ignore
        augmentation_config=augmentation_config,
        height=inferer.height,
        width=inferer.width,
        keep_aspect_ratio=cfg.dataset_cfg.preprocessing.keep_aspect_ratio,
        color_space=cfg.dataset_cfg.preprocessing.color_space,
    )
    logger.info(
        f"Dataset loaded with {len(loader)} samples with images of size {inferer.height}x{inferer.width}."
    )

    if (
        cfg.engine_cfg.name == "depthai"
        and cfg.dataset_cfg.preprocessing.normalize.active
    ):
        logger.warning(
            "Normalization is usually part of the model's preprocessing pipeline in DepthAI. Consider disabling normalization in the dataset config."
        )
    if (
        cfg.engine_cfg.name == "depthai"
        and cfg.dataset_cfg.preprocessing.color_space == "RGB"
    ):
        logger.warning(
            "Color space is set to RGB in the dataset config. DepthAI expects BGR color space."
        )
    inferer.infer(
        loader,
        dataset_cfg=cfg.dataset_cfg.model_dump(),
        task_cfg=cfg.task_cfg.model_dump(),
        parser_cfg=cfg.parser_cfg.model_dump(),
        metrics_cfg=cfg.metrics_cfg.model_dump(),
        visualizer_cfg=cfg.visualizer_cfg.model_dump(),
        engine_cfg=cfg.engine_cfg.model_dump(),
    )


if __name__ == "__main__":
    app()
