from importlib.metadata import version
from typing import Any, Literal

from cyclopts import App, Group
from luxonis_ml.typing import Params, PathType

from luxonis_eval.config import EvalConfig
from luxonis_eval.core import LuxonisEval
from luxonis_eval.core.results import EvaluationResult
from luxonis_eval.utils.json_utils import write_output_json

app = App(
    help="Luxonis Eval CLI",
    version=lambda: f"LuxonisEval v{version('luxonis_eval')}",
)
app.meta.group_parameters = Group("Global Parameters", sort_key=0)
app["--help"].group = app.meta.group_parameters
app["--version"].group = app.meta.group_parameters


def eval_run(
    cfg: PathType | Params | EvalConfig,
    opts: Params | list[str] | tuple[str, ...] | None = None,
    output_json: str | None = None,
) -> EvaluationResult:
    """Run evaluation with the given configuration."""
    # Temporary: until benchmark execution is implemented, `eval` delegates
    # to the quality-only path.
    return quality_run(cfg, opts, output_json=output_json)


def quality_run(
    cfg: PathType | Params | EvalConfig,
    opts: Params | list[str] | tuple[str, ...] | None = None,
    output_json: str | None = None,
) -> EvaluationResult:
    """Run the configured quality evaluators."""
    evaluator = LuxonisEval(cfg, opts)
    evaluator.setup()
    try:
        result = evaluator.evaluate()
        if output_json is not None:
            write_output_json(output_json, result)
        return result
    finally:
        evaluator.close()


def _build_overrides(
    *,
    dataset_name: str | None = None,
    model_path: str | None = None,
    backend: Literal["depthai", "onnx"] | None = None,
    device_ip: str | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if dataset_name is not None:
        overrides["pipeline.loader.params.dataset_name"] = dataset_name
    if model_path is not None:
        overrides["pipeline.engine.model_path"] = model_path
    if backend is not None:
        overrides["pipeline.engine.name"] = backend
    if device_ip is not None:
        overrides["pipeline.engine.params.device_ip"] = device_ip
    return overrides


@app.command()
def eval(
    config: str,
    dataset_name: str | None = None,
    model_path: str | None = None,
    backend: Literal["depthai", "onnx"] | None = None,
    device_ip: str | None = None,
    output_json: str | None = None,
) -> None:
    """Run evaluation on a dataset using a specified neural network.

    Parameters
    ----------
    config : str
        Path to the evaluation configuration file in YAML format.
    dataset_name : str | None, optional
        Name of the dataset to evaluate on.
    model_path : str | None, optional
        Path to the model file (NNArchive or ONNX).
    backend : Literal["depthai", "onnx"] | None, optional
        Backend to use for inference.
    device_ip : str | None, optional
        IP address of the device to connect to. Only applicable for RVC4 devices.
    output_json : str | None, optional
        Path to write a JSON summary containing engine, model name, and metrics.
    """
    eval_run(
        config,
        _build_overrides(
            dataset_name=dataset_name,
            model_path=model_path,
            backend=backend,
            device_ip=device_ip,
        ),
        output_json=output_json,
    )


@app.command()
def quality(
    config: str,
    dataset_name: str | None = None,
    model_path: str | None = None,
    backend: Literal["depthai", "onnx"] | None = None,
    device_ip: str | None = None,
    output_json: str | None = None,
) -> None:
    """Run only the configured quality evaluators."""
    quality_run(
        config,
        _build_overrides(
            dataset_name=dataset_name,
            model_path=model_path,
            backend=backend,
            device_ip=device_ip,
        ),
        output_json=output_json,
    )


if __name__ == "__main__":
    app()
