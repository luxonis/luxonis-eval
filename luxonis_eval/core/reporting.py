import re
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from tabulate import tabulate
from tqdm.auto import tqdm


class TQDMProgressAdapter(AbstractContextManager["TQDMProgressAdapter"]):
    def __init__(self, description: str, total: int) -> None:
        self._description = description
        self._total = total
        self._progress: tqdm[Any] | None = None

    def __enter__(self) -> "TQDMProgressAdapter":
        self._progress = tqdm(
            total=self._total,
            desc=self._description,
            leave=True,
        )
        return self

    def __exit__(self, *args: object) -> None:
        assert self._progress is not None
        self._progress.close()

    def update(self, *, advance: int = 1) -> None:
        assert self._progress is not None
        self._progress.update(advance)


class RichProgressAdapter(AbstractContextManager["RichProgressAdapter"]):
    def __init__(self, description: str, total: int) -> None:
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        self._description = description
        self._total = total
        self._task_id: TaskID | None = None

    def __enter__(self) -> "RichProgressAdapter":
        self._progress.__enter__()
        self._task_id = self._progress.add_task(
            self._description,
            total=self._total,
        )
        return self

    def __exit__(self, *args: object) -> None:
        self._progress.__exit__(*args)

    def update(self, *, advance: int = 1) -> None:
        assert self._task_id is not None
        self._progress.update(self._task_id, advance=advance)


def get_model_name(path: str) -> str:
    name = Path(path).name
    return re.sub(r"\.((rvc\d+)?\.?tar\.xz|onnx)$", "", name)


def section(
    title: str, width: int = 35, line_char: str = "="
) -> list[list[str]]:
    label = f" {title} "
    centered = label.center(width, line_char)
    return [[centered, ""]]


def make_report_table(
    engine_name: str,
    model_name: str,
    tp: dict[str, float | int],
    results: list[tuple[str, dict[str, Any]]],
) -> str:
    def format_stage(name: str) -> str:
        ms = float(tp[f"{name}_ms_per_sample"])
        total = float(tp["ms_per_sample"])
        pct = (ms / total * 100.0) if total else 0.0
        return f"{ms:5.2f} ms | {pct:4.1f}%"

    rows: list[list[str]] = []

    rows += section("SETTINGS")
    rows += [
        ["Model", model_name],
        ["Engine", str(engine_name).upper()],
    ]

    rows += section("PERFORMANCE")
    rows += [
        ["Throughput", f"{tp['samples_per_s']:.2f} samples/s"],
        ["End-to-end Latency", f"{tp['ms_per_sample']:.2f} ms/sample"],
    ]

    rows += section("STAGE BREAKDOWN", line_char="-")
    rows += [
        ["Inference", format_stage("inference")],
        ["Parsing", format_stage("parsing")],
        ["Metric Update", format_stage("metric_update")],
        ["Metric Compute", format_stage("metric_compute")],
        ["Pipeline Overhead", format_stage("overhead")],
    ]

    rows += section("QUALITY")
    for metric_name, result in results:
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
