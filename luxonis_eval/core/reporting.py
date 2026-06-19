import re
from pathlib import Path
from typing import Any

from tabulate import tabulate


def get_model_name(path: str) -> str:
    name = Path(path).name
    return re.sub(r"\.((rvc\d+)?\.?tar\.xz|onnx)$", "", name)


def section(
    title: str, width: int = 35, line_char: str = "═"
) -> list[list[str]]:
    label = f" {title} "
    centered = label.center(width, line_char)
    return [[centered, ""]]


def make_report_table(
    *,
    backend: str,
    model_name: str,
    device: str,
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
        ["Backend", str(backend).upper()],
        ["Device", str(device)],
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
