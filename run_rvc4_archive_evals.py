from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_ARCHIVES_DIR = Path("256_512_experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find .rvc4.tar.xz archives in a directory and evaluate each one with "
            "luxonis-eval."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_ARCHIVES_DIR,
        help="Directory containing .rvc4.tar.xz archives.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Evaluation config passed to luxonis-eval eval.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively instead of only the input directory.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue evaluating later archives if one archive fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--luxonis-eval-bin",
        default=None,
        help=(
            "Optional console executable name or path. By default the script "
            "uses the current interpreter: python -m luxonis_eval."
        ),
    )
    return parser.parse_args()


def format_command(command: list[str]) -> str:
    return " ".join(quote_arg(arg) for arg in command)


def quote_arg(arg: str) -> str:
    if not arg:
        return '""'
    if any(char.isspace() for char in arg) or any(char in arg for char in "\"'()"):
        return '"' + arg.replace('"', '\\"') + '"'
    return arg


def collect_archives(directory: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.rvc4.tar.xz" if recursive else "*.rvc4.tar.xz"
    return sorted(path for path in directory.glob(pattern) if path.is_file())


def build_luxonis_eval_prefix(luxonis_eval_bin: str | None) -> list[str]:
    if luxonis_eval_bin is not None:
        return [luxonis_eval_bin]
    return [sys.executable, "-m", "luxonis_eval"]


def main() -> None:
    args = parse_args()

    if not args.directory.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.directory}")
    if not args.config.is_file():
        raise SystemExit(f"Config file does not exist: {args.config}")

    archives = collect_archives(args.directory, args.recursive)
    if not archives:
        raise SystemExit(f"No .rvc4.tar.xz archives found in: {args.directory}")

    failures: list[tuple[Path, int]] = []
    command_prefix = build_luxonis_eval_prefix(args.luxonis_eval_bin)
    for archive in archives:
        command = [
            *command_prefix,
            "eval",
            "--config",
            str(args.config),
            "--model-path",
            str(archive),
        ]
        print(format_command(command), flush=True)

        if args.dry_run:
            continue

        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            if not args.continue_on_error:
                raise SystemExit(result.returncode)
            failures.append((archive, result.returncode))

    if failures:
        print("\nFailed archives:")
        for archive, returncode in failures:
            print(f"  {archive} exited with code {returncode}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
