import sys
from pathlib import Path
from typing import Literal

from loguru import logger

from mlflow_3_tutorials.lib.constants import (
    DEFAULT_ARTIFACT_ROOT,
    LOG_LEVEL,
    SERVER_ADDRESS,
    SERVER_PORT,
)
from mlflow_3_tutorials.lib.runner import run_command

logger.bind(name="runner").add(sys.stderr, level=LOG_LEVEL)


def start_tracking_server() -> None:
    """Start an MLflow tracking server."""

    url = f"http://{SERVER_ADDRESS}:{SERVER_PORT}"
    # run_command(
    #     "mlflow server "
    #     f"--host={SERVER_ADDRESS} "
    #     f"--port={SERVER_PORT} "
    #     f"--default-artifact-root={DEFAULT_ARTIFACT_ROOT}",
    #     f"MLflow tracking server: {url} - default-artifact-root={DEFAULT_ARTIFACT_ROOT}",
    #     None,
    # )
    run_command(
        f"mlflow server --host={SERVER_ADDRESS} --port={SERVER_PORT}",
        f"MLflow tracking server: {url}",
        None,
    )


def uv_sync(
    extras: list[Literal["tensorflow", "pytorch", "mlflow_extras"]]
    | None = None,
) -> None:
    """Install project dependencies using uv."""

    extras_str = (
        " ".join(f"--extra {extra}" for extra in extras) if extras else ""
    )
    description = f"uv sync [{' '.join(extras)}]" if extras else "uv sync"
    run_command(
        f"uv sync --managed-python --all-groups {extras_str}",
        description,
    )


def run_pyment(style: str = "google") -> None:
    """Format docstrings using pyment."""

    run_command(
        f"pyment -f false -o {style} .",
        f"Format docstrings with pyment ({style})",
    )


def run_precommit() -> None:
    """Run pre-commit hooks."""

    steps = [
        ("pre-commit autoupdate", "Update the pre-commit configuration"),
        ("pre-commit run -a", "Run all pre-commit hooks"),
    ]
    for cmd, desc in steps:
        run_command(cmd, desc, check=False, show_output=True)


def remove_all_experiments() -> None:
    """
    Remove all MLflow experiments.

    If the '--all' or '-a' option is specified as a command-line argument,
    deletes the entire 'mlruns/' directory. Otherwise, it identifies and deletes
    individual experiment directories (excluding '0') and the '.trash' directory
    if they exist within the 'mlruns/' path.

    Logs a message if no experiments are found to delete.
    """

    if len(sys.argv) > 1 and (sys.argv[1] == "--all" or sys.argv[1] == "-a"):
        run_command(
            "rm -rf mlruns/",
            f"Remove MLflow tracking directory: {DEFAULT_ARTIFACT_ROOT}",
        )
        return

    to_delete = [
        p
        for p in Path().cwd().rglob("mlruns/*")
        if (p.name.isdigit() and p.name != "0") or p.name == ".trash"
    ]

    if not to_delete:
        logger.info("No experiments to delete")
        return

    for path in to_delete:
        if path.name == ".trash":
            run_command(f"rm -rf {path}", "Remove .trash directory")
        else:
            run_command(
                f"mlflow experiments delete -x {path.name}",
                f"Remove experiment: {path.name}",
            )
