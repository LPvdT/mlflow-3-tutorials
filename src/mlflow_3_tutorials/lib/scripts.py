from pathlib import Path

import mlflow
from loguru import logger

from mlflow_3_tutorials.lib.constants import (
    SERVER_ADDRESS,
    SERVER_PORT,
    TRACKING_URI,
)
from mlflow_3_tutorials.lib.runner import run_command


def configure_tracking_server() -> None:
    """
    Configure MLflow to use the MLflow tracking server running on
    http://{SERVER_ADDRESS}:{SERVER_PORT}.

    This function sets the MLflow tracking URI to
    http://{SERVER_ADDRESS}:{SERVER_PORT} and logs the result.
    """

    logger.info(f"Setting MLflow tracking URI to: {TRACKING_URI}")
    mlflow.set_tracking_uri(TRACKING_URI)


def start_tracking_server() -> None:
    """
    Start an MLflow tracking server on http://{SERVER_ADDRESS}:{SERVER_PORT}.

    This function runs the command `mlflow server --host {SERVER_ADDRESS} --port {SERVER_PORT}` and logs the result.
    """

    run_command(
        f"mlflow server --host {SERVER_ADDRESS} --port {SERVER_PORT}",
        f"MLflow tracking server: http://{SERVER_ADDRESS}:{SERVER_PORT}",
    )


def uv_sync() -> None:
    """
    Run the command `uv sync --managed-python --all-groups --compile-bytecode`
    to install the project's dependencies and compile bytecode.

    This function logs the result of the command.
    """

    run_command(
        "uv sync --managed-python --all-groups --compile-bytecode",
        "uv sync",
    )


def run_pyment(style: str = "google") -> None:
    """
    Run the `pyment` docstring formatter with the specified style.

    This function logs the result of the command.

    Args:
        style (str): The docstring style to format with, defaults to 'google'.
    """

    run_command(
        f"pyment -f false -o {style} .",
        "pyment docstring formatter",
    )


def run_precommit() -> None:
    """
    Run the `pre-commit` hooks for the project.

    This function runs the following commands in order to update the `pre-commit`
    configuration and run all `pre-commit` hooks:

    - `pre-commit autoupdate`: Update the pre-commit configuration.
    - `pre-commit run -a`: Run all pre-commit hooks.
    """

    commands = {
        "pre-commit autoupdate": "Update the pre-commit configuration",
        "pre-commit run -a": "Run all pre-commit hooks",
    }

    for cmd, desc in commands.items():
        run_command(cmd, desc, check=False, show_output=True)


def remove_all_experiments() -> None:
    experiment_ids = [
        p
        for p in Path().cwd().rglob(r"mlruns/*")
        if (p.name.isdigit() and p.name != "0") or p.name == ".trash"
    ]

    if not experiment_ids:
        logger.info("No experiments to delete")
        return

    for exp in experiment_ids:
        if exp.name == ".trash":
            run_command(f"rm -rf {exp}", "Remove .trash directory")
        else:
            run_command(
                f"mlflow experiments delete -x {exp.name}",
                f"Remove experiment: {exp.name}",
            )


def serve_wine_model(
    model_name: str = "wine-quality-predictor",
    version: int = 1,
    port: int = 5002,
) -> None:
    run_command(
        f'mlflow models serve -m "models:/{model_name}/{version}" --port {port} --env-manager local',
        f"Serving: '{model_name}' - version {version}",
    )
