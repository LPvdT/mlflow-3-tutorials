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
    run_command(
        f"mlflow server --host {SERVER_ADDRESS} --port {SERVER_PORT}",
        "MLflow tracking server",
    )


def uv_sync() -> None:
    run_command(
        "uv sync --managed-python --all-groups --compile-bytecode",
        "uv sync",
    )


def run_pyment() -> None:
    run_command(
        "pyment -f false -o google .",
        "pyment docstring formatter",
    )


def run_precommit() -> None:
    commands = {
        "pre-commit autoupdate": "pre-commit update",
        "pre-commit run -a": "pre-commit run all hooks",
    }

    for cmd, desc in commands.items():
        run_command(cmd, desc)
