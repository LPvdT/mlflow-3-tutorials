import subprocess

import mlflow
from loguru import logger

from .constants import (
    SERVER_ADDRESS,
    SERVER_PORT,
    TRACKING_URI,
)


def configure_tracking_server() -> None:
    """
    Configure MLflow to use the MLflow tracking server running on
    http://{SERVER_ADDRESS}:{SERVER_PORT}.

    This function sets the MLflow tracking URI to
    http://{SERVER_ADDRESS}:{SERVER_PORT} and logs the result.

    :return: None
    """

    logger.info(f"Setting MLflow tracking URI to: {TRACKING_URI}")
    mlflow.set_tracking_uri(TRACKING_URI)


def start_tracking_server() -> None:
    """
    Start an MLflow tracking server on http://{SERVER_ADDRESS}:{SERVER_PORT}.

    This function will block until the MLflow tracking server is stopped.

    :return: None
    """

    logger.info(
        f"Starting MLflow tracking server on http://{SERVER_ADDRESS}:{SERVER_PORT}...",
    )
    try:
        _ = subprocess.run(
            [
                "mlflow",
                "server",
                f"--host={SERVER_ADDRESS}",
                f"--port={SERVER_PORT}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        logger.error(f"Failed to start MLflow tracking server: {e!s}")
        raise
    except KeyboardInterrupt:
        logger.warning("MLflow tracking server shutdown.")
        return


def uv_sync() -> None:
    """
    Execute the 'uv sync' command to synchronize all groups and compile bytecode.

    This function runs the 'uv sync --managed-python --all-groups --compile-bytecode'
    command to synchronize the environment and compile bytecode for all groups.
    It captures the command's output and logs any exceptions or interruptions
    that occur during its execution.

    :return: None
    """

    logger.info(
        "Running 'uv sync --managed-python --all-groups --compile-bytecode'...",
    )
    try:
        _ = subprocess.run(
            [
                "uv",
                "sync",
                "--managed-python",
                "--all-groups",
                "--compile-bytecode",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        logger.error(f"Failed to run 'uv sync': {e!s}")
        raise
    except KeyboardInterrupt:
        logger.warning("'uv sync' interrupted.")
        return
