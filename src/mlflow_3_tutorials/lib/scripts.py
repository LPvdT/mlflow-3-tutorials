import subprocess

import mlflow
from loguru import logger

from .constants import (
    SERVER_ADDRESS,
    SERVER_PORT,
    TRACKING_URI,
)


def configure_tracking_server() -> None:
    logger.info(f"Setting MLflow tracking URI to: {TRACKING_URI}")
    mlflow.set_tracking_uri(TRACKING_URI)


def start_tracking_server() -> None:
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
