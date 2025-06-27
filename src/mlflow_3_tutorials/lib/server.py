import subprocess

import mlflow
from loguru import logger

from mlflow_3_tutorials.constants import (
    SERVER_ADDRESS,
    SERVER_PORT,
    TRACKING_PORT,
    TRACKING_URI,
    TRACKING_URL,
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
