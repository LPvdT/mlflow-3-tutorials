import subprocess

import mlflow
from loguru import logger

from mlflow_3_tutorials.constants import TRACKING_URI


def configure_tracking_server() -> None:
    logger.info(f"Setting MLflow tracking URI to: {TRACKING_URI}")
    mlflow.set_tracking_uri(TRACKING_URI)


def start_tracking_server() -> None:
    logger.info("Starting MLflow tracking server...")
    host = TRACKING_URI.split(":")[0]
    port = TRACKING_URI.split(":")[1]
    subprocess.run(
        ["mlflow", "server", "--host", host, "--port", port],
        check=True,
    )
