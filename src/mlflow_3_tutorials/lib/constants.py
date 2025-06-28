import logging
from typing import Final

# URL and port for the MLflow tracking server.
TRACKING_URL: Final[str] = "http://localhost"
TRACKING_PORT: Final[int] = 5000

# The tracking URI for MLflow, which is the URL and port where the MLflow server is running.
TRACKING_URI: Final[str] = f"{TRACKING_URL}:{TRACKING_PORT}"

# Log level
LOG_LEVEL: Final[int] = logging.INFO

# Mlflow server configuration
SERVER_ADDRESS: Final[str] = "127.0.0.1"
SERVER_PORT: Final[int] = 8080
