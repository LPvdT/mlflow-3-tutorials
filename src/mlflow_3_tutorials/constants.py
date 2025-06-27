import logging
from typing import Final

# URL and port for the MLflow tracking server.
_URL: Final[str] = "http://localhost"
_PORT: Final[int] = 5000

# The tracking URI for MLflow, which is the URL and port where the MLflow server is running.
TRACKING_URI: Final[str] = f"{_URL}:{_PORT}"

# Log level
LOG_LEVEL: Final[int] = logging.INFO
