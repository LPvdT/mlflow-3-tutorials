import logging
from typing import Final

# URL for the wine quality dataset
WINE_QUALITY_DATA_URL: Final[str] = (
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv"
)

# URL and port for MLflow
TRACKING_URL: Final[str]
SERVER_ADDRESS: Final[str]
TRACKING_URL = SERVER_ADDRESS = "127.0.0.1"

SERVER_PORT: Final[int]
TRACKING_PORT: Final[int]
SERVER_PORT = TRACKING_PORT = 5000

# MLflow backend store and default artifact root
BACKEND_STORE: Final[str] = "sqlite:///mlflow.db"
DEFAULT_ARTIFACT_ROOT: Final[str] = "./mlruns"

# MLflow tracking URI
TRACKING_URI: Final[str] = f"{TRACKING_URL}:{TRACKING_PORT}"

# Log level
LOG_LEVEL: Final[int] = logging.INFO
