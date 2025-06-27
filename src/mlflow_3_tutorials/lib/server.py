import mlflow

from ..constants import TRACKING_URI


def configure_tracking_server():
    mlflow.set_tracking_uri(TRACKING_URI)
