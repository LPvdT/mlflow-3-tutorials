import pytest

from mlflow_3_tutorials.lib.constants import TRACKING_URI


@pytest.fixture(scope="session")
def mlflow_tracking_uri() -> str:
    return TRACKING_URI
