import pytest


def test_dummy(mlflow_tracking_uri: pytest.FixtureDef[str]) -> None:
    assert mlflow_tracking_uri == mlflow_tracking_uri  # noqa
