from loguru import logger
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor

from mlflow_3_tutorials.lib.constants import TRACKING_URI


def main() -> None:
    client = MlflowClient(tracking_uri=TRACKING_URI)

    all_experiments = client.search_experiments()
    logger.info(f"experiments: {all_experiments}")

    default_experiment = next(
        (
            {"name": exp.name, "lifecycle_stage": exp.lifecycle_stage}
            for exp in all_experiments
            if exp.name == "Default"
        ),
        None,
    )
    logger.info(f"default_experiment: {default_experiment}")
