from loguru import logger
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor  # noqa

from mlflow_3_tutorials.lib.constants import TRACKING_URI
from mlflow_3_tutorials.lib.utils import (
    as_json,
    generate_apple_sales_data_with_promo_adjustment,
)


def main() -> None:
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # List all experiments
    all_experiments = client.search_experiments()
    logger.info(f"experiments: {as_json(vars(all_experiments))}")

    # Default experiment
    default_experiment = next(
        (
            {"name": exp.name, "lifecycle_stage": exp.lifecycle_stage}
            for exp in all_experiments
            if exp.name == "Default"
        ),
        None,
    )
    logger.info(f"default_experiment: {as_json(default_experiment)}")

    # apples_experiment
    experiment_description = (
        "This is the grocery forecasting project. "
        "This experiment contains the produce models for apples."
    )

    experiment_tags = {
        "project_name": "grocery-forecasting",
        "store_dept": "produce",
        "team": "stores-ml",
        "project_quarter": "Q3-2023",
        "mlflow.note.content": experiment_description,
    }

    _produce_apples_experiment = client.create_experiment(
        name="produce-apples",
        tags=experiment_tags,
    )

    apples_experiment = client.search_experiments(
        filter_string="tags.`project_name` = 'grocery-forecasting'",
    )

    logger.info(f"apples_experiment: {as_json(vars(apples_experiment[0]))}")

    data = generate_apple_sales_data_with_promo_adjustment(1_000, 5_000)
    logger.info(f"data: {as_json(data.sample(3).to_dict(orient='records'))}")
