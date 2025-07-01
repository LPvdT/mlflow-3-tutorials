import mlflow
from loguru import logger
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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

    # Train and log first model
    data = generate_apple_sales_data_with_promo_adjustment(1_000, 5_000)
    logger.info(f"data: {as_json(data.sample(3).to_dict(orient='records'))}")

    _apple_experiment = mlflow.set_experiment("apple-models")
    run_name = "apples_rf_test"
    artifact_path = "rf_apples"

    X = data.drop(columns=["date", "demand"])
    y = data["demand"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "bootstrap": True,
        "oob_score": False,
        "random_state": 888,
        "n_jobs": -1,
    }

    logger.info(f"params: {as_json(params)}")

    rf = RandomForestRegressor(**params)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }

    with mlflow.start_run(run_name=run_name) as _run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(  # type: ignore
            sk_model=rf,
            input_example=X_val,
            name=artifact_path,
        )


if __name__ == "__main__":
    main()
