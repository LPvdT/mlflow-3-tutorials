import functools
import math
import sys
from pathlib import Path

import mlflow
import optuna
import psutil
import xgboost
from loguru import logger
from mlflow import exceptions
from optuna.integration.mlflow import MLflowCallback
from sklearn.model_selection import train_test_split

from mlflow_3_tutorials.lib.constants import LOG_LEVEL, TRACKING_URI
from mlflow_3_tutorials.lib.utils import (
    champion_callback,
    generate_apple_sales_data_with_promo_adjustment,
    get_or_create_experiment,
    objective,
    plot_correlation_with_demand,
    plot_feature_importance,
    plot_residuals_xgboost,
)


def main() -> None:
    # Configure logger
    logger.remove()
    logger.bind(name=__file__).add(sys.stderr, level=LOG_LEVEL)

    # Set Optuna logging level
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # Set the MLflow tracking URI
    mlflow.set_tracking_uri(TRACKING_URI)

    # Generate synthetic apple sales data with promotional adjustments
    data_apples = generate_apple_sales_data_with_promo_adjustment(
        base_demand=1_000, n_rows=5000
    )

    # Generate correlation plot with demand
    correlation_plot = plot_correlation_with_demand(
        data_apples, save_path="figures/correlation_plot.png"
    )

    # Get or create an MLflow experiment
    experiment_id = get_or_create_experiment("Apples Demand")

    # Set the current active MLflow experiment
    mlflow.set_experiment(experiment_id=experiment_id)

    # Preprocess the dataset
    X = data_apples.drop(columns=["date", "demand"])
    y = data_apples["demand"]

    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.25)

    dtrain = xgboost.DMatrix(train_x, label=train_y)
    dvalid = xgboost.DMatrix(valid_x, label=valid_y)

    # Set run name for MLflow tracking
    run_name = "first_attempt"
    model_uri = ""

    # Initiate the parent run and call the hyperparameter tuning child run logic
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=run_name, nested=True
    ):
        # Initialize the Optuna study
        study = optuna.create_study(
            direction="minimize", pruner=optuna.pruners.HyperbandPruner()
        )

        # Execute the hyperparameter optimization trials.
        # Note the addition of the `champion_callback` inclusion to control our logging
        study.optimize(
            functools.partial(
                objective,
                dtrain=dtrain,  # type: ignore
                dvalid=dvalid,  # type: ignore
                y_valid=valid_y,  # type: ignore
            ),
            n_trials=1_000,
            callbacks=[
                champion_callback,
                MLflowCallback(
                    create_experiment=False,
                    tracking_uri=TRACKING_URI,
                    metric_name="rmse",
                    mlflow_kwargs={"nested": True},
                ),
            ],
            n_jobs=psutil.cpu_count(logical=True) - 1,  # type: ignore
            show_progress_bar=True,
            gc_after_trial=True,
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_mse", study.best_value)
        mlflow.log_metric("best_rmse", math.sqrt(study.best_value))

        # Log tags
        mlflow.set_tags(
            tags={
                "project": "Apple Demand Project",
                "optimizer_engine": "optuna",
                "model_family": "xgboost",
                "feature_set_version": 1,
            }
        )

        # Log a fit model instance
        model = xgboost.train(study.best_params, dtrain)

        # Log the correlation plot
        mlflow.log_figure(
            figure=correlation_plot, artifact_file="correlation_plot.png"
        )

        # Log the feature importances plot
        importances = plot_feature_importance(
            model, booster=study.best_params.get("booster", "")
        )
        mlflow.log_figure(
            figure=importances, artifact_file="feature_importances.png"
        )

        # Log the residuals plot
        residuals = plot_residuals_xgboost(model, dvalid, valid_y)  # type: ignore
        mlflow.log_figure(figure=residuals, artifact_file="residuals.png")

        artifact_path = "model"

        mlflow.xgboost.log_model(  # type: ignore
            xgb_model=model,
            name=artifact_path,
            input_example=train_x.iloc[[0]],  # type: ignore
            model_format="ubj",
            metadata={"model_data_version": 1},
        )

        # Get the logged model uri so that we can load it from the artifact store
        model_uri = mlflow.get_artifact_uri(artifact_path)
        logger.info(f"Model logged to MLflow at: {model_uri}")

    # Load the model from the logged URI
    try:
        logger.info("Trying to load model from local directory...")
        loaded = mlflow.xgboost.load_model(  # type: ignore
            str(
                next(
                    Path(__file__)
                    .parents[2]
                    .rglob("*mlartifacts/**/models/**/artifacts")
                )
            )
        )
    except FileNotFoundError:
        logger.warning(
            "No artifacts found in local directory. Downloading model from MLflow..."
        )
        loaded = mlflow.xgboost.load_model(model_uri)  # type: ignore
    except exceptions.MlflowException:
        logger.warning(
            "Timed out even though model was downloaded successfully. Loading from local directory..."
        )
        loaded = mlflow.xgboost.load_model(  # type: ignore
            str(
                next(
                    Path(__file__)
                    .parents[2]
                    .rglob("*mlartifacts/**/models/**/artifacts")
                )
            )
        )

    # Perform inference using the loaded model
    batch_dmatrix = xgboost.DMatrix(X)
    inference = loaded.predict(batch_dmatrix)

    infer_df = data_apples.copy()
    infer_df["predicted_demand"] = inference

    # Log some inference results
    logger.info("Inference results:")
    logger.info(infer_df.head(5).to_json(indent=2, orient="records"))
