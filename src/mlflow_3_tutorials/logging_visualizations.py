import math
import sys
from typing import TYPE_CHECKING

import mlflow
from loguru import logger
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from mlflow_3_tutorials.lib.constants import LOG_LEVEL, TRACKING_URI
from mlflow_3_tutorials.lib.utils import (
    generate_apple_sales_data_with_promo_adjustment,
    plot_box_weekend,
    plot_coefficients,
    plot_correlation_matrix,
    plot_density_weekday_weekend,
    plot_prediction_error,
    plot_qq,
    plot_residuals,
    plot_scatter_demand_price,
    plot_time_series_demand,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def main() -> None:
    # Configure logger
    logger.remove()
    logger.bind(name=__file__).add(sys.stderr, level=LOG_LEVEL)

    logger.info("Starting the logging and visualizations demo...")

    sales_data = generate_apple_sales_data_with_promo_adjustment(
        1_000, 25_000, -25.0
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("Visualizations Demo")

    X = sales_data.drop(columns=["demand", "date"])
    y = sales_data["demand"]

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray

    X_train, X_test, y_train, y_test = train_test_split(  # type: ignore
        X, y, test_size=0.2, random_state=0
    )

    fig1 = plot_time_series_demand(sales_data, window_size=28)
    fig2 = plot_box_weekend(sales_data)
    fig3 = plot_scatter_demand_price(sales_data)
    fig4 = plot_density_weekday_weekend(sales_data)

    # Execute the correlation plot, saving the plot
    plot_correlation_matrix(sales_data)

    # Define our Ridge model
    model = Ridge(alpha=1.0)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate error metrics
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    msle = metrics.mean_squared_log_error(y_test, y_pred)
    medae = metrics.median_absolute_error(y_test, y_pred)

    # Generate prediction-dependent plots
    fig5 = plot_residuals(y_test, y_pred)
    fig6 = plot_coefficients(model, X_test.columns.tolist())
    fig7 = plot_prediction_error(y_test, y_pred)
    fig8 = plot_qq(y_test, y_pred)

    # Start an MLflow run for logging metrics, parameters, the model, and our figures
    with mlflow.start_run() as _run:
        # Log the model
        mlflow.sklearn.log_model(  # type: ignore
            sk_model=model, input_example=X_test, name="model"
        )

        # Log the metrics
        mlflow.log_metrics({
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "msle": msle,
            "medae": medae,
        })

        # Log the hyperparameter
        mlflow.log_param("alpha", 1.0)

        # Log plots
        mlflow.log_figure(fig1, "time_series_demand.png")
        mlflow.log_figure(fig2, "box_weekend.png")
        mlflow.log_figure(fig3, "scatter_demand_price.png")
        mlflow.log_figure(fig4, "density_weekday_weekend.png")
        mlflow.log_figure(fig5, "residuals_plot.png")
        mlflow.log_figure(fig6, "coefficients_plot.png")
        mlflow.log_figure(fig7, "prediction_errors.png")
        mlflow.log_figure(fig8, "qq_plot.png")

        # Log the saved correlation matrix plot by referring to the local file system location
        mlflow.log_artifact("figures/corr_plot.png")

        logger.success("Finished logging and visualizations demo.")
