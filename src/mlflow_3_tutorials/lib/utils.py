import json
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
import xgboost
from loguru import logger
from matplotlib.figure import Figure


def as_json(obj: object, *, indent: int = 2) -> str:
    """
    Serializes a given object to a JSON formatted string.

    Handles pandas Timestamps and falls back to string conversion for non-serializable objects.
    """

    def _fallback_serializer(x: Any) -> Any:
        if isinstance(x, pd.Timestamp):
            return x.isoformat()

        try:
            return vars(x)  # Try converting custom objects to dict
        except TypeError:
            return str(x)  # Final fallback

    try:
        payload = json.dumps(obj, indent=indent, default=_fallback_serializer)
    except (TypeError, OverflowError) as e:
        msg = f"Object {obj} cannot be serialized to JSON: {e!s}"
        logger.error(msg)
        raise ValueError(msg) from e
    else:
        return payload


def generate_apple_sales_data_with_promo_adjustment(
    base_demand: int = 1_000,
    n_rows: int = 5_000,
    competitor_price_effect: float = -50.0,
) -> pd.DataFrame:
    """
    Generates a synthetic dataset for predicting apple sales demand with multiple influencing factors.

    This function creates a pandas DataFrame with features relevant to apple sales.
    The features include date, average_temperature, rainfall, weekend flag, holiday flag, promotional flag,
    price_per_kg, competitor's price, marketing intensity, stock availability, and the previous day's demand.
    The target variable, 'demand', is generated based on a combination of these features with some added noise.

    Args:
        base_demand (int, optional): Base demand for apples. Defaults to 1000.
        n_rows (int, optional): Number of rows (days) of data to generate. Defaults to 5000.
        competitor_price_effect (float, optional): Effect of competitor's price being lower on our sales.
                                                    Defaults to -50.

    Returns:
        pd.DataFrame: DataFrame with features and target variable for apple sales prediction.

    Example:
    >>> df = generate_apple_sales_data_with_promo_adjustment(base_demand=1200, n_rows=6000)
    >>> df.head()
    """

    rng = np.random.default_rng(9999)

    # Constants
    WEEKENED_DAY_START = 5  # Saturday
    HARVEST_PRICE_IMPACT = 0.5
    PRICE_SENSITIVITY = -50
    HARVEST_EFFECT_MULTIPLIER = 50
    PROMO_BOOST = 200
    WEEKEND_BOOST = 300
    STOCK_THRESHOLD = 0.95
    MARKETING_DURATION = 7

    # Date range
    dates = pd.date_range(
        end=datetime.now(tz=pytz.timezone("Europe/Amsterdam")),
        periods=n_rows,
        freq="D",
    )

    # Generate features
    data_sales = pd.DataFrame({
        "date": dates,
        "average_temperature": rng.uniform(5, 30, n_rows),
        "rainfall": rng.exponential(5, n_rows),
        "weekend": pd.Series(dates).dt.weekday >= WEEKENED_DAY_START,
        "holiday": rng.choice([0, 1], n_rows, p=[0.97, 0.03]),
        "price_per_kg": rng.uniform(0.5, 3.0, n_rows),
        "month": pd.Series(dates).dt.month,
    })

    # Inflation and seasonality
    year_min = data_sales["date"].dt.year.min()
    data_sales["inflation_multiplier"] = (
        1 + (data_sales["date"].dt.year - year_min) * 0.03
    )

    data_sales["harvest_effect"] = np.sin(
        2 * np.pi * (data_sales["month"] - 3) / 12
    ) + np.sin(2 * np.pi * (data_sales["month"] - 9) / 12)

    # Adjust price using harvest effect
    data_sales["price_per_kg"] -= (
        data_sales["harvest_effect"] * HARVEST_PRICE_IMPACT
    )
    data_sales["price_per_kg"] = data_sales["price_per_kg"].clip(lower=0.1)

    # Promotions
    peak_months = [4, 10]
    is_peak_month = data_sales["month"].isin(peak_months)
    data_sales["promo"] = 0
    data_sales.loc[is_peak_month, "promo"] = 1
    data_sales.loc[~is_peak_month, "promo"] = rng.choice(
        [0, 1], size=(~is_peak_month).sum(), p=[0.85, 0.15]
    )

    # Calculate demand components
    price_penalty = data_sales["price_per_kg"] * PRICE_SENSITIVITY
    seasonality = data_sales["harvest_effect"] * HARVEST_EFFECT_MULTIPLIER
    promo_bonus = data_sales["promo"] * PROMO_BOOST
    weekend_bonus = data_sales["weekend"].astype(int) * WEEKEND_BOOST

    data_sales["demand"] = (
        base_demand
        + price_penalty
        + seasonality
        + promo_bonus
        + weekend_bonus
        + rng.normal(0, 50, n_rows)
    ) * data_sales["inflation_multiplier"]

    # Previous day's demand (fill with next day's value for first entry)
    data_sales["previous_days_demand"] = data_sales["demand"].shift(1)
    data_sales["previous_days_demand"] = data_sales[
        "previous_days_demand"
    ].fillna(method="bfill")

    # Competitor pricing
    data_sales["competitor_price_per_kg"] = rng.uniform(0.5, 3.0, n_rows)
    data_sales["competitor_price_effect"] = (
        data_sales["competitor_price_per_kg"] < data_sales["price_per_kg"]
    ) * competitor_price_effect

    # Stock availability (lagged)
    price_lag_3 = data_sales["price_per_kg"].shift(3).fillna(method="bfill")
    stock_available = -np.log(price_lag_3 + 1) + 2
    data_sales["stock_available"] = np.clip(stock_available, 0.7, 1)

    # Marketing intensity
    data_sales["marketing_intensity"] = np.nan

    high_stock_indices = data_sales[
        data_sales["stock_available"] > STOCK_THRESHOLD
    ].index

    for idx in high_stock_indices:
        end_idx = min(idx + MARKETING_DURATION - 1, n_rows - 1)
        data_sales.loc[idx:end_idx, "marketing_intensity"] = rng.uniform(0.7, 1)

    data_sales["marketing_intensity"] = data_sales[
        "marketing_intensity"
    ].fillna(pd.Series(rng.uniform(0, 0.5, n_rows), index=data_sales.index))

    # Final demand adjustment
    data_sales["demand"] += (
        data_sales["competitor_price_effect"]
        + data_sales["marketing_intensity"]
    )

    # Cleanup
    return data_sales.drop(
        columns=[
            "inflation_multiplier",
            "harvest_effect",
            "month",
            "competitor_price_effect",
            "stock_available",
        ]
    )


def plot_correlation_with_demand(
    df: pd.DataFrame, save_path: str | None = None
) -> Figure:
    """
    Plots the correlation of each variable in the dataframe with the 'demand' column.

    Args:
        df (pd.DataFrame): DataFrame containing the data, including a 'demand' column.
        save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
        None (Displays the plot on a Jupyter window)
    """

    # Compute correlations between all variables and 'demand'
    correlations = df.corr()["demand"].drop("demand").sort_values()  # type: ignore

    # Generate a color palette from red to green
    colors = sns.diverging_palette(10, 130, as_cmap=True)
    color_mapped = correlations.map(colors)

    # Set Seaborn style
    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )  # Light grey background and thicker grid lines

    # Create bar plot
    fig = plt.figure(figsize=(12, 8))
    plt.barh(correlations.index, correlations.values, color=color_mapped)  # type: ignore

    # Set labels and title with increased font size
    plt.title("Correlation with Demand", fontsize=18)
    plt.xlabel("Correlation Coefficient", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")

    plt.tight_layout()

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    plt.close(fig)

    return fig


def plot_residuals(
    model: xgboost.core.Booster,
    dvalid: xgboost.core.DMatrix,
    valid_y: pd.Series,
    save_path: str | None = None,
) -> Figure:
    """
    Plots the residuals of the model predictions against the true values.

    Args:
        model: The trained XGBoost model.
        dvalid (xgb.DMatrix): The validation data in XGBoost DMatrix format.
        valid_y (pd.Series): The true values for the validation set.
        save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
        None (Displays the residuals plot on a Jupyter window)
    """

    # Predict using the model
    preds = model.predict(dvalid)

    # Calculate residuals
    residuals = valid_y - preds

    # Set Seaborn style
    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )

    # Create scatter plot
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(valid_y, residuals, color="blue", alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="-")

    # Set labels, title and other plot properties
    plt.title("Residuals vs True Values", fontsize=18)
    plt.xlabel("True Values", fontsize=16)
    plt.ylabel("Residuals", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y")

    plt.tight_layout()

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    plt.close(fig)

    return fig


def plot_feature_importance(
    model: xgboost.core.Booster, booster: str
) -> Figure:
    """
    Plots feature importance for an XGBoost model.

    Args:
        model: A trained XGBoost model

    Returns:
        fig: The matplotlib figure object
    """

    fig, ax = plt.subplots(figsize=(10, 8))
    importance_type = "weight" if booster == "gblinear" else "gain"
    xgboost.plot_importance(
        model,
        importance_type=importance_type,
        ax=ax,
        title=f"Feature Importance based on {importance_type}",
    )
    plt.tight_layout()
    plt.close(fig)

    return fig
