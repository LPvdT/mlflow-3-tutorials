import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytz
from loguru import logger


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
    base_demand: int = 1000,
    n_rows: int = 5000,
) -> pd.DataFrame:
    """
    Generates a DataFrame containing synthetic apple sales data with promotional adjustments.
    """

    rng = np.random.default_rng(9999)

    # Create date range (vectorized)
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
        "weekend": pd.Series(dates).dt.weekday >= 5,  # noqa
        "holiday": rng.choice([0, 1], n_rows, p=[0.97, 0.03]),
        "price_per_kg": rng.uniform(0.5, 3.0, n_rows),
        "month": pd.Series(dates).dt.month,
    })

    # Vectorized calculations
    year_min = data_sales["date"].dt.year.min()
    data_sales["inflation_multiplier"] = (
        1 + (data_sales["date"].dt.year - year_min) * 0.03
    )

    data_sales["harvest_effect"] = np.sin(
        2 * np.pi * (data_sales["month"] - 3) / 12
    ) + np.sin(2 * np.pi * (data_sales["month"] - 9) / 12)

    # Adjust price per kg based on harvest effect
    data_sales["price_per_kg"] -= data_sales["harvest_effect"] * 0.5

    # Promo assignment (vectorized)
    peak_months = [4, 10]
    data_sales["promo"] = 0
    is_peak_month = data_sales["month"].isin(peak_months)
    data_sales.loc[is_peak_month, "promo"] = 1
    data_sales.loc[~is_peak_month, "promo"] = rng.choice(
        [0, 1], (~is_peak_month).sum(), p=[0.85, 0.15]
    )

    # Calculate demand components
    base_price_effect = -data_sales["price_per_kg"] * 50
    seasonality_effect = data_sales["harvest_effect"] * 50
    promo_effect = data_sales["promo"] * 200

    data_sales["demand"] = (
        base_demand
        + base_price_effect
        + seasonality_effect
        + promo_effect
        + data_sales["weekend"].astype(int) * 300
        + rng.normal(0, 50, n_rows)
    ) * data_sales["inflation_multiplier"]

    # Previous day's demand with fill backward for first row
    data_sales["previous_days_demand"] = data_sales["demand"].shift(1).bfill()

    # Drop temporary columns
    return data_sales.drop(
        columns=["inflation_multiplier", "harvest_effect", "month"]
    )
