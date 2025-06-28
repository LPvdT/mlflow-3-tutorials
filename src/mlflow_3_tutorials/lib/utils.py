import json
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytz


def as_json(obj: Any) -> str:  # noqa
    try:
        payload = json.dumps(obj, indent=2)
    except json.JSONDecodeError as e:
        raise ValueError(  # noqa
            f"Object {obj} cannot be serialized to JSON: {e!s}"  # noqa
        ) from e
    else:
        return payload


def generate_apple_sales_data_with_promo_adjustment(
    base_demand: int = 1000,
    n_rows: int = 5000,
) -> pd.DataFrame:
    rng = np.random.default_rng(9999)

    # Create date range
    dates = [
        datetime.now(tz=pytz.timezone("Europe/Amsterdam")) - timedelta(days=i)
        for i in range(n_rows)
    ]
    dates.reverse()

    # Generate features
    random_apple_sales = pd.DataFrame({
        "date": dates,
        "average_temperature": rng.uniform(5, 30, n_rows),
        "rainfall": rng.exponential(5, n_rows),
        "weekend": [(date.weekday() >= 5) * 1 for date in dates],  # noqa
        "holiday": rng.choice([0, 1], n_rows, p=[0.97, 0.03]),
        "price_per_kg": rng.uniform(0.5, 3.0, n_rows),
        "month": [date.month for date in dates],
    })

    random_apple_sales = random_apple_sales.assign(
        inflation_multiplier=1
        + (
            random_apple_sales["date"].dt.year
            - random_apple_sales["date"].dt.year.min()
        )
        * 0.03,
    )

    random_apple_sales = random_apple_sales.assign(
        harvest_effect=np.sin(
            2 * np.pi * (random_apple_sales["month"] - 3) / 12,
        )
        + np.sin(2 * np.pi * (random_apple_sales["month"] - 9) / 12),
    )

    random_apple_sales = random_apple_sales.assign(
        price_per_kg=random_apple_sales["price_per_kg"]
        - random_apple_sales["harvest_effect"] * 0.5,
    )

    peak_months = [4, 10]
    random_apple_sales = random_apple_sales.assign(
        promo=np.where(
            random_apple_sales["month"].isin(peak_months),
            1,
            rng.choice([0, 1], n_rows, p=[0.85, 0.15]),
        ),
    )

    base_price_effect = -random_apple_sales["price_per_kg"] * 50
    seasonality_effect = random_apple_sales["harvest_effect"] * 50
    promo_effect = random_apple_sales["promo"] * 200

    random_apple_sales = random_apple_sales.assign(
        demand=(
            base_demand
            + base_price_effect
            + seasonality_effect
            + promo_effect
            + random_apple_sales["weekend"] * 300
            + rng.normal(0, 50, n_rows)
        )
        * random_apple_sales["inflation_multiplier"],
    )

    random_apple_sales = random_apple_sales.assign(
        previous_days_demand=random_apple_sales["demand"].shift(1),
    )
    random_apple_sales["previous_days_demand"] = random_apple_sales[
        "previous_days_demand"
    ].bfill()

    return random_apple_sales.drop(
        columns=["inflation_multiplier", "harvest_effect", "month"],
    )
