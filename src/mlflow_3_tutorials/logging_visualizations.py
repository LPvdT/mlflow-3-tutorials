from mlflow_3_tutorials.lib.utils import (
    generate_apple_sales_data_with_promo_adjustment,
)

sales_data = generate_apple_sales_data_with_promo_adjustment(
    1_000, 25_000, -25.0
)
