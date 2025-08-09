import sys

from loguru import logger

from mlflow_3_tutorials.lib.constants import LOG_LEVEL
from mlflow_3_tutorials.lib.utils import (
    generate_apple_sales_data_with_promo_adjustment,
)

logger.bind(name="runner").add(sys.stderr, level=LOG_LEVEL)

sales_data = generate_apple_sales_data_with_promo_adjustment(
    1_000, 25_000, -25.0
)
