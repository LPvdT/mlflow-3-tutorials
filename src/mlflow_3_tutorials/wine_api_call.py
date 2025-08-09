import json
import sys

import requests
from loguru import logger

from mlflow_3_tutorials.lib.constants import LOG_LEVEL
from mlflow_3_tutorials.lib.utils import as_json

if __name__ == "__main__":
    # Configure logger
    logger.bind(name=__file__).add(sys.stderr, level=LOG_LEVEL)

    # Prepare test data
    test_wine = {
        "dataframe_split": {
            "columns": [
                "fixed acidity",
                "volatile acidity",
                "citric acid",
                "residual sugar",
                "chlorides",
                "free sulfur dioxide",
                "total sulfur dioxide",
                "density",
                "pH",
                "sulphates",
                "alcohol",
            ],
            "data": [
                [7.0, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3.0, 0.45, 8.8]
            ],
        }
    }

    # Make prediction request
    response = requests.post(
        "http://localhost:5002/invocations",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_wine),
        timeout=500,
    )

    prediction = response.json()
    logger.info(
        f"Predicted wine quality: {as_json(prediction['predictions'][0])}"
    )
