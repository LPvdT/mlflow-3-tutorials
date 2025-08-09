import sys
from typing import cast

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from loguru import logger
from mlflow.data import pandas_dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch import nn

from mlflow_3_tutorials.lib.constants import LOG_LEVEL
from mlflow_3_tutorials.lib.dl_utils import (
    IrisClassifier,
    compute_accuracy,
    prepare_data,
)
from mlflow_3_tutorials.lib.utils import as_json

# Configure logger
logger.bind(name=__file__).add(sys.stderr, level=LOG_LEVEL)


def main() -> None:
    # Load Iris dataset and prepare the DataFrame
    iris = load_iris()
    iris_df, _iris_target = (
        pd.DataFrame(iris.data, columns=iris.feature_names),  # type: ignore
        iris.target,  # type: ignore
    )

    # Split into training and testing datasets
    train_df, test_df = train_test_split(
        iris_df, test_size=0.2, random_state=42
    )
    train_df = cast("pd.DataFrame", train_df)
    test_df = cast("pd.DataFrame", test_df)

    # Prepare training data
    train_dataset = pandas_dataset.from_pandas(train_df, name="train")
    X_train, y_train = prepare_data(train_dataset.df)

    # Define the PyTorch model and move it to the device
    input_size = X_train.shape[1]
    hidden_size = 16
    output_size = len(iris.target_names)  # type: ignore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scripted_model = IrisClassifier(input_size, hidden_size, output_size).to(
        device
    )
    scripted_model = torch.jit.script(scripted_model)

    # Start a run to represent the training job
    with mlflow.start_run() as run:
        # Load the training dataset with MLflow and link training metrics
        train_dataset = pandas_dataset.from_pandas(train_df, name="train")
        X_train, y_train = prepare_data(train_dataset.df)  # type: ignore

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(scripted_model.parameters(), lr=1e-2)

        for epoch in range(101):
            X_train, y_train = X_train.to(device), y_train.to(device)

            out = scripted_model(X_train)
            loss = criterion(out, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                # Log model on CPU to avoid signature error
                model_cpu = scripted_model.to("cpu")
                input_example = X_train.cpu().numpy()

                # Each newly created LoggedModel checkpoint is linked with its name and step
                model_info = mlflow.pytorch.log_model(  # type: ignore
                    pytorch_model=model_cpu,
                    name=f"torch-iris-{epoch}",
                    step=epoch,
                    input_example=input_example,
                )

                # Move the model back to the device
                scripted_model = scripted_model.to(device)

                # Log params to the run, LoggedModel inherits those params
                mlflow.log_params({
                    "n_layers": 3,
                    "activation": "ReLU",
                    "criterion": "CrossEntropyLoss",
                    "optimizer": "Adam",
                })

                # Log metric on training dataset at step and link to LoggedModel
                mlflow.log_metric(
                    "accuracy",
                    compute_accuracy(scripted_model, X_train, y_train),
                    epoch,
                    model_id=model_info.model_id,
                    dataset=train_dataset,  # type: ignore
                )

    # Search for the best and worst checkpoints
    ranked_checkpoints = mlflow.search_logged_models(
        filter_string=f"source_run_id='{run.info.run_id}'",
        order_by=[{"field_name": "metrics.accuracy", "ascending": False}],
        output_format="list",
    )

    best_checkpoint = ranked_checkpoints[0]
    logger.info(f"Best model: {as_json(best_checkpoint)}")
    logger.info(f"Metrics: {as_json(best_checkpoint.metrics)}")

    worst_checkpoint = ranked_checkpoints[-1]
    logger.warning(f"Worst model: {as_json(worst_checkpoint)}")
    logger.warning(f"Metrics: {as_json(worst_checkpoint.metrics)}")
