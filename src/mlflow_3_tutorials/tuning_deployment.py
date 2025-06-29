from typing import Any, TypeVar, cast

import keras
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from loguru import logger
from mlflow.models import infer_signature
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlflow_3_tutorials.lib.constants import WINE_QUALITY_DATA_URL

DType = TypeVar("DType", bound=np.generic)


def as_ndarray_dtype(arr: object) -> NDArray[DType]:
    return cast("NDArray[DType]", arr)


# Load data
data = pd.read_csv(WINE_QUALITY_DATA_URL, sep=";")

# Create train/validation/test splits
data_array = data.to_numpy()
del data

# Create train/validation/test splits
train_x, test_x, train_y, test_y = train_test_split(
    data_array[:, :-1],
    data_array[:, -1],
    test_size=0.25,
    random_state=42,
)

# Further split training data for validation
train_x, valid_x, train_y, valid_y = train_test_split(
    train_x,
    train_y,
    test_size=0.2,
    random_state=42,
)

# Create model signature for deployment
signature = infer_signature(train_x, train_y)


def create_and_train_model(
    learning_rate: float,
    momentum: float,
    epochs: int = 10,
) -> dict[str, Any]:
    """
    Create and train a neural network with specified hyperparameters.

    Returns:
        dict: Training results including model and metrics
    """

    # Normalize input features for better training stability
    mean = np.mean(train_x, axis=0)
    var = np.var(train_x, axis=0)

    # Define model architecture
    model = keras.Sequential([
        keras.Input([as_ndarray_dtype(train_x).shape[1]]),
        keras.layers.Normalization(mean=mean, variance=var),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),  # Add regularization
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

    # Compile with specified hyperparameters
    model.compile(
        optimizer=cast(
            "str",
            keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=momentum,
            ),
        ),
        loss=keras.losses.mean_squared_error,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    # Train with early stopping for efficiency
    early_stopping = keras.callbacks.EarlyStopping(
        patience=3,
        restore_best_weights=True,
    )

    # Train the model
    history = model.fit(
        train_x,
        train_y,
        validation_data=(valid_x, valid_y),
        epochs=epochs,
        batch_size=64,
        callbacks=[early_stopping],
        verbose="0",  # Reduce output for cleaner logs
    )

    # Evaluate on validation set
    val_loss, val_rmse = model.evaluate(valid_x, valid_y, verbose="0")

    return {
        "model": model,
        "val_rmse": val_rmse,
        "val_loss": val_loss,
        "history": history,
        "epochs_trained": len(history.history["loss"]),
    }
