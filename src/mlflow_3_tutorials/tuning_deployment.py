import sys
from operator import itemgetter
from typing import Any, TypeVar, cast

import keras
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from loguru import logger
from mlflow.models import infer_signature
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from mlflow_3_tutorials.lib.constants import LOG_LEVEL, WINE_QUALITY_DATA_URL
from mlflow_3_tutorials.lib.utils import as_json

# Configure logger
logger.bind(name=__file__).add(sys.stderr, level=LOG_LEVEL)


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


def objective(params: dict[str, Any]) -> dict[str, Any] | None:
    """
    Objective function for hyperparameter optimization.

    This function will be called by Hyperopt for each trial.
    """

    with mlflow.start_run(nested=True):
        # Log hyperparameters being tested
        logger.info(f"Hyperparameters: {as_json(params)}")
        mlflow.log_params({
            "learning_rate": params["learning_rate"],
            "momentum": params["momentum"],
            "optimizer": "SGD",
            "architecture": "64-32-1",
        })

        # Train model with current hyperparameters
        result = create_and_train_model(
            learning_rate=params["learning_rate"],
            momentum=params["momentum"],
            epochs=15,
        )

        # Log training results
        training_metrics = {
            "val_rmse": result["val_rmse"],
            "val_loss": result["val_loss"],
            "epochs_trained": result["epochs_trained"],
        }
        logger.info(f"training_metrics: {as_json(training_metrics)}")
        mlflow.log_metrics(training_metrics)

        # Log the trained model
        mlflow.tensorflow.log_model(  # type: ignore
            result["model"],
            name="model",
            signature=signature,
        )

        # Log training curves as artifacts
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(result["history"].history["loss"], label="Training Loss")
        plt.plot(result["history"].history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(
            result["history"].history["root_mean_squared_error"],
            label="Training RMSE",
        )
        plt.plot(
            result["history"].history["val_root_mean_squared_error"],
            label="Validation RMSE",
        )
        plt.title("Model RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_curves.png")
        mlflow.log_artifact("training_curves.png")
        plt.close()

        # Return loss for Hyperopt (it minimizes)
        return {"loss": result["val_rmse"], "status": STATUS_OK}


def main() -> None:
    # Define search space for hyperparameters
    search_space = {
        "learning_rate": hp.loguniform(
            "learning_rate",
            np.log(1e-5),
            np.log(1e-1),
        ),
        "momentum": hp.uniform("momentum", 0.0, 0.9),
    }

    logger.info("Search space defined:")
    logger.info("learning_rate: 1e-5 to 1e-1 (log-uniform)")
    logger.info("momentum: 0.0 to 0.9 (uniform)")

    # Create or set experiment
    experiment_name = "wine-quality-optimization"
    mlflow.set_experiment(experiment_name)

    logger.info(
        f"Starting hyperparameter optimization experiment: {experiment_name}",
    )
    logger.info("This will run 15 trials to find optimal hyperparameters...")

    with mlflow.start_run(run_name="hyperparameter-sweep"):
        # Log experiment metadata
        mlflow.log_params({
            "optimization_method": "Tree-structured Parzen Estimator (TPE)",
            "max_evaluations": 15,
            "objective_metric": "validation_rmse",
            "dataset": "wine-quality",
            "model_type": "neural_network",
        })

        # Run optimization
        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=15,
            trials=trials,
            verbose=True,
        )

        # Find and log best results
        best_trial = min(trials.results, key=itemgetter("loss"))
        best_rmse = best_trial["loss"]

        # Log optimization results
        if best_params is not None:
            logger.info(f"Best parameters found: {as_json(best_params)}")
            mlflow.log_params({
                "best_learning_rate": best_params["learning_rate"],
                "best_momentum": best_params["momentum"],
            })
        else:
            logger.error("Optimization failed to find valid parameters.")

        final_metrics = {
            "best_val_rmse": best_rmse,
            "total_trials": len(trials.trials),
            "optimization_completed": 1,
        }
        logger.info(f"final_metrics: {as_json(final_metrics)}")
        mlflow.log_metrics(final_metrics)


if __name__ == "__main__":
    main()
