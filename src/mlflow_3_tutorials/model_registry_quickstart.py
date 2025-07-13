import mlflow
import mlflow.sklearn
from loguru import logger
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlflow_3_tutorials.lib.utils import as_json


def main() -> None:
    with mlflow.start_run() as _run:
        X, y = make_regression(  # type: ignore
            n_features=4, n_informative=2, random_state=0, shuffle=False
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        params = {
            "max_depth": 2,
            "random_state": 42,
            "warm_start": False,
            "criterion": "squared_error",
        }
        model = RandomForestRegressor(**params)
        logger.info(f"Training model with parameters: {as_json(params)}")
        model.fit(X_train, y_train)

        # Log parameters and metrics using the MLflow APIs
        mlflow.log_params(params)

        y_pred = model.predict(X_test)
        mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})

        # Log the sklearn model and register as version 1
        logger.info("Logging sklearn model...")
        mlflow.sklearn.log_model(  # type: ignore
            sk_model=model,
            name="sklearn-model",
            input_example=X_train,
            registered_model_name="sk-learn-random-forest-reg-model",
        )

        # Set alias and tags
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            "sk-learn-random-forest-reg-model", "the_best_model_ever", "1"
        )
        client.set_model_version_tag(
            "sk-learn-random-forest-reg-model",
            "1",
            "problem_type",
            "regression",
        )
        mv = client.get_model_version_by_alias(
            "sk-learn-random-forest-reg-model", "the_best_model_ever"
        )
        logger.info(mv.tags)

        # Loadmodel from registry using alias
        model_name = "sk-learn-random-forest-reg-model"
        model_alias = "the_best_model_ever"
        model_uri = f"models:/{model_name}@{model_alias}"

        logger.info(f"Loading model from: {model_uri}...")
        model_loaded = mlflow.sklearn.load_model(model_uri=model_uri)  # type: ignore
        logger.info(model_loaded)

        # Generate a new dataset for prediction and predict
        X_new, _ = make_regression(  # type: ignore
            n_features=4, n_informative=2, random_state=0, shuffle=False
        )
        y_pred_new = model_loaded.predict(X_new)  # type: ignore

        logger.info(f"Predictions: {as_json(y_pred_new[:10].tolist())}")


if __name__ == "__main__":
    main()
