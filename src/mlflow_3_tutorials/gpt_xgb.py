import json
import logging
import sys
import textwrap
import warnings

import optuna
import xgboost as xgb
from loguru import logger
from optuna.exceptions import ExperimentalWarning
from optuna.integration import XGBoostPruningCallback
from optuna.trial._frozen import FrozenTrial
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

logger.remove()
logger.add(sys.stderr, level=logging.DEBUG)
optuna.logging.set_verbosity(optuna.logging.INFO)


def objective(trial: optuna.Trial) -> float:
    bunch = load_breast_cancer(as_frame=True)
    X = bunch["data"]  # type:ignore
    y = bunch["target"]  # type:ignore
    feature_names = bunch["feature_names"].tolist()  # type:ignore

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)  # type:ignore
    dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_names)  # type:ignore

    param = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "device": "cuda",  # GPU acceleration
        "tree_method": "hist",
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    pruning_callback = XGBoostPruningCallback(trial, "validation-logloss")

    bst = xgb.train(
        param,
        dtrain,
        evals=[(dvalid, "validation")],
        num_boost_round=500,  # let pruning stop early if needed
        callbacks=[pruning_callback],
        verbose_eval=None,
        early_stopping_rounds=1,
    )

    preds = bst.predict(dvalid)
    pred_labels = (preds > 0.5).astype(int)  # noqa

    return accuracy_score(y_valid, pred_labels)


if __name__ == "__main__":
    base_pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=500,  # sync with num_boost_round
        reduction_factor=3,
    )

    warnings.simplefilter("ignore", ExperimentalWarning)
    pruner = optuna.pruners.PatientPruner(base_pruner, patience=2)
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,  # random exploration before TPE kicks in
        multivariate=True,  # joint distributions for better correlation handling
        group=True,  # improves handling of conditional params
    )

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=1000,
        timeout=600,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    trial: FrozenTrial = study.best_trial
    logger.info("Best trial:")
    logger.info(f"  Accuracy: {study.best_value:.4f}")
    logger.info(
        f"  Params:\n{textwrap.indent(json.dumps(trial.params, indent=2, sort_keys=True), ' ' * 4)}"
    )
