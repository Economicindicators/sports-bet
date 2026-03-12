"""Optuna hyperparameter optimization for LightGBM."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def optimize_lgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    timeout: int = 600,
) -> dict:
    """
    Optuna LightGBM hyperparameter search.

    Returns:
        Best params dict.
    """
    import optuna
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(30),
                lgb.log_evaluation(period=0),
            ],
        )

        pred = model.predict(X_val)
        return roc_auc_score(y_val, pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info(f"Best AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return {
        **study.best_params,
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "n_estimators": 3000,
    }


def optimize_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 30,
    timeout: int = 600,
) -> dict:
    """Optuna XGBoost hyperparameter search."""
    import optuna
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "verbosity": 0,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "n_estimators": 2000,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info(f"Best XGB AUC: {study.best_value:.4f}")

    return {
        **study.best_params,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "verbosity": 0,
        "n_estimators": 3000,
    }
