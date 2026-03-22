"""Ensemble model: LightGBM + XGBoost + CatBoost (stacking)"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EnsembleModel:
    """3-model ensemble with simple averaging or stacking."""

    def __init__(self, weights: Optional[list[float]] = None):
        self.models = {}  # name -> model object
        self.feature_names: list[str] = []
        self.weights = weights or [0.5, 0.25, 0.25]  # lgb, xgb, cat

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50,
        lgb_params: Optional[dict] = None,
        xgb_params: Optional[dict] = None,
        cat_params: Optional[dict] = None,
    ) -> dict:
        self.feature_names = list(X_train.columns)
        metrics = {}

        # --- LightGBM ---
        metrics["lgb"] = self._train_lgb(
            X_train, y_train, X_val, y_val, early_stopping_rounds, lgb_params
        )

        # --- XGBoost ---
        metrics["xgb"] = self._train_xgb(
            X_train, y_train, X_val, y_val, early_stopping_rounds, xgb_params
        )

        # --- CatBoost ---
        metrics["cat"] = self._train_cat(
            X_train, y_train, X_val, y_val, early_stopping_rounds, cat_params
        )

        # Ensemble metrics
        if X_val is not None and y_val is not None:
            from sklearn.metrics import log_loss, roc_auc_score

            y_pred = self.predict_proba(X_val)
            metrics["ensemble_auc"] = roc_auc_score(y_val, y_pred)
            metrics["ensemble_logloss"] = log_loss(y_val, y_pred)
            logger.info(
                f"Ensemble AUC: {metrics['ensemble_auc']:.4f}, "
                f"LogLoss: {metrics['ensemble_logloss']:.4f}"
            )

        return metrics

    def _train_lgb(self, X_train, y_train, X_val, y_val, es_rounds, params):
        import lightgbm as lgb

        default = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "max_depth": 5,
            "learning_rate": 0.01,
            "n_estimators": 3000,
            "min_child_samples": 20,
            "subsample": 0.7,
            "colsample_bytree": 0.6,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "verbosity": -1,
            "bagging_freq": 5,
        }
        p = {**default, **(params or {})}
        n_est = p.pop("n_estimators", 3000)

        train_data = lgb.Dataset(X_train, label=y_train)
        callbacks = [lgb.log_evaluation(period=200)]
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val)
            valid_sets.append(val_data)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(es_rounds))

        model = lgb.train(
            p, train_data, num_boost_round=n_est,
            valid_sets=valid_sets, valid_names=valid_names,
            callbacks=callbacks,
        )
        self.models["lgb"] = model

        result = {"best_iteration": model.best_iteration}
        if X_val is not None:
            from sklearn.metrics import roc_auc_score
            pred = model.predict(X_val[self.feature_names])
            result["val_auc"] = roc_auc_score(y_val, pred)
        logger.info(f"LGB AUC: {result.get('val_auc', 'N/A')}")
        return result

    def _train_xgb(self, X_train, y_train, X_val, y_val, es_rounds, params):
        import xgboost as xgb

        default = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "learning_rate": 0.01,
            "n_estimators": 3000,
            "min_child_weight": 5,
            "subsample": 0.7,
            "colsample_bytree": 0.6,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "verbosity": 0,
            "tree_method": "hist",
        }
        p = {**default, **(params or {})}

        model = xgb.XGBClassifier(**p)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        model.fit(X_train, y_train, **fit_params)
        self.models["xgb"] = model

        result = {}
        if X_val is not None:
            from sklearn.metrics import roc_auc_score
            pred = model.predict_proba(X_val)[:, 1]
            result["val_auc"] = roc_auc_score(y_val, pred)
        logger.info(f"XGB AUC: {result.get('val_auc', 'N/A')}")
        return result

    def _train_cat(self, X_train, y_train, X_val, y_val, es_rounds, params):
        from catboost import CatBoostClassifier

        default = {
            "iterations": 3000,
            "depth": 5,
            "learning_rate": 0.01,
            "l2_leaf_reg": 5.0,
            "verbose": 0,
            "eval_metric": "Logloss",
            "early_stopping_rounds": es_rounds,
        }
        p = {**default, **(params or {})}

        model = CatBoostClassifier(**p)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = (X_val, y_val)

        model.fit(X_train, y_train, **fit_params)
        self.models["cat"] = model

        result = {}
        if X_val is not None:
            from sklearn.metrics import roc_auc_score
            pred = model.predict_proba(X_val)[:, 1]
            result["val_auc"] = roc_auc_score(y_val, pred)
        logger.info(f"CAT AUC: {result.get('val_auc', 'N/A')}")
        return result

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of all model predictions."""
        preds = []
        w = []

        # Each sub-model may have been trained with different feature counts.
        # Pass by .values to avoid column name mismatch, slice to model's expected count.

        if "lgb" in self.models:
            n = self.models["lgb"].num_feature()
            X_lgb = X.iloc[:, :n].values if X.shape[1] >= n else X.values
            preds.append(self.models["lgb"].predict(X_lgb))
            w.append(self.weights[0])

        if "xgb" in self.models:
            try:
                n = len(self.models["xgb"].get_booster().feature_names or [])
                X_xgb = X.iloc[:, :n].values if n and X.shape[1] >= n else X.values
                preds.append(self.models["xgb"].predict_proba(X_xgb)[:, 1])
                w.append(self.weights[1])
            except Exception as e:
                logger.warning(f"XGB predict failed (feature mismatch), skipping: {e}")

        if "cat" in self.models:
            try:
                n = len(self.models["cat"].feature_names_ or [])
                X_cat = X.iloc[:, :n].values if n and X.shape[1] >= n else X.values
                preds.append(self.models["cat"].predict_proba(X_cat)[:, 1])
                w.append(self.weights[2])
            except Exception as e:
                logger.warning(f"CAT predict failed (feature mismatch), skipping: {e}")

        if not preds:
            raise RuntimeError("No models trained")

        w = np.array(w) / sum(w)
        return sum(p * wi for p, wi in zip(preds, w))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save LGB
        if "lgb" in self.models:
            self.models["lgb"].save_model(str(path.with_suffix(".lgb")))

        # Save XGB
        if "xgb" in self.models:
            self.models["xgb"].save_model(str(path.with_suffix(".xgb")))

        # Save CatBoost
        if "cat" in self.models:
            self.models["cat"].save_model(str(path.with_suffix(".cat")))

        # Meta
        meta = {
            "feature_names": self.feature_names,
            "weights": self.weights,
            "models": list(self.models.keys()),
        }
        path.with_suffix(".ensemble.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )
        logger.info(f"Ensemble saved to {path}")

    def load(self, path: Path) -> None:
        path = Path(path)

        meta_path = path.with_suffix(".ensemble.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Ensemble meta not found: {meta_path}")

        meta = json.loads(meta_path.read_text())
        self.feature_names = meta["feature_names"]
        self.weights = meta.get("weights", [0.4, 0.35, 0.25])

        if "lgb" in meta["models"]:
            import lightgbm as lgb
            self.models["lgb"] = lgb.Booster(model_file=str(path.with_suffix(".lgb")))

        if "xgb" in meta["models"]:
            import xgboost as xgb
            m = xgb.XGBClassifier()
            m.load_model(str(path.with_suffix(".xgb")))
            self.models["xgb"] = m

        if "cat" in meta["models"]:
            from catboost import CatBoostClassifier
            m = CatBoostClassifier()
            m.load_model(str(path.with_suffix(".cat")))
            self.models["cat"] = m

        logger.info(f"Ensemble loaded from {path} ({list(self.models.keys())})")
