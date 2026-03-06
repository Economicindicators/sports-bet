"""LightGBM モデルラッパー (keiba-aiベース)"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LightGBMModel:
    """LightGBM binary classifier"""

    DEFAULT_PARAMS = {
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

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = None
        self.feature_names: list[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50,
    ) -> dict:
        """モデルを学習する"""
        import lightgbm as lgb

        self.feature_names = list(X_train.columns)

        train_data = lgb.Dataset(X_train, label=y_train)

        callbacks = [lgb.log_evaluation(period=100)]
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val)
            valid_sets.append(val_data)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(early_stopping_rounds))

        params = {k: v for k, v in self.params.items() if k != "n_estimators"}

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.params["n_estimators"],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # メトリクス
        metrics = {"best_iteration": self.model.best_iteration}
        if X_val is not None:
            from sklearn.metrics import log_loss, roc_auc_score

            y_pred = self.predict_proba(X_val)
            metrics["val_logloss"] = log_loss(y_val, y_pred)
            metrics["val_auc"] = roc_auc_score(y_val, y_pred)
            logger.info(
                f"Val AUC: {metrics['val_auc']:.4f}, "
                f"Val LogLoss: {metrics['val_logloss']:.4f}"
            )

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """確率を予測する [0, 1]"""
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict(X[self.feature_names])

    def get_feature_importance(self, top_n: int = 50) -> pd.DataFrame:
        """特徴量重要度を返す"""
        if self.model is None:
            raise RuntimeError("Model not trained")

        importance = self.model.feature_importance(importance_type="gain")
        fi_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return fi_df.head(top_n)

    def save(self, path: Path) -> None:
        """モデルを保存"""
        if self.model is None:
            raise RuntimeError("Model not trained")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(path))

        # メタデータも保存
        meta_path = path.with_suffix(".json")
        meta = {
            "params": self.params,
            "feature_names": self.feature_names,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """モデルを読み込み"""
        import lightgbm as lgb

        path = Path(path)
        self.model = lgb.Booster(model_file=str(path))

        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.params = meta.get("params", self.params)
            self.feature_names = meta.get("feature_names", [])

        logger.info(f"Model loaded from {path}")
