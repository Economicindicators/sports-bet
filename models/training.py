"""学習パイプライン + 時系列CV"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.constants import CV_FOLDS, CV_GAP_DAYS, EARLY_STOPPING_ROUNDS
from config.settings import MODELS_DIR
from models.lgbm_model import LightGBMModel

logger = logging.getLogger(__name__)


def time_series_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = CV_FOLDS,
    gap_days: int = CV_GAP_DAYS,
) -> list[dict]:
    """
    時系列CV (expanding window + gap)。

    Args:
        df: 特徴量付きDataFrame (date, target, feature_cols 必須)
        feature_cols: 特徴量カラムリスト
        n_splits: CV分割数
        gap_days: train/val間のギャップ日数

    Returns:
        各foldのメトリクスリスト
    """
    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].unique()
    n_dates = len(dates)

    if n_dates < n_splits + 1:
        logger.warning(f"Not enough dates ({n_dates}) for {n_splits}-fold CV")
        return []

    fold_size = n_dates // (n_splits + 1)
    results = []

    for fold in range(n_splits):
        # Expanding window: train = 先頭 ～ fold境界, val = gap後 ～ 次の境界
        train_end_idx = (fold + 1) * fold_size
        val_start_idx = train_end_idx  # gap_daysは日付ベースで適用

        train_end_date = dates[min(train_end_idx, n_dates - 1)]
        gap_date = train_end_date + timedelta(days=gap_days)

        if fold < n_splits - 1:
            val_end_idx = (fold + 2) * fold_size
            val_end_date = dates[min(val_end_idx, n_dates - 1)]
        else:
            val_end_date = dates[-1]

        train_mask = df["date"] <= train_end_date
        val_mask = (df["date"] > gap_date) & (df["date"] <= val_end_date)

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, "target"]
        X_val = df.loc[val_mask, feature_cols]
        y_val = df.loc[val_mask, "target"]

        if len(X_train) < 50 or len(X_val) < 10:
            logger.warning(f"Fold {fold}: insufficient data (train={len(X_train)}, val={len(X_val)})")
            continue

        logger.info(
            f"Fold {fold}: train={len(X_train)} (～{train_end_date}), "
            f"val={len(X_val)} ({gap_date}～{val_end_date})"
        )

        model = LightGBMModel()
        metrics = model.train(
            X_train, y_train, X_val, y_val,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        metrics["fold"] = fold
        metrics["train_size"] = len(X_train)
        metrics["val_size"] = len(X_val)
        metrics["train_end_date"] = str(train_end_date)
        metrics["val_date_range"] = f"{gap_date}~{val_end_date}"
        results.append(metrics)

    if results:
        avg_auc = np.mean([r.get("val_auc", 0) for r in results])
        avg_ll = np.mean([r.get("val_logloss", 0) for r in results])
        logger.info(f"CV Results: Avg AUC={avg_auc:.4f}, Avg LogLoss={avg_ll:.4f}")

    return results


def train_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    sport_code: str,
    model_version: str = "v1",
    val_ratio: float = 0.2,
) -> tuple[LightGBMModel, dict]:
    """
    最終モデルを学習する。

    Args:
        df: 特徴量付きDataFrame
        feature_cols: 特徴量カラムリスト
        sport_code: スポーツコード
        model_version: モデルバージョン
        val_ratio: validation比率 (時系列で末尾から)

    Returns:
        (trained_model, metrics)
    """
    df = df.sort_values("date").reset_index(drop=True)

    # 時系列split
    split_idx = int(len(df) * (1 - val_ratio))
    X_train = df.iloc[:split_idx][feature_cols]
    y_train = df.iloc[:split_idx]["target"]
    X_val = df.iloc[split_idx:][feature_cols]
    y_val = df.iloc[split_idx:]["target"]

    logger.info(f"Training: {len(X_train)} train, {len(X_val)} val")

    model = LightGBMModel()
    metrics = model.train(
        X_train, y_train, X_val, y_val,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    # モデル保存
    model_path = MODELS_DIR / f"{sport_code}_{model_version}.lgb"
    model.save(model_path)

    return model, metrics


def load_model(sport_code: str, model_version: str = "v1") -> LightGBMModel:
    """保存済みモデルを読み込み"""
    model_path = MODELS_DIR / f"{sport_code}_{model_version}.lgb"
    model = LightGBMModel()
    model.load(model_path)
    return model
