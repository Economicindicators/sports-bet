"""野球固有特徴量 (5個): 投手成績、得点差パターン"""

import pandas as pd
import numpy as np


def add_baseball_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    野球固有の特徴量を追加。

    必要カラム: home_pitcher_id, away_pitcher_id, home_score, away_score, date
    """
    df = df.sort_values("date").reset_index(drop=True)

    # 1. 先発投手の過去勝率 (expanding, shift)
    # ホーム投手が先発した試合でのチーム勝率
    df["_hp_win"] = (df["home_score"] > df["away_score"]).astype(float)
    df["home_pitcher_wr"] = (
        df.groupby("home_pitcher_id")["_hp_win"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.5)

    df["_ap_win"] = (df["away_score"] > df["home_score"]).astype(float)
    df["away_pitcher_wr"] = (
        df.groupby("away_pitcher_id")["_ap_win"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.5)

    # 2. 先発投手の過去平均失点 (ERA的な指標)
    df["home_pitcher_era"] = (
        df.groupby("home_pitcher_id")["away_score"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(df["away_score"].mean())

    df["away_pitcher_era"] = (
        df.groupby("away_pitcher_id")["home_score"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(df["home_score"].mean())

    # 3. 投手力差
    df["pitcher_era_diff"] = df["away_pitcher_era"] - df["home_pitcher_era"]
    # (ホーム投手のERAが低い=ホーム有利 → 正の値)

    # 4. 先発投手の登板回数 (経験値)
    df["home_pitcher_starts"] = (
        df.groupby("home_pitcher_id").cumcount()
    )
    df["away_pitcher_starts"] = (
        df.groupby("away_pitcher_id").cumcount()
    )

    # 5. 得点差の大きさ (ロースコア/ハイスコア傾向)
    df["total_runs_avg"] = (
        (df["home_score"] + df["away_score"])
        .expanding().mean().shift(1)
    ).fillna(df["home_score"].mean() + df["away_score"].mean())

    # 一時カラム削除
    df = df.drop(columns=["_hp_win", "_ap_win"], errors="ignore")

    return df
