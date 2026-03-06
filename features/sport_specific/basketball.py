"""バスケ固有特徴量 (5個): 合計得点、ペース、連戦"""

import pandas as pd
import numpy as np


def add_basketball_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    バスケ固有の特徴量を追加。

    必要カラム: home_score, away_score, home_team_id, away_team_id, date
    """
    df = df.sort_values("date").reset_index(drop=True)

    # 1. 合計得点 (オーバー/アンダー傾向)
    df["total_points_avg"] = (
        (df["home_score"] + df["away_score"])
        .expanding().mean().shift(1)
    ).fillna(df["home_score"].mean() + df["away_score"].mean())

    # 2. チーム別ペース (平均得点)
    df["home_pace"] = (
        df.groupby("home_team_id")["home_score"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(df["home_score"].mean())

    df["away_pace"] = (
        df.groupby("away_team_id")["away_score"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(df["away_score"].mean())

    # 3. ペース差
    df["pace_diff"] = df["home_pace"] - df["away_pace"]

    # 4. 僅差試合率 (5点差以内)
    df["_close_game"] = (abs(df["home_score"] - df["away_score"]) <= 5).astype(float)
    df["home_close_game_rate"] = (
        df.groupby("home_team_id")["_close_game"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.3)

    # 5. 大量リード率 (15点差以上で勝利)
    df["_blowout"] = (
        ((df["home_score"] - df["away_score"]) >= 15).astype(float)
    )
    df["home_blowout_rate"] = (
        df.groupby("home_team_id")["_blowout"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.1)

    # 一時カラム削除
    df = df.drop(columns=["_close_game", "_blowout"], errors="ignore")

    return df
