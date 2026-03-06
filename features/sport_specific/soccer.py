"""サッカー固有特徴量 (5個): 無失点率、引分率、ダービー"""

import pandas as pd
import numpy as np


def add_soccer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    サッカー固有の特徴量を追加。

    必要カラム: home_score, away_score, home_team_id, away_team_id, date
    """
    df = df.sort_values("date").reset_index(drop=True)

    # 1. クリーンシート率 (ホームチームの無失点率)
    df["_home_cs"] = (df["away_score"] == 0).astype(float)
    df["home_clean_sheet_rate"] = (
        df.groupby("home_team_id")["_home_cs"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.3)

    df["_away_cs"] = (df["home_score"] == 0).astype(float)
    df["away_clean_sheet_rate"] = (
        df.groupby("away_team_id")["_away_cs"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.3)

    # 2. 引分率
    df["_is_draw"] = (df["home_score"] == df["away_score"]).astype(float)
    df["home_draw_rate"] = (
        df.groupby("home_team_id")["_is_draw"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.25)

    df["away_draw_rate"] = (
        df.groupby("away_team_id")["_is_draw"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.25)

    # 3. 平均ゴール数
    df["league_avg_goals"] = (
        (df["home_score"] + df["away_score"])
        .expanding().mean().shift(1)
    ).fillna(2.5)

    # 一時カラム削除
    df = df.drop(columns=["_home_cs", "_away_cs", "_is_draw"], errors="ignore")

    return df
