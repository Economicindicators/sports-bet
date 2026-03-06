"""チーム成績特徴量 (12個)"""

import pandas as pd
import numpy as np


def add_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    チーム成績の特徴量を追加。

    必要カラム: home_team_id, away_team_id, home_score, away_score, date
    """
    df = df.sort_values("date").reset_index(drop=True)

    # ホームチーム視点の勝敗
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(float)
    df["away_win"] = (df["away_score"] > df["home_score"]).astype(float)

    # 直近N試合の勝率 (expanding + shift で未来リーク防止)
    for window in [5, 10, 20]:
        df[f"home_wr_{window}"] = (
            df.groupby("home_team_id")["home_win"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        ).fillna(0.5)

        df[f"away_wr_{window}"] = (
            df.groupby("away_team_id")["away_win"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
        ).fillna(0.5)

    # 得点/失点平均 (直近10試合)
    df["home_score_avg"] = (
        df.groupby("home_team_id")["home_score"]
        .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
    ).fillna(df["home_score"].mean())

    df["away_score_avg"] = (
        df.groupby("away_team_id")["away_score"]
        .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
    ).fillna(df["away_score"].mean())

    df["home_concede_avg"] = (
        df.groupby("home_team_id")["away_score"]
        .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
    ).fillna(df["away_score"].mean())

    df["away_concede_avg"] = (
        df.groupby("away_team_id")["home_score"]
        .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
    ).fillna(df["home_score"].mean())

    # ストリーク (連勝/連敗)
    def _streak(series):
        streak = []
        current = 0
        for val in series:
            if pd.isna(val):
                streak.append(0)
                continue
            if val == 1:
                current = max(0, current) + 1
            else:
                current = min(0, current) - 1
            streak.append(current)
        return pd.Series(streak, index=series.index).shift(1).fillna(0)

    df["home_streak"] = df.groupby("home_team_id")["home_win"].transform(_streak)
    df["away_streak"] = df.groupby("away_team_id")["away_win"].transform(_streak)

    # 不要カラム削除
    df = df.drop(columns=["home_win", "away_win"], errors="ignore")

    return df
