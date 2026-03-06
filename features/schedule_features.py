"""日程特徴量 (4個)"""

import pandas as pd
import numpy as np


def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    休養日数・連戦の特徴量を追加。

    必要カラム: home_team_id, away_team_id, date
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["_date_num"] = pd.to_datetime(df["date"]).astype(np.int64) // 10**9 // 86400  # days

    # --- ホームチームの休養日数 ---
    def _rest_days(group):
        """前回試合からの日数"""
        return group.diff().fillna(7).clip(upper=30)

    df["home_rest_days"] = (
        df.groupby("home_team_id")["_date_num"].transform(_rest_days)
    )

    # アウェイの前回試合も考慮 (ホーム/アウェイ両方)
    # ← 完全にやるにはhome/away両方の出場を追跡する必要がある
    # 簡略化: 同チームのホーム試合間隔のみ
    df["away_rest_days"] = (
        df.groupby("away_team_id")["_date_num"].transform(_rest_days)
    )

    # 1. 休養日数
    # (above)

    # 2. 休養日数差
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]

    # 3. 直近7日間の試合数 (ホーム)
    df["_date_dt"] = pd.to_datetime(df["date"])

    def _games_in_7d(group):
        """直近7日間の試合数"""
        result = []
        dates = group.values
        seven_days = np.timedelta64(7, 'D')
        zero_days = np.timedelta64(0, 'D')
        for i, d in enumerate(dates):
            count = sum(1 for j in range(i) if zero_days < (d - dates[j]) <= seven_days)
            result.append(count)
        return pd.Series(result, index=group.index)

    df["home_games_7d"] = df.groupby("home_team_id")["_date_dt"].transform(_games_in_7d)
    df["away_games_7d"] = df.groupby("away_team_id")["_date_dt"].transform(_games_in_7d)

    # 4. 連戦フラグ (前日に試合)
    df["home_back_to_back"] = (df["home_rest_days"] <= 1).astype(int)
    df["away_back_to_back"] = (df["away_rest_days"] <= 1).astype(int)

    # 一時カラム削除
    df = df.drop(columns=["_date_num", "_date_dt"], errors="ignore")

    return df
