"""ハンデ系特徴量 (5個)"""

import pandas as pd
import numpy as np


def add_handicap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ハンデ関連の特徴量を追加。

    入力dfに必要なカラム: handicap_value, handicap_team_is_home, league_code
    """
    # 1. ハンデ値そのまま
    # (already in df as handicap_value)

    # 2. 正規化ハンデ (リーグ平均との偏差)
    league_mean = df.groupby("league_code")["handicap_value"].transform("mean")
    league_std = df.groupby("league_code")["handicap_value"].transform("std").replace(0, 1)
    df["handicap_normalized"] = (df["handicap_value"] - league_mean) / league_std

    # 3. ハンデ方向 (ホーム有利=1, アウェイ有利=0)
    df["handicap_direction"] = df["handicap_team_is_home"].astype(int)

    # 4. ハンデチームの過去精度 (expanding windowで未来リーク防止)
    # handicap_team_idごとの過去のfavorable率
    df = df.sort_values("date").reset_index(drop=True)
    df["handi_team_past_accuracy"] = (
        df.groupby("handicap_team_id")["target"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )
    df["handi_team_past_accuracy"] = df["handi_team_past_accuracy"].fillna(0.5)

    # 5. リーグ平均との偏差
    df["handicap_deviation"] = df["handicap_value"] - league_mean

    return df
