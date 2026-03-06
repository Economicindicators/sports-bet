"""直接対決特徴量 (4個)"""

import pandas as pd
import numpy as np


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    直接対決 (H2H) の特徴量を追加。

    必要カラム: home_team_id, away_team_id, home_score, away_score, date
    """
    df = df.sort_values("date").reset_index(drop=True)

    # 対戦ペアキー (小さいID-大きいIDで統一)
    df["_pair"] = df.apply(
        lambda r: f"{min(r['home_team_id'], r['away_team_id'])}_{max(r['home_team_id'], r['away_team_id'])}",
        axis=1,
    )
    # このペアでホームチームが「小さいID側」かどうか
    df["_home_is_first"] = df["home_team_id"] < df["away_team_id"]

    # ペアごとの対戦結果 (first側の勝率で統一)
    df["_first_win"] = np.where(
        df["_home_is_first"],
        (df["home_score"] > df["away_score"]).astype(float),
        (df["away_score"] > df["home_score"]).astype(float),
    )

    # 1. H2H勝率 (expanding, shift)
    df["_h2h_first_wr"] = (
        df.groupby("_pair")["_first_win"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.5)

    # ホーム視点に変換
    df["h2h_home_wr"] = np.where(
        df["_home_is_first"], df["_h2h_first_wr"], 1.0 - df["_h2h_first_wr"]
    )

    # 2. H2Hアウェイ勝率
    df["h2h_away_wr"] = 1.0 - df["h2h_home_wr"]

    # 3. H2H平均得点差 (ホーム視点)
    df["_score_diff"] = df["home_score"] - df["away_score"]
    df["h2h_score_diff"] = (
        df.groupby("_pair")["_score_diff"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.0)
    # ペアの向き補正
    df["h2h_score_diff"] = np.where(
        df["_home_is_first"], df["h2h_score_diff"], -df["h2h_score_diff"]
    )

    # 4. H2H対戦数
    df["h2h_count"] = (
        df.groupby("_pair").cumcount()
    )

    # 一時カラム削除
    df = df.drop(
        columns=["_pair", "_home_is_first", "_first_win", "_h2h_first_wr", "_score_diff"],
        errors="ignore",
    )

    return df
