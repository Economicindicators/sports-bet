"""バスケ固有特徴量 (14個): 合計得点、ペース、僅差率、大量リード率、B2B、NBA Advanced Stats"""

from __future__ import annotations

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


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

    # 6. ホームチームBack-to-Back (前日も試合 → パフォーマンス低下)
    df["date"] = pd.to_datetime(df["date"])
    df["_home_prev_date"] = df.groupby("home_team_id")["date"].shift(1)
    df["home_b2b"] = (
        (df["date"] - df["_home_prev_date"]).dt.days == 1
    ).astype(float).fillna(0.0)

    # 7-14. NBA Advanced Stats (DBから取得)
    df = _add_nba_advanced_stats(df)

    # 一時カラム削除
    df = df.drop(columns=["_close_game", "_blowout", "_home_prev_date"], errors="ignore")

    return df


def _add_nba_advanced_stats(df: pd.DataFrame) -> pd.DataFrame:
    """NbaTeamStatsテーブルからアドバンスドスタッツを特徴量として追加"""
    try:
        from database.models import NbaTeamStats, get_session
        session = get_session()
        stats = session.query(NbaTeamStats).all()
        session.close()
    except Exception as e:
        logger.warning(f"Could not load NBA stats: {e}")
        return df

    if not stats:
        logger.info("No NBA advanced stats in DB, skipping")
        return df

    # team_id → {stat_type: value} のマップを構築
    team_stats = {}
    for s in stats:
        if s.team_id not in team_stats:
            team_stats[s.team_id] = {}
        team_stats[s.team_id][s.stat_type] = s.value

    # 使用するスタッツ
    stat_types = ["off_rtg", "def_rtg", "net_rtg", "pace"]

    # ホーム/アウェイそれぞれのスタッツを追加
    for stat in stat_types:
        home_vals = df["home_team_id"].map(
            lambda tid: team_stats.get(tid, {}).get(stat, np.nan)
        )
        away_vals = df["away_team_id"].map(
            lambda tid: team_stats.get(tid, {}).get(stat, np.nan)
        )

        df[f"home_{stat}"] = home_vals
        df[f"away_{stat}"] = away_vals

    # 派生特徴量
    if "home_off_rtg" in df.columns:
        # 攻守効率差
        df["off_rtg_diff"] = df["home_off_rtg"] - df["away_off_rtg"]
        df["def_rtg_diff"] = df["home_def_rtg"] - df["away_def_rtg"]
        # 攻守マッチアップ (ホーム攻撃力 vs アウェイ守備力)
        df["home_matchup_edge"] = df["home_off_rtg"] - df["away_def_rtg"]
        df["away_matchup_edge"] = df["away_off_rtg"] - df["home_def_rtg"]

    # NaN埋め (DBにないチームはリーグ平均で)
    for col in df.columns:
        if any(s in col for s in ["off_rtg", "def_rtg", "net_rtg", "pace", "matchup_edge", "rtg_diff"]):
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())

    matched = sum(1 for tid in df["home_team_id"].unique() if tid in team_stats)
    logger.info(f"NBA advanced stats: {matched}/{df['home_team_id'].nunique()} teams matched")

    return df
