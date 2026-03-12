"""メイン特徴量パイプライン: DB → DataFrame → 特徴量行列"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from config.constants import FAVORABLE_PAYOUT_THRESHOLD
from database.models import Match, HandicapData, Team, Player, get_session
from features.handicap_features import add_handicap_features
from features.team_features import add_team_features
from features.matchup_features import add_matchup_features
from features.league_features import add_league_features
from features.schedule_features import add_schedule_features
from features.elo_features import add_elo_features
from features.form_features import add_form_features
from features.line_movement_features import add_line_movement_features
from features.odds_features import add_odds_features

logger = logging.getLogger(__name__)

# スポーツ別の特徴量モジュール
SPORT_FEATURE_FUNCS = {
    "baseball": "features.sport_specific.baseball",
    "soccer": "features.sport_specific.soccer",
    "basketball": "features.sport_specific.basketball",
}


def load_matches_df(
    sport_code: Optional[str] = None,
    league_code: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    session: Optional[Session] = None,
    include_scheduled: bool = False,
) -> pd.DataFrame:
    """DBから試合データをDataFrameとして読み込む"""
    sess = session or get_session()
    q = sess.query(
        Match.match_id,
        Match.sport_code,
        Match.league_code,
        Match.date,
        Match.home_team_id,
        Match.away_team_id,
        Match.home_score,
        Match.away_score,
        Match.venue,
        Match.status,
        Match.home_pitcher_id,
        Match.away_pitcher_id,
        HandicapData.handicap_team_id,
        HandicapData.handicap_value,
        HandicapData.result_type,
        HandicapData.payout_rate,
    ).join(HandicapData, Match.match_id == HandicapData.match_id)

    if sport_code:
        q = q.filter(Match.sport_code == sport_code)
    if league_code:
        q = q.filter(Match.league_code == league_code)
    if start_date:
        q = q.filter(Match.date >= start_date)
    if end_date:
        q = q.filter(Match.date <= end_date)

    if include_scheduled:
        q = q.filter(Match.status.in_(["finished", "scheduled"]))
    else:
        q = q.filter(Match.status == "finished")
    q = q.order_by(Match.date)

    rows = q.all()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "match_id", "sport_code", "league_code", "date",
        "home_team_id", "away_team_id", "home_score", "away_score",
        "venue", "status",
        "home_pitcher_id", "away_pitcher_id",
        "handicap_team_id", "handicap_value", "result_type", "payout_rate",
    ])

    if not session:
        sess.close()

    return df


def build_features(
    df: pd.DataFrame,
    sport_code: Optional[str] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    DataFrameに全特徴量を追加する。

    Args:
        df: load_matches_dfの出力
        sport_code: スポーツコード (None=自動判定)

    Returns:
        (feature_df, feature_columns)
    """
    if df.empty:
        return df, []

    # 自動判定
    if sport_code is None:
        sport_code = df["sport_code"].iloc[0]

    # ターゲット: ホームチームが勝つか (シンプル勝敗予測)
    # scheduled試合はスコアがNone → targetもNaN
    # EV計算時にハンデを後から考慮する
    home_score = pd.to_numeric(df["home_score"], errors="coerce")
    away_score = pd.to_numeric(df["away_score"], errors="coerce")
    df["target"] = (home_score > away_score).where(home_score.notna() & away_score.notna()).astype(float)

    # ハンデチームがホームかどうか
    df["handicap_team_is_home"] = (
        df["handicap_team_id"] == df["home_team_id"]
    ).astype(int)

    # ホームアドバンテージ
    df["is_home_favorite"] = df["handicap_team_is_home"]

    # --- 共通特徴量 ---
    logger.info("Adding Elo ratings...")
    df = add_elo_features(df)

    logger.info("Adding handicap features...")
    df = add_handicap_features(df)

    logger.info("Adding team features...")
    df = add_team_features(df)

    logger.info("Adding matchup features...")
    df = add_matchup_features(df)

    logger.info("Adding league features...")
    df = add_league_features(df)

    logger.info("Adding schedule features...")
    df = add_schedule_features(df)

    # --- 直近フォーム特徴量 ---
    logger.info("Adding form features...")
    df = add_form_features(df, sport_code)

    # --- ライン移動特徴量 ---
    logger.info("Adding line movement features...")
    df = add_line_movement_features(df)

    # --- ブックメーカーオッズ特徴量 ---
    logger.info("Adding bookmaker odds features...")
    df = add_odds_features(df)

    # --- 高度特徴量 ---
    from features.advanced_features import add_advanced_features
    logger.info("Adding advanced features...")
    df = add_advanced_features(df)

    # --- スポーツ固有特徴量 ---
    if sport_code == "baseball":
        from features.sport_specific.baseball import add_baseball_features
        logger.info("Adding baseball features...")
        df = add_baseball_features(df)

        # セイバーメトリクス (NPBのみ)
        from features.sabermetrics_features import add_sabermetrics_features
        logger.info("Adding sabermetrics features...")
        df = add_sabermetrics_features(df)
    elif sport_code == "soccer":
        from features.sport_specific.soccer import add_soccer_features
        logger.info("Adding soccer features...")
        df = add_soccer_features(df)
    elif sport_code == "basketball":
        from features.sport_specific.basketball import add_basketball_features
        logger.info("Adding basketball features...")
        df = add_basketball_features(df)

    # 特徴量カラムのリスト
    exclude_cols = {
        "match_id", "sport_code", "league_code", "date",
        "home_team_id", "away_team_id", "home_score", "away_score",
        "venue", "status",
        "home_pitcher_id", "away_pitcher_id",
        "handicap_team_id", "handicap_value", "result_type", "payout_rate",
        "target", "handicap_team_is_home",
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # NaN処理
    df[feature_cols] = df[feature_cols].fillna(0)

    logger.info(f"Built {len(feature_cols)} features for {len(df)} matches")
    return df, feature_cols


def build_training_data(
    sport_code: str,
    league_code: Optional[str] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    学習用の特徴量行列を構築する (エントリポイント)。

    Returns:
        (df_with_features, feature_column_names)
    """
    logger.info(f"Loading matches for {sport_code}/{league_code or 'all'}...")
    df = load_matches_df(sport_code=sport_code, league_code=league_code)

    if df.empty:
        logger.warning("No matches found")
        return df, []

    logger.info(f"Loaded {len(df)} matches, building features...")
    return build_features(df, sport_code=sport_code)
