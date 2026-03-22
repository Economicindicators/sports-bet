"""欠場情報特徴量: NBAの欠場データを予測時の特徴量として追加

- 学習時 (finished試合): 欠場の影響はスコアに反映済みなので 0 を返す
- 予測時 (scheduled試合): injury_scraperからリアルタイム欠場情報を取得し、
  チームごとの欠場影響度 (home_injury_impact / away_injury_impact) を計算
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# チーム名マッチング用キャッシュ
_injury_cache: Optional[Dict] = None


def _fetch_injury_map() -> Dict[str, float]:
    """Rotowireから欠場情報を取得し、チーム日本語名 → impact_score のマップを返す"""
    global _injury_cache
    if _injury_cache is not None:
        return _injury_cache

    try:
        from scraper.injury_scraper import scrape_nba_injuries
        games = scrape_nba_injuries()
    except Exception as e:
        logger.warning(f"Failed to fetch NBA injuries: {e}")
        _injury_cache = {}
        return _injury_cache

    impact_map: Dict[str, float] = {}
    for game in games:
        # home_team_ja → impact_score を蓄積
        if game.home_team_ja:
            impact_map[game.home_team_ja] = game.home_impact_score
        if game.away_team_ja:
            impact_map[game.away_team_ja] = game.away_impact_score
        # 英語名もマッピング (フォールバック用)
        if game.home_team:
            impact_map[game.home_team] = game.home_impact_score
        if game.away_team:
            impact_map[game.away_team] = game.away_impact_score

    _injury_cache = impact_map
    logger.info(f"Injury map loaded: {len(games)} games, {len(impact_map)} team entries")
    return _injury_cache


def clear_injury_cache():
    """キャッシュをクリア (新しい予測サイクルの前に呼ぶ)"""
    global _injury_cache
    _injury_cache = None


def add_injury_features(df: pd.DataFrame) -> pd.DataFrame:
    """欠場影響度の特徴量を追加

    追加カラム:
        - home_injury_impact: ホームチームの欠場影響度 (0=影響なし)
        - away_injury_impact: アウェイチームの欠場影響度
        - injury_impact_diff: ホーム - アウェイ の差 (正=ホーム不利)
    """
    has_scheduled = (df["status"] == "scheduled").any() if "status" in df.columns else False

    if not has_scheduled:
        # 学習時: 全てfinished → 欠場影響はスコアに反映済み
        df["home_injury_impact"] = 0.0
        df["away_injury_impact"] = 0.0
        df["injury_impact_diff"] = 0.0
        return df

    # scheduled試合がある → 欠場情報を取得
    injury_map = _fetch_injury_map()

    if not injury_map:
        logger.info("No injury data available, using zeros")
        df["home_injury_impact"] = 0.0
        df["away_injury_impact"] = 0.0
        df["injury_impact_diff"] = 0.0
        return df

    # チーム名の解決: team_id → team.name のマップを作成
    team_name_map = _build_team_name_map(df)

    home_impacts = []
    away_impacts = []

    for _, row in df.iterrows():
        if row.get("status") == "scheduled":
            home_name = team_name_map.get(row["home_team_id"], "")
            away_name = team_name_map.get(row["away_team_id"], "")
            h_impact = injury_map.get(home_name, 0.0)
            a_impact = injury_map.get(away_name, 0.0)
        else:
            # finished試合は0
            h_impact = 0.0
            a_impact = 0.0

        home_impacts.append(h_impact)
        away_impacts.append(a_impact)

    df["home_injury_impact"] = home_impacts
    df["away_injury_impact"] = away_impacts
    df["injury_impact_diff"] = df["home_injury_impact"] - df["away_injury_impact"]

    scheduled = df[df["status"] == "scheduled"]
    nonzero = ((scheduled["home_injury_impact"] != 0) | (scheduled["away_injury_impact"] != 0)).sum()
    logger.info(f"Injury features: {nonzero}/{len(scheduled)} scheduled games have injury data")

    return df


def _build_team_name_map(df: pd.DataFrame) -> Dict[int, str]:
    """team_id → team.name のマップを構築"""
    try:
        from database.models import Team, get_session
        session = get_session()

        team_ids = set(df["home_team_id"].unique()) | set(df["away_team_id"].unique())
        teams = session.query(Team).filter(Team.team_id.in_([int(tid) for tid in team_ids])).all()
        session.close()

        return {t.team_id: t.name for t in teams}
    except Exception as e:
        logger.warning(f"Failed to build team name map: {e}")
        return {}
