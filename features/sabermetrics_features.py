"""セイバーメトリクス特徴量 — 1point02.jp (DELTA Inc) データから算出"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from database.models import Team, get_session

logger = logging.getLogger(__name__)

# キャッシュ: {season: {team_name: stats_dict}}
_SABER_CACHE: dict[int, dict[str, dict]] = {}


def _load_sabermetrics(season: int) -> dict[str, dict]:
    """DELTAからセイバーメトリクスを取得してキャッシュ"""
    if season in _SABER_CACHE:
        return _SABER_CACHE[season]

    try:
        from scraper.delta_scraper import DeltaScraper
        scraper = DeltaScraper(use_cache=True)
        agg = scraper.get_team_aggregate(season)
        _SABER_CACHE[season] = agg
        logger.info(f"Loaded sabermetrics for {season}: {len(agg)} teams")
        return agg
    except Exception as e:
        logger.warning(f"Failed to load sabermetrics for {season}: {e}")
        return {}


def add_sabermetrics_features(
    df: pd.DataFrame, session: Session | None = None
) -> pd.DataFrame:
    """
    NPBチームのセイバーメトリクス特徴量を追加。

    追加される特徴量:
    - home_team_ops / away_team_ops: チームOPS
    - team_ops_diff: OPS差
    - home_team_rpg / away_team_rpg: チーム1試合あたり得点
    - home_rotation_era / away_rotation_era: 先発ローテERA
    - home_rotation_fip / away_rotation_fip: 先発ローテFIP
    - home_rotation_whip / away_rotation_whip: 先発ローテWHIP
    - home_rotation_k9 / away_rotation_k9: 先発ローテ奪三振率
    - rotation_era_diff / rotation_fip_diff: 投手力差
    - offense_vs_pitching: 攻撃力(OPS) vs 相手投手力(FIP)の乖離
    """
    if df.empty:
        return df

    sess = session or get_session()
    own_session = session is None

    # team_id → team_name マッピング
    team_ids = set(df["home_team_id"].tolist() + df["away_team_id"].tolist())
    teams = sess.query(Team).filter(Team.team_id.in_(team_ids)).all()
    id_to_name = {t.team_id: t.name for t in teams}
    id_to_league = {t.team_id: t.league_code for t in teams}

    # 年ごとのセイバーデータを取得
    seasons_needed = set()
    for _, row in df.iterrows():
        d = row["date"]
        year = d.year if hasattr(d, "year") else int(str(d)[:4])
        seasons_needed.add(year)

    saber_by_season = {}
    for s in seasons_needed:
        saber_by_season[s] = _load_sabermetrics(s)

    # 特徴量配列
    features = {
        "home_team_ops": [],
        "away_team_ops": [],
        "team_ops_diff": [],
        "home_team_rpg": [],
        "away_team_rpg": [],
        "home_rotation_era": [],
        "away_rotation_era": [],
        "home_rotation_fip": [],
        "away_rotation_fip": [],
        "home_rotation_whip": [],
        "away_rotation_whip": [],
        "home_rotation_k9": [],
        "away_rotation_k9": [],
        "rotation_era_diff": [],
        "rotation_fip_diff": [],
        "offense_vs_pitching": [],
    }

    for _, row in df.iterrows():
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        home_name = id_to_name.get(home_id, "")
        away_name = id_to_name.get(away_id, "")
        home_league = id_to_league.get(home_id, "")
        away_league = id_to_league.get(away_id, "")

        # NPBのみセイバーメトリクス適用
        if home_league not in ("npb",) and away_league not in ("npb",):
            for k in features:
                features[k].append(0.0)
            continue

        d = row["date"]
        year = d.year if hasattr(d, "year") else int(str(d)[:4])
        saber = saber_by_season.get(year, {})

        home_stats = saber.get(home_name, {})
        away_stats = saber.get(away_name, {})

        h_ops = home_stats.get("team_ops", 0)
        a_ops = away_stats.get("team_ops", 0)
        h_rpg = home_stats.get("team_runs_per_game", 0)
        a_rpg = away_stats.get("team_runs_per_game", 0)
        h_r_era = home_stats.get("rotation_era", 0)
        a_r_era = away_stats.get("rotation_era", 0)
        h_r_fip = home_stats.get("rotation_fip", 0)
        a_r_fip = away_stats.get("rotation_fip", 0)
        h_r_whip = home_stats.get("rotation_whip", 0)
        a_r_whip = away_stats.get("rotation_whip", 0)
        h_r_k9 = home_stats.get("rotation_k9", 0)
        a_r_k9 = away_stats.get("rotation_k9", 0)

        features["home_team_ops"].append(h_ops)
        features["away_team_ops"].append(a_ops)
        features["team_ops_diff"].append(h_ops - a_ops)
        features["home_team_rpg"].append(h_rpg)
        features["away_team_rpg"].append(a_rpg)
        features["home_rotation_era"].append(h_r_era)
        features["away_rotation_era"].append(a_r_era)
        features["home_rotation_fip"].append(h_r_fip)
        features["away_rotation_fip"].append(a_r_fip)
        features["home_rotation_whip"].append(h_r_whip)
        features["away_rotation_whip"].append(a_r_whip)
        features["home_rotation_k9"].append(h_r_k9)
        features["away_rotation_k9"].append(a_r_k9)
        features["rotation_era_diff"].append(h_r_era - a_r_era if h_r_era and a_r_era else 0)
        features["rotation_fip_diff"].append(h_r_fip - a_r_fip if h_r_fip and a_r_fip else 0)

        # 攻撃力 vs 相手投手力: 自チームOPSが高く相手FIPも高い(=相手投手弱い)なら有利
        ovp = 0.0
        if h_ops > 0 and a_r_fip > 0:
            ovp = h_ops * a_r_fip  # 高いほどホーム攻撃有利
        features["offense_vs_pitching"].append(ovp)

    for k, v in features.items():
        df[k] = v

    if own_session:
        sess.close()

    n_with_data = sum(1 for x in features["home_team_ops"] if x > 0)
    logger.info(f"Added sabermetrics features ({n_with_data}/{len(df)} matches had data)")
    return df
