"""ブックメーカーオッズ特徴量 — consensus probability, odds dispersion, line value"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from database.models import BookmakerOdds, get_session

logger = logging.getLogger(__name__)

# シャープブックメーカー (プライスが正確)
SHARP_BOOKMAKERS = {"pinnacle", "betfair_ex"}
# ソフトブックメーカー (レクリエーション向け)
SOFT_BOOKMAKERS = {"bet365", "williamhill", "1xbet", "unibet", "bwin"}


def add_odds_features(df: pd.DataFrame, session: Session | None = None) -> pd.DataFrame:
    """
    ブックメーカーオッズから特徴量を追加。

    追加される特徴量:
    - consensus_home_prob: 全ブックメーカー平均のホーム勝利確率
    - consensus_away_prob: 全ブックメーカー平均のアウェイ勝利確率
    - sharp_home_prob: シャープ (Pinnacle等) のホーム勝利確率
    - odds_dispersion: ブックメーカー間のオッズ分散 (意見の不一致度)
    - sharp_soft_gap: シャープ vs ソフトの確率差 (sharp money indicator)
    - implied_overround: 暗黙のマージン (vig)
    - odds_count: オッズ提供ブックメーカー数
    - best_home_odds: 最良ホームオッズ
    - best_away_odds: 最良アウェイオッズ
    - odds_vs_handicap: オッズ示唆確率 vs ハンデの乖離
    """
    if df.empty:
        return df

    sess = session or get_session()
    own_session = session is None

    match_ids = df["match_id"].tolist()

    # 一括取得
    all_odds = (
        sess.query(BookmakerOdds)
        .filter(BookmakerOdds.match_id.in_(match_ids))
        .all()
    )

    # match_id → list of odds
    odds_map: dict[int, list[BookmakerOdds]] = {}
    for o in all_odds:
        odds_map.setdefault(o.match_id, []).append(o)

    # 特徴量配列
    feat = {
        "consensus_home_prob": [],
        "consensus_away_prob": [],
        "sharp_home_prob": [],
        "odds_dispersion": [],
        "sharp_soft_gap": [],
        "implied_overround": [],
        "odds_count": [],
        "best_home_odds": [],
        "best_away_odds": [],
        "odds_vs_handicap": [],
    }

    for _, row in df.iterrows():
        mid = int(row["match_id"])
        odds_list = odds_map.get(mid, [])

        if not odds_list:
            for k in feat:
                feat[k].append(0.0)
            continue

        # h2h odds を持つものだけ
        h2h = [o for o in odds_list if o.home_odds and o.away_odds]
        if not h2h:
            for k in feat:
                feat[k].append(0.0)
            continue

        # 確率変換 (1/odds)
        home_probs = [1.0 / o.home_odds for o in h2h]
        away_probs = [1.0 / o.away_odds for o in h2h]

        # overround除去した正規化確率
        avg_home_raw = np.mean(home_probs)
        avg_away_raw = np.mean(away_probs)
        overround = avg_home_raw + avg_away_raw
        if any(o.draw_odds for o in h2h):
            draw_probs = [1.0 / o.draw_odds for o in h2h if o.draw_odds]
            overround += np.mean(draw_probs) if draw_probs else 0

        # Consensus probability (正規化)
        total = avg_home_raw + avg_away_raw
        consensus_home = avg_home_raw / total if total > 0 else 0.5
        consensus_away = avg_away_raw / total if total > 0 else 0.5

        # Sharp bookmaker probability
        sharp = [o for o in h2h if o.bookmaker in SHARP_BOOKMAKERS]
        if sharp:
            sharp_home_raw = np.mean([1.0 / o.home_odds for o in sharp])
            sharp_away_raw = np.mean([1.0 / o.away_odds for o in sharp])
            sharp_total = sharp_home_raw + sharp_away_raw
            sharp_home_prob = sharp_home_raw / sharp_total if sharp_total > 0 else 0.5
        else:
            sharp_home_prob = consensus_home

        # Soft bookmaker probability
        soft = [o for o in h2h if o.bookmaker in SOFT_BOOKMAKERS]
        if soft:
            soft_home_raw = np.mean([1.0 / o.home_odds for o in soft])
            soft_away_raw = np.mean([1.0 / o.away_odds for o in soft])
            soft_total = soft_home_raw + soft_away_raw
            soft_home_prob = soft_home_raw / soft_total if soft_total > 0 else 0.5
        else:
            soft_home_prob = consensus_home

        # Odds dispersion (意見の不一致)
        dispersion = np.std(home_probs) if len(home_probs) > 1 else 0.0

        # Sharp vs soft gap (positive = sharp thinks home is more likely than soft does)
        sharp_soft_diff = sharp_home_prob - soft_home_prob

        # Best odds
        best_home = max(o.home_odds for o in h2h)
        best_away = max(o.away_odds for o in h2h)

        # Odds vs handicap (ハンデチームの implied prob vs 0.5)
        handicap_is_home = int(row.get("handicap_team_is_home", 0))
        if handicap_is_home:
            odds_vs_handi = consensus_home - 0.5
        else:
            odds_vs_handi = consensus_away - 0.5

        feat["consensus_home_prob"].append(consensus_home)
        feat["consensus_away_prob"].append(consensus_away)
        feat["sharp_home_prob"].append(sharp_home_prob)
        feat["odds_dispersion"].append(dispersion)
        feat["sharp_soft_gap"].append(sharp_soft_diff)
        feat["implied_overround"].append(overround)
        feat["odds_count"].append(float(len(h2h)))
        feat["best_home_odds"].append(best_home)
        feat["best_away_odds"].append(best_away)
        feat["odds_vs_handicap"].append(odds_vs_handi)

    for k, v in feat.items():
        df[k] = v

    if own_session:
        sess.close()

    n_with_odds = sum(1 for x in feat["odds_count"] if x > 0)
    logger.info(f"Added odds features ({n_with_odds}/{len(df)} matches had bookmaker odds)")
    return df
