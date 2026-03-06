"""EV閾値フィルタ"""

from __future__ import annotations

import pandas as pd

from config.constants import MIN_EV_THRESHOLD


def filter_positive_ev(
    df: pd.DataFrame,
    ev_threshold: float = MIN_EV_THRESHOLD,
    ev_col: str = "handicap_ev",
) -> pd.DataFrame:
    """EV閾値以上のベットのみ抽出"""
    return df[df[ev_col] >= ev_threshold].copy()


def recommend_bets(
    df: pd.DataFrame,
    ev_threshold: float = MIN_EV_THRESHOLD,
    max_bets: int = 10,
) -> pd.DataFrame:
    """推奨ベットをEV降順で返す"""
    filtered = filter_positive_ev(df, ev_threshold)
    return filtered.nlargest(max_bets, "handicap_ev")
