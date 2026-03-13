"""ハンデベットEV計算

モデルは「ホームチーム勝利確率」を予測する。
ハンデ有利チームがホームかアウェイかに応じて確率を変換し、
ハンデ値による補正を加えてEVを計算する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ハンデ結果区分別の平均ペイアウト
FAVORABLE_PAYOUTS = {
    "丸勝ち": 2.0,
    "7分勝ち": 1.7,
    "5分勝ち": 1.5,
}

UNFAVORABLE_PAYOUTS = {
    "3分勝ち": 1.3,
    "3分負け": 0.7,
    "5分負け": 0.5,
    "7分負け": 0.3,
    "丸負け": 0.0,
}

AVG_FAVORABLE_PAYOUT = np.mean(list(FAVORABLE_PAYOUTS.values()))  # 1.733
AVG_UNFAVORABLE_PAYOUT = np.mean(list(UNFAVORABLE_PAYOUTS.values()))  # 0.5

# ハンデ値帯ごとの「勝った場合のfavorable率」(実データから算出)
# = P(5分勝ち以上 | ハンデ有利チーム勝利)
WIN_TO_FAVORABLE_RATE = {
    "baseball": {
        0.0: 0.99, 0.5: 0.80, 1.0: 0.72, 1.5: 0.70, 2.0: 0.58, 3.0: 0.57, 5.0: 0.42,
    },
    "soccer": {
        0.0: 0.81, 0.5: 0.87, 1.0: 0.55, 1.5: 0.59, 2.0: 0.45, 3.0: 0.48, 5.0: 0.40,
    },
    "basketball": {
        0.0: 1.00, 1.0: 0.93, 2.0: 0.93, 3.0: 0.85, 4.0: 0.79,
        5.0: 0.71, 6.0: 0.66, 7.0: 0.68, 8.0: 0.64, 10.0: 0.61, 12.0: 0.58, 15.0: 0.60,
    },
}


def _get_win_to_favorable(sport: str, handicap_value: float) -> float:
    """ハンデ値からfavorable補正率を線形補間で取得"""
    rates = WIN_TO_FAVORABLE_RATE.get(sport, WIN_TO_FAVORABLE_RATE["baseball"])
    thresholds = sorted(rates.keys())
    abs_hv = abs(handicap_value)

    if abs_hv <= thresholds[0]:
        return rates[thresholds[0]]
    if abs_hv >= thresholds[-1]:
        return rates[thresholds[-1]]

    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        if lo <= abs_hv <= hi:
            t = (abs_hv - lo) / (hi - lo)
            return rates[lo] * (1 - t) + rates[hi] * t

    return 0.5


def home_prob_to_handicap_prob(
    home_win_prob: float,
    handicap_team_is_home: bool,
    handicap_value: float = 0,
    sport: str = "baseball",
) -> float:
    """ホーム勝率 → ハンデ有利チームのfavorable確率に変換。

    1. ホーム勝率をハンデ有利チーム勝率に変換
    2. ハンデ値による補正（勝っても大差でないとfavorableにならない）

    Args:
        home_win_prob: モデル予測のホーム勝率
        handicap_team_is_home: ハンデ有利チームがホームか
        handicap_value: ハンデ値
        sport: スポーツコード

    Returns:
        ハンデ有利チームがfavorable(5分勝ち以上)になる確率
    """
    # Step 1: ホーム勝率 → ハンデ有利チーム勝率
    if handicap_team_is_home:
        win_prob = home_win_prob
    else:
        win_prob = 1.0 - home_win_prob

    # Step 2: 勝った場合のfavorable率で補正
    fav_rate = _get_win_to_favorable(sport, handicap_value)
    favorable_prob = win_prob * fav_rate

    return favorable_prob


def calculate_handicap_ev(
    prob_favorable: float,
    prob_push: float = 0.02,
    border_pct: float = 50,
) -> float:
    """
    ハンデベットのEVを計算。

    EV = P(fav) × avg_fav_payout + P(push) × push_payout + P(unfav) × avg_unfav_payout - 1.0

    Args:
        prob_favorable: ハンデ有利チームのfavorable確率
        prob_push: 勝負無しの確率 (デフォルト2%)
        border_pct: ボーダー%

    Returns:
        EV値
    """
    push_payout = border_pct / 100.0
    if border_pct == 0 or border_pct == 50:
        push_payout = 1.0

    prob_unfavorable = 1.0 - prob_favorable - prob_push
    ev = (
        prob_favorable * AVG_FAVORABLE_PAYOUT
        + prob_push * push_payout
        + prob_unfavorable * AVG_UNFAVORABLE_PAYOUT
        - 1.0
    )
    return ev


def calculate_contrarian_ev(
    prob_favorable: float,
    prob_push: float = 0.02,
    border_pct: float = 50,
) -> float:
    """
    逆張りEVを計算。

    Args:
        prob_favorable: ハンデ有利チームのfavorable確率
        prob_push: 勝負無しの確率 (デフォルト2%)
        border_pct: ボーダー%

    Returns:
        逆張りEV値
    """
    push_payout = border_pct / 100.0
    if border_pct == 0 or border_pct == 50:
        push_payout = 1.0

    prob_unfavorable = 1.0 - prob_favorable - prob_push
    ev = (
        prob_unfavorable * AVG_FAVORABLE_PAYOUT
        + prob_push * push_payout
        + prob_favorable * AVG_UNFAVORABLE_PAYOUT
        - 1.0
    )
    return ev


def add_ev_to_predictions(df: pd.DataFrame, prob_col: str = "pred_prob") -> pd.DataFrame:
    """DataFrameにEVカラムを追加"""
    df["handicap_ev"] = df[prob_col].apply(calculate_handicap_ev)
    return df
