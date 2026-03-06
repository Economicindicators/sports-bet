"""ハンデベットEV計算"""

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


def calculate_handicap_ev(
    prob_favorable: float,
    prob_push: float = 0.02,
) -> float:
    """
    ハンデベットのEVを計算。

    EV = P(fav) × avg_fav_payout + P(push) × 1.0 + P(unfav) × avg_unfav_payout - 1.0

    Args:
        prob_favorable: 5分勝ち以上の確率 (モデル予測値)
        prob_push: 勝負無しの確率 (デフォルト2%)

    Returns:
        EV値
    """
    prob_unfavorable = 1.0 - prob_favorable - prob_push
    ev = (
        prob_favorable * AVG_FAVORABLE_PAYOUT
        + prob_push * 1.0
        + prob_unfavorable * AVG_UNFAVORABLE_PAYOUT
        - 1.0
    )
    return ev


def add_ev_to_predictions(df: pd.DataFrame, prob_col: str = "pred_prob") -> pd.DataFrame:
    """DataFrameにEVカラムを追加"""
    df["handicap_ev"] = df[prob_col].apply(calculate_handicap_ev)
    return df
