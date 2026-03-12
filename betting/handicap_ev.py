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
    border_pct: float = 50,
) -> float:
    """
    ハンデベットのEVを計算。

    EV = P(fav) × avg_fav_payout + P(push) × push_payout + P(unfav) × avg_unfav_payout - 1.0

    Args:
        prob_favorable: 5分勝ち以上の確率 (モデル予測値)
        prob_push: 勝負無しの確率 (デフォルト2%)
        border_pct: ボーダー% (0半3→30, 0半5→50, 0半7→70)
                    ボーダー時に有利側が受け取る割合

    Returns:
        EV値
    """
    # ボーダー時のペイアウト: border_pct% が有利側に返金
    push_payout = border_pct / 100.0  # 30→0.3, 50→0.5, 70→0.7, ボーダーなし→1.0
    if border_pct == 0 or border_pct == 50:
        push_payout = 1.0  # 整数ハンデ(0/0)や半球(0半5)は引分=全額返金

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
    ハンデ有利チームの確率が低い場合、逆側（不利チーム側）に賭けるEV。

    考え方: 不利チームが勝つ = 逆張り側の "favorable" outcome
    EV = P(unfav) × avg_fav_payout + P(push) × push_payout + P(fav) × avg_unfav_payout - 1.0

    Args:
        prob_favorable: ハンデ有利チームの5分勝ち以上確率 (モデル予測値)
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
