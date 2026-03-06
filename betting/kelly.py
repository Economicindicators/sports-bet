"""Kelly基準 (keiba-aiベース)"""

from __future__ import annotations

from config.constants import KELLY_FRACTION, MAX_BET_FRACTION


def kelly_fraction(
    prob: float,
    avg_payout: float = 1.7,
    fraction: float = KELLY_FRACTION,
) -> float:
    """
    Kelly基準によるベットサイズを計算。

    Full Kelly: f* = (p*b - q) / b
    Fractional Kelly: f* × fraction

    Args:
        prob: 勝利確率
        avg_payout: 平均ペイアウト倍率
        fraction: Kelly分率 (0.25 = 1/4 Kelly)

    Returns:
        ベット比率 [0, MAX_BET_FRACTION]
    """
    b = avg_payout - 1.0  # net odds
    if b <= 0:
        return 0.0

    q = 1.0 - prob
    f_star = (prob * b - q) / b

    if f_star <= 0:
        return 0.0

    bet = f_star * fraction
    return min(bet, MAX_BET_FRACTION)


def calculate_bet_size(
    prob: float,
    bankroll: float,
    avg_payout: float = 1.7,
    fraction: float = KELLY_FRACTION,
) -> float:
    """ベット金額を計算"""
    f = kelly_fraction(prob, avg_payout, fraction)
    return round(bankroll * f)
