"""ハンデ判定ロジック: 得点差 + ハンデ → 9段階結果判定"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HandicapResult:
    """ハンデ判定結果"""

    adjusted_margin: float
    result_type: str
    payout_rate: float
    is_favorable: bool  # 5分勝ち以上か


def resolve_handicap(
    fav_score: int,
    underdog_score: int,
    handicap_value: float,
) -> HandicapResult:
    """
    ハンデ込み勝敗を判定する。

    Args:
        fav_score: ハンデを背負うチーム（有利チーム）の得点
        underdog_score: 不利チームの得点
        handicap_value: ハンデ値（正の数、有利チームが背負う点数）

    Returns:
        HandicapResult: 判定結果
    """
    adjusted = (fav_score - underdog_score) - handicap_value

    if adjusted >= 2.0:
        return HandicapResult(adjusted, "丸勝ち", 2.0, True)
    elif adjusted >= 1.0:
        return HandicapResult(adjusted, "7分勝ち", 1.7, True)
    elif adjusted >= 0.5:
        return HandicapResult(adjusted, "5分勝ち", 1.5, True)
    elif adjusted > 0:
        return HandicapResult(adjusted, "3分勝ち", 1.3, False)
    elif adjusted == 0:
        return HandicapResult(adjusted, "勝負無し", 1.0, False)
    elif adjusted > -0.5:
        return HandicapResult(adjusted, "3分負け", 0.7, False)
    elif adjusted > -1.0:
        return HandicapResult(adjusted, "5分負け", 0.5, False)
    elif adjusted > -2.0:
        return HandicapResult(adjusted, "7分負け", 0.3, False)
    else:
        return HandicapResult(adjusted, "丸負け", 0.0, False)


def calculate_ev(
    prob_favorable: float,
    prob_push: float = 0.0,
    avg_favorable_payout: float = 1.7,
    avg_unfavorable_payout: float = 0.4,
) -> float:
    """
    ハンデベットのEVを計算する。

    EV = P(favorable) × avg_favorable_payout
       + P(push) × 1.0
       + P(unfavorable) × avg_unfavorable_payout
       - 1.0

    Args:
        prob_favorable: 5分勝ち以上の確率
        prob_push: 勝負無しの確率
        avg_favorable_payout: favorable時の平均ペイアウト
        avg_unfavorable_payout: unfavorable時の平均ペイアウト

    Returns:
        EV値 (0以上がプラス期待値)
    """
    prob_unfavorable = 1.0 - prob_favorable - prob_push
    ev = (
        prob_favorable * avg_favorable_payout
        + prob_push * 1.0
        + prob_unfavorable * avg_unfavorable_payout
        - 1.0
    )
    return ev
