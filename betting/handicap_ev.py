"""ハンデベットEV計算 — ハンデの森テーブル準拠

ハンデの種類(整数/半/ボーダー)に応じて、正しい配当構造でEVを計算する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from betting.handicap_resolver import (
    parse_handicap_display,
    _normalize_border,
    PAYOUT_TABLE,
    BORDER_TO_WIN,
    BORDER_TO_LOSS,
)


# ==================== スコア差分布の推定 ====================

# 各スポーツのハンデ有利チーム勝利時のスコア差分布 (実データベース)
# P(diff=d | ハンデ有利チーム勝利)
SCORE_DIFF_DIST = {
    "baseball": {
        # 野球: 1点差が最多、大差は少ない
        1: 0.35, 2: 0.25, 3: 0.18, 4: 0.10, 5: 0.06, 6: 0.03, 7: 0.02, 8: 0.01,
    },
    "soccer": {
        # サッカー: 1点差が圧倒的
        1: 0.55, 2: 0.28, 3: 0.11, 4: 0.04, 5: 0.02,
    },
    "basketball": {
        # バスケ: 幅広い得点差 (1-35を連続で定義)
        1: 0.035, 2: 0.035, 3: 0.045, 4: 0.045, 5: 0.055, 6: 0.055, 7: 0.055,
        8: 0.055, 9: 0.055, 10: 0.055, 11: 0.045, 12: 0.045, 13: 0.045,
        14: 0.038, 15: 0.038, 16: 0.035, 17: 0.030, 18: 0.028, 19: 0.025,
        20: 0.022, 21: 0.018, 22: 0.016, 23: 0.014, 24: 0.012, 25: 0.010,
        26: 0.009, 27: 0.008, 28: 0.007, 29: 0.006, 30: 0.005,
        31: 0.004, 32: 0.003, 33: 0.003, 34: 0.002, 35: 0.002,
    },
}

# 引き分けの確率
DRAW_PROB = {
    "baseball": 0.00,   # 野球は引き分けほぼなし (延長)
    "soccer": 0.25,     # サッカーは引き分け多い
    "basketball": 0.00, # バスケは引き分けなし (延長)
}


def _get_score_diff_probs(
    win_prob: float,
    sport: str = "baseball",
) -> dict:
    """ハンデ有利チームの勝率から各得点差の確率分布を生成。

    Returns:
        {diff: probability} (diffは負=負け、0=引き分け、正=勝ち)
    """
    dist = SCORE_DIFF_DIST.get(sport, SCORE_DIFF_DIST["baseball"])
    draw_rate = DRAW_PROB.get(sport, 0.0)

    # 引き分け確率
    p_draw = draw_rate  # スポーツ固有の引き分け率 (modelのwin_probとは独立)

    # 残りを勝ち/負けに分配
    p_win = win_prob * (1 - p_draw)
    p_lose = (1 - win_prob) * (1 - p_draw)

    probs = {}

    # 引き分け
    probs[0] = p_draw

    # 勝ちの場合のスコア差分布
    for diff, rate in dist.items():
        probs[diff] = p_win * rate

    # 負けの場合のスコア差分布 (ミラー)
    for diff, rate in dist.items():
        probs[-diff] = p_lose * rate

    return probs


# ==================== EV計算 ====================

def calculate_handicap_ev(
    prob_favorable: float,
    handicap_value: float = 0,
    handicap_display: str = "",
    sport: str = "baseball",
) -> float:
    """ハンデの種類に応じた正確なEV計算。

    ハンデの森テーブルに基づき、各スコア差での配当を正しく反映。

    Args:
        prob_favorable: ハンデ有利チームの勝率 (0-1)
        handicap_value: ハンデ値 (float)
        handicap_display: ハンデ表示文字列
        sport: スポーツ

    Returns:
        EV値 (0以上がプラス期待値)
    """
    base, is_half, border = parse_handicap_display(handicap_display, handicap_value)
    border = _normalize_border(border)
    diff_probs = _get_score_diff_probs(prob_favorable, sport)

    ev = 0.0

    for diff, prob in diff_probs.items():
        payout = _get_payout(diff, base, is_half, border)
        ev += prob * payout

    return ev - 1.0


def _get_payout(diff: int, base: int, is_half: bool, border: int) -> float:
    """スコア差・ハンデ構造から配当を返す。"""
    if is_half:
        threshold = base + 1
        if diff < threshold:
            return 0.0  # 丸負け
        elif diff == threshold:
            if border == 0:
                return 2.0  # 丸勝ち
            return PAYOUT_TABLE[BORDER_TO_WIN[border]]
        else:
            return 2.0  # 丸勝ち
    else:
        if base == 0:
            if diff < 0:
                return 0.0  # 丸負け
            elif diff == 0:
                if border == 0:
                    return 1.0  # 勝負無し
                return PAYOUT_TABLE[BORDER_TO_LOSS[border]]
            elif diff == 1:
                if border == 0:
                    return 2.0  # 丸勝ち
                return PAYOUT_TABLE[BORDER_TO_WIN[border]]
            else:
                return 2.0  # 丸勝ち
        else:
            if diff < base:
                return 0.0  # 丸負け
            elif diff == base:
                if border == 0:
                    return 1.0  # 勝負無し
                return PAYOUT_TABLE[BORDER_TO_LOSS[border]]
            else:
                return 2.0  # 丸勝ち


def calculate_handicap_ev_simple(
    prob_favorable: float,
    prob_push: float = 0.02,
) -> float:
    """旧互換: シンプルなEV計算（handicap_display不明時のフォールバック）"""
    avg_fav = 1.733
    avg_unfav = 0.5
    prob_unfavorable = 1.0 - prob_favorable - prob_push
    return (
        prob_favorable * avg_fav
        + prob_push * 1.0
        + prob_unfavorable * avg_unfav
        - 1.0
    )


def home_prob_to_handicap_prob(
    home_win_prob: float,
    handicap_team_is_home: bool,
    handicap_value: float = 0,
    sport: str = "baseball",
) -> float:
    """ホーム勝率 → ハンデ有利チームの勝率に変換（方向転換のみ）。

    注意: この関数はモデルの「ホーム勝率」をハンデ有利チーム視点に
    変換するだけ。ハンデ値に基づくカバー確率の調整は
    calculate_handicap_ev内の_get_score_diff_probsで行う。
    """
    if handicap_team_is_home:
        return home_win_prob
    else:
        return 1.0 - home_win_prob


def calculate_contrarian_ev(
    prob_favorable: float,
    handicap_value: float = 0,
    handicap_display: str = "",
    sport: str = "baseball",
) -> float:
    """逆張りEVを計算。prob_favorableを反転して計算。"""
    return calculate_handicap_ev(
        1.0 - prob_favorable,
        handicap_value,
        handicap_display,
        sport,
    )


def add_ev_to_predictions(df: pd.DataFrame, prob_col: str = "pred_prob") -> pd.DataFrame:
    """DataFrameにEVカラムを追加"""
    df["handicap_ev"] = df.apply(
        lambda r: calculate_handicap_ev(
            r[prob_col],
            r.get("handicap_value", 0),
            r.get("handicap_display", ""),
            r.get("sport", "baseball"),
        ),
        axis=1,
    )
    return df
