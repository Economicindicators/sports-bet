"""ハンデ判定ロジック — ハンデの森テーブル準拠

ハンデの森のハンデ表:
https://handenomori.com/handi-table/

ハンデには2つの軸がある:
1. ベース (整数 or 半): 判定の基準となる得点差
2. ボーダー (0, 3, 5, 7): 境界での配当割合

【整数ベース X.B】(例: 0.5, 1.3, 2)
  - diff < X: 丸負け
  - diff = X: B分負け (B=0なら勝負無し)
  - diff = X+1: X=0の場合のみ (10-B)分勝ち、X≥1は丸勝ち
  - diff > X+1: 丸勝ち

【半ベース X半B】(例: 1半, 1半5)
  - diff ≤ X: 丸負け
  - diff = X+1: (10-B)分勝ち (B=0なら丸勝ち)
  - diff > X+1: 丸勝ち

配当:
  丸勝ち=2.0, 7分勝ち=1.7, 5分勝ち=1.5, 3分勝ち=1.3
  勝負無し=1.0
  3分負け=0.7, 5分負け=0.5, 7分負け=0.3, 丸負け=0.0
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import re


@dataclass
class HandicapResult:
    """ハンデ判定結果"""
    result_type: str
    payout_rate: float
    is_favorable: bool  # 5分勝ち以上か


# 配当テーブル
PAYOUT_TABLE = {
    "丸勝ち": 2.0,
    "7分勝ち": 1.7,
    "5分勝ち": 1.5,
    "3分勝ち": 1.3,
    "勝負無し": 1.0,
    "3分負け": 0.7,
    "5分負け": 0.5,
    "7分負け": 0.3,
    "丸負け": 0.0,
}

BORDER_TO_LOSS = {0: "勝負無し", 3: "3分負け", 5: "5分負け", 7: "7分負け"}
BORDER_TO_WIN = {0: "丸勝ち", 3: "7分勝ち", 5: "5分勝ち", 7: "3分勝ち"}
VALID_BORDERS = {0, 3, 5, 7}


def _normalize_border(b: int) -> int:
    """不正なボーダー値を最も近い有効値に丸める"""
    if b in VALID_BORDERS:
        return b
    return min(VALID_BORDERS, key=lambda x: abs(x - b))


def parse_handicap_display(display: str, fallback_value: float = 0) -> Tuple[int, bool, int]:
    """ハンデ表示文字列を (ベース整数, 半か, ボーダー) に分解。

    Args:
        display: ハンデ表示 (例: "1半5", "0.7", "2", "-1半3")
        fallback_value: displayが空の場合のフォールバック値

    Returns:
        (base_int, is_half, border)
        例: "1半5" → (1, True, 5)
            "0.7"  → (0, False, 7)
            "2"    → (2, False, 0)
    """
    if not display or display.strip() == "":
        # フォールバック: float値から推定
        return _parse_from_float(abs(fallback_value))

    s = display.strip().lstrip("-")

    # "1半5", "1半3", "0半5", "1半"
    m = re.match(r"^(\d+)半(\d?)$", s)
    if m:
        base = int(m.group(1))
        border = int(m.group(2)) if m.group(2) else 0
        return (base, True, border)

    # "1.3", "0.5", "1.7", "2.0"
    m = re.match(r"^(\d+)\.(\d)$", s)
    if m:
        base = int(m.group(1))
        decimal = int(m.group(2))
        return (base, False, _normalize_border(decimal))

    # "1", "2", "13" (= -13 from NBA etc)
    m = re.match(r"^(\d+)$", s)
    if m:
        val = int(m.group(1))
        return (val, False, 0)

    # パース失敗 → フォールバック
    return _parse_from_float(abs(fallback_value))


def _parse_from_float(value: float) -> Tuple[int, bool, int]:
    """float値からハンデ構造を推定（displayがない場合のフォールバック）。

    バスケ/サッカーなど半ハンデが主流のスポーツ用。
    整数 → (X, False, 0)、.5 → (X, True, 0)、その他 → 近似。
    """
    value = abs(value)
    base = int(value)
    frac = round(value - base, 2)

    if frac == 0:
        return (base, False, 0)
    elif frac == 0.5:
        return (base, True, 0)
    elif frac == 0.25:
        # クォーターライン → 半+ボーダー5に近似
        return (base, True, 5)
    elif frac == 0.75:
        # 0.75 → 整数+ボーダー5に近似 (次の整数に近い)
        return (base, False, 5)
    elif frac in (0.3, 0.03):
        return (base, False, 3)
    elif frac in (0.7, 0.07):
        return (base, False, 7)
    else:
        # 最も近いパターンに丸め
        if frac < 0.25:
            return (base, False, 3)
        elif frac < 0.5:
            return (base, True, 5)
        elif frac < 0.75:
            return (base, False, 5)
        else:
            return (base, False, 7)


def resolve_handicap(
    fav_score: int,
    underdog_score: int,
    handicap_value: float,
    handicap_display: str = "",
) -> HandicapResult:
    """ハンデ込み勝敗を判定する（ハンデの森テーブル準拠）。

    Args:
        fav_score: ハンデを背負うチーム（有利チーム）の得点
        underdog_score: 不利チームの得点
        handicap_value: ハンデ値
        handicap_display: ハンデ表示文字列 ("1半5"等)

    Returns:
        HandicapResult: 判定結果
    """
    diff = fav_score - underdog_score
    base, is_half, border = parse_handicap_display(handicap_display, handicap_value)

    if is_half:
        # 半ベース: 閾値は base+1
        threshold = base + 1
        if diff < threshold:
            return HandicapResult("丸負け", 0.0, False)
        elif diff == threshold:
            if border == 0:
                return HandicapResult("丸勝ち", 2.0, True)
            else:
                result_type = BORDER_TO_WIN[border]
                payout = PAYOUT_TABLE[result_type]
                return HandicapResult(result_type, payout, payout >= 1.5)
        else:
            return HandicapResult("丸勝ち", 2.0, True)
    else:
        # 整数ベース: 閾値は base
        if base == 0:
            # base=0 は特殊: diff=0 と diff=1 の両方が部分配当
            if diff < 0:
                return HandicapResult("丸負け", 0.0, False)
            elif diff == 0:
                if border == 0:
                    return HandicapResult("勝負無し", 1.0, False)
                else:
                    result_type = BORDER_TO_LOSS[border]
                    payout = PAYOUT_TABLE[result_type]
                    return HandicapResult(result_type, payout, False)
            elif diff == 1:
                if border == 0:
                    return HandicapResult("丸勝ち", 2.0, True)
                else:
                    result_type = BORDER_TO_WIN[border]
                    payout = PAYOUT_TABLE[result_type]
                    return HandicapResult(result_type, payout, payout >= 1.5)
            else:
                return HandicapResult("丸勝ち", 2.0, True)
        else:
            # base≥1: diff=base のみ部分配当、diff≥base+1 は丸勝ち
            if diff < base:
                return HandicapResult("丸負け", 0.0, False)
            elif diff == base:
                if border == 0:
                    return HandicapResult("勝負無し", 1.0, False)
                else:
                    result_type = BORDER_TO_LOSS[border]
                    payout = PAYOUT_TABLE[result_type]
                    return HandicapResult(result_type, payout, False)
            else:
                return HandicapResult("丸勝ち", 2.0, True)


def get_possible_outcomes(
    handicap_value: float,
    handicap_display: str = "",
) -> list:
    """ハンデの可能な結果パターンと配当を返す（EV計算用）。

    Returns:
        [(result_type, payout, diff_condition), ...]
        diff_condition: "lt_base", "eq_base", "eq_base_plus1", "gt_base_plus1"
    """
    base, is_half, border = parse_handicap_display(handicap_display, handicap_value)
    outcomes = []

    if is_half:
        threshold = base + 1
        outcomes.append(("丸負け", 0.0, f"diff<{threshold}"))
        if border == 0:
            outcomes.append(("丸勝ち", 2.0, f"diff>={threshold}"))
        else:
            win_type = BORDER_TO_WIN[border]
            outcomes.append((win_type, PAYOUT_TABLE[win_type], f"diff=={threshold}"))
            outcomes.append(("丸勝ち", 2.0, f"diff>{threshold}"))
    else:
        if base == 0:
            outcomes.append(("丸負け", 0.0, "diff<0"))
            if border == 0:
                outcomes.append(("勝負無し", 1.0, "diff==0"))
                outcomes.append(("丸勝ち", 2.0, "diff>=1"))
            else:
                loss_type = BORDER_TO_LOSS[border]
                win_type = BORDER_TO_WIN[border]
                outcomes.append((loss_type, PAYOUT_TABLE[loss_type], "diff==0"))
                outcomes.append((win_type, PAYOUT_TABLE[win_type], "diff==1"))
                outcomes.append(("丸勝ち", 2.0, "diff>=2"))
        else:
            outcomes.append(("丸負け", 0.0, f"diff<{base}"))
            if border == 0:
                outcomes.append(("勝負無し", 1.0, f"diff=={base}"))
            else:
                loss_type = BORDER_TO_LOSS[border]
                outcomes.append((loss_type, PAYOUT_TABLE[loss_type], f"diff=={base}"))
            outcomes.append(("丸勝ち", 2.0, f"diff>={base + 1}"))

    return outcomes


# 旧互換用
def calculate_ev(
    prob_favorable: float,
    prob_push: float = 0.0,
    avg_favorable_payout: float = 1.7,
    avg_unfavorable_payout: float = 0.4,
) -> float:
    """旧EV計算（互換性のため残す）"""
    prob_unfavorable = 1.0 - prob_favorable - prob_push
    return (
        prob_favorable * avg_favorable_payout
        + prob_push * 1.0
        + prob_unfavorable * avg_unfavorable_payout
        - 1.0
    )
