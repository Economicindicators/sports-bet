"""ハンデ判定の全9パターン ユニットテスト"""

import pytest
from betting.handicap_resolver import resolve_handicap, calculate_ev, HandicapResult


class TestResolveHandicap:
    """全9段階の判定テスト"""

    def test_marugachi_large_win(self):
        """丸勝ち: adjusted >= 2.0"""
        # 有利チーム5点, 不利チーム1点, ハンデ1.5 → adj = (5-1)-1.5 = 2.5
        result = resolve_handicap(fav_score=5, underdog_score=1, handicap_value=1.5)
        assert result.result_type == "丸勝ち"
        assert result.payout_rate == 2.0
        assert result.is_favorable is True
        assert result.adjusted_margin == 2.5

    def test_marugachi_exact_boundary(self):
        """丸勝ち: adjusted == 2.0 (境界)"""
        # 4-0, ハンデ2.0 → adj = 4-0-2.0 = 2.0
        result = resolve_handicap(fav_score=4, underdog_score=0, handicap_value=2.0)
        assert result.result_type == "丸勝ち"
        assert result.payout_rate == 2.0

    def test_nanabugachi(self):
        """7分勝ち: 1.0 <= adjusted < 2.0"""
        # 3-0, ハンデ1.5 → adj = 3-0-1.5 = 1.5
        result = resolve_handicap(fav_score=3, underdog_score=0, handicap_value=1.5)
        assert result.result_type == "7分勝ち"
        assert result.payout_rate == 1.7
        assert result.is_favorable is True

    def test_nanabugachi_exact_boundary(self):
        """7分勝ち: adjusted == 1.0"""
        # 3-1, ハンデ1.0 → adj = 2-1.0 = 1.0
        result = resolve_handicap(fav_score=3, underdog_score=1, handicap_value=1.0)
        assert result.result_type == "7分勝ち"
        assert result.payout_rate == 1.7

    def test_gobugachi(self):
        """5分勝ち: 0.5 <= adjusted < 1.0"""
        # 2-0, ハンデ1.5 → adj = 2-0-1.5 = 0.5
        result = resolve_handicap(fav_score=2, underdog_score=0, handicap_value=1.5)
        assert result.result_type == "5分勝ち"
        assert result.payout_rate == 1.5
        assert result.is_favorable is True

    def test_sanbugachi(self):
        """3分勝ち: 0 < adjusted < 0.5"""
        # 2-1, ハンデ0.5 → adj = 1-0.5 = 0.5 → これは5分勝ちになる
        # 2-1, ハンデ0.8 → adj = 1-0.8 = 0.2
        result = resolve_handicap(fav_score=2, underdog_score=1, handicap_value=0.8)
        assert result.result_type == "3分勝ち"
        assert result.payout_rate == 1.3
        assert result.is_favorable is False

    def test_shoubunashi(self):
        """勝負無し: adjusted == 0"""
        # 2-1, ハンデ1.0 → adj = 1-1.0 = 0.0
        result = resolve_handicap(fav_score=2, underdog_score=1, handicap_value=1.0)
        assert result.result_type == "勝負無し"
        assert result.payout_rate == 1.0
        assert result.is_favorable is False

    def test_sanbumake(self):
        """3分負け: -0.5 < adjusted < 0"""
        # 1-1, ハンデ0.3 → adj = 0-0.3 = -0.3
        result = resolve_handicap(fav_score=1, underdog_score=1, handicap_value=0.3)
        assert result.result_type == "3分負け"
        assert result.payout_rate == 0.7
        assert result.is_favorable is False

    def test_gobumake(self):
        """5分負け: -1.0 < adjusted <= -0.5"""
        # 1-1, ハンデ1.0 → adj = 0-1.0 = -1.0 → これは7分負け
        # 1-1, ハンデ0.5 → adj = 0-0.5 = -0.5
        result = resolve_handicap(fav_score=1, underdog_score=1, handicap_value=0.5)
        assert result.result_type == "5分負け"
        assert result.payout_rate == 0.5
        assert result.is_favorable is False

    def test_nanabumake(self):
        """7分負け: -2.0 < adjusted <= -1.0"""
        # 0-0, ハンデ1.5 → adj = 0-0-1.5 = -1.5
        result = resolve_handicap(fav_score=0, underdog_score=0, handicap_value=1.5)
        assert result.result_type == "7分負け"
        assert result.payout_rate == 0.3
        assert result.is_favorable is False

    def test_nanabumake_exact_boundary(self):
        """7分負け: adjusted == -1.0"""
        result = resolve_handicap(fav_score=1, underdog_score=1, handicap_value=1.0)
        assert result.result_type == "7分負け"
        assert result.payout_rate == 0.3

    def test_marumake(self):
        """丸負け: adjusted <= -2.0"""
        # 0-2, ハンデ1.0 → adj = -2-1.0 = -3.0
        result = resolve_handicap(fav_score=0, underdog_score=2, handicap_value=1.0)
        assert result.result_type == "丸負け"
        assert result.payout_rate == 0.0
        assert result.is_favorable is False

    def test_marumake_exact_boundary(self):
        """丸負け: adjusted == -2.0"""
        result = resolve_handicap(fav_score=0, underdog_score=0, handicap_value=2.0)
        assert result.result_type == "丸負け"
        assert result.payout_rate == 0.0

    # ========== 実用的なシナリオ ==========

    def test_baseball_scenario(self):
        """野球: 巨人5-3阪神, ハンデ1.5 (巨人有利)"""
        result = resolve_handicap(fav_score=5, underdog_score=3, handicap_value=1.5)
        # adj = 2-1.5 = 0.5 → 5分勝ち
        assert result.result_type == "5分勝ち"
        assert result.payout_rate == 1.5

    def test_soccer_scenario(self):
        """サッカー: 鹿島1-0浦和, ハンデ0.5 (鹿島有利)"""
        result = resolve_handicap(fav_score=1, underdog_score=0, handicap_value=0.5)
        # adj = 1-0.5 = 0.5 → 5分勝ち
        assert result.result_type == "5分勝ち"

    def test_basketball_scenario(self):
        """バスケ: LAL 110-105 BOS, ハンデ3.5"""
        result = resolve_handicap(fav_score=110, underdog_score=105, handicap_value=3.5)
        # adj = 5-3.5 = 1.5 → 7分勝ち
        assert result.result_type == "7分勝ち"
        assert result.payout_rate == 1.7


class TestCalculateEV:
    """EV計算テスト"""

    def test_positive_ev(self):
        """プラスEV"""
        ev = calculate_ev(prob_favorable=0.6, avg_favorable_payout=1.7, avg_unfavorable_payout=0.4)
        # 0.6*1.7 + 0.4*0.4 - 1.0 = 1.02 + 0.16 - 1.0 = 0.18
        assert ev == pytest.approx(0.18, abs=0.01)

    def test_negative_ev(self):
        """マイナスEV"""
        ev = calculate_ev(prob_favorable=0.3, avg_favorable_payout=1.7, avg_unfavorable_payout=0.4)
        # 0.3*1.7 + 0.7*0.4 - 1.0 = 0.51 + 0.28 - 1.0 = -0.21
        assert ev == pytest.approx(-0.21, abs=0.01)

    def test_with_push_probability(self):
        """勝負無し確率あり"""
        ev = calculate_ev(
            prob_favorable=0.5,
            prob_push=0.1,
            avg_favorable_payout=1.7,
            avg_unfavorable_payout=0.4,
        )
        # 0.5*1.7 + 0.1*1.0 + 0.4*0.4 - 1.0 = 0.85 + 0.1 + 0.16 - 1.0 = 0.11
        assert ev == pytest.approx(0.11, abs=0.01)

    def test_break_even(self):
        """ブレークイーブン付近"""
        ev = calculate_ev(prob_favorable=0.5, avg_favorable_payout=1.5, avg_unfavorable_payout=0.5)
        # 0.5*1.5 + 0.5*0.5 - 1.0 = 0.75 + 0.25 - 1.0 = 0.0
        assert ev == pytest.approx(0.0, abs=0.01)
