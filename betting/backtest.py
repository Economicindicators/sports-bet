"""Walk-forward バックテスト"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config.constants import KELLY_FRACTION, MIN_EV_THRESHOLD
from betting.handicap_ev import calculate_handicap_ev, home_prob_to_handicap_prob, AVG_FAVORABLE_PAYOUT
from betting.kelly import kelly_fraction
from models.lgbm_model import LightGBMModel
from models.training import SPORT_LGBM_PARAMS

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """バックテスト結果"""

    total_bets: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    roi: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    period_results: list = field(default_factory=list)
    bet_history: list = field(default_factory=list)


def walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_periods: int = 4,
    retrain_gap_days: int = 30,
    ev_threshold: float = MIN_EV_THRESHOLD,
    bet_size: float = 1000,
    sport_code: str = None,
) -> BacktestResult:
    """
    Walk-forward バックテスト。

    各期間: 過去データで学習 → gap → 未来データでベット

    Args:
        df: 特徴量付きDataFrame
        feature_cols: 特徴量カラム
        n_periods: 期間数
        retrain_gap_days: 再学習のギャップ日数
        ev_threshold: EV閾値
        bet_size: 1ベットの金額

    Returns:
        BacktestResult
    """
    # NaN target (scheduled試合) を除外
    df = df.dropna(subset=["target"]).sort_values("date").reset_index(drop=True)
    dates = df["date"].unique()
    n_dates = len(dates)

    if n_dates < n_periods + 1:
        logger.warning("Not enough data for walk-forward")
        return BacktestResult()

    period_size = n_dates // (n_periods + 1)
    result = BacktestResult()

    cumulative_pnl = 0.0
    peak_pnl = 0.0
    max_dd = 0.0
    pnl_list = []

    for period in range(n_periods):
        # Train: 先頭 ～ boundary
        train_end_idx = (period + 1) * period_size
        train_end_date = dates[min(train_end_idx, n_dates - 1)]
        gap_date = train_end_date + timedelta(days=retrain_gap_days)

        # Test: gap後 ～ 次のboundary
        if period < n_periods - 1:
            test_end_idx = (period + 2) * period_size
            test_end_date = dates[min(test_end_idx, n_dates - 1)]
        else:
            test_end_date = dates[-1]

        train_mask = df["date"] <= train_end_date
        test_mask = (df["date"] > gap_date) & (df["date"] <= test_end_date)

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, "target"]
        X_test = df.loc[test_mask, feature_cols]

        if len(X_train) < 50 or len(X_test) < 5:
            continue

        # 学習 (スポーツ別Optunaパラメータ使用)
        sport_params = SPORT_LGBM_PARAMS.get(sport_code)
        model = LightGBMModel(params=sport_params)
        # train/valを8:2で分離
        split = int(len(X_train) * 0.8)
        model.train(
            X_train.iloc[:split], y_train.iloc[:split],
            X_train.iloc[split:], y_train.iloc[split:],
        )

        # 予測: ホーム勝率 → ハンデ補正 → EV
        probs = model.predict_proba(X_test)
        test_df = df.loc[test_mask].copy()
        test_df["home_win_prob"] = probs

        sport_code = test_df["sport_code"].iloc[0] if "sport_code" in test_df.columns else "baseball"
        test_df["handicap_team_is_home"] = (
            test_df["handicap_team_id"] == test_df["home_team_id"]
        ).astype(int)
        test_df["pred_prob"] = test_df.apply(
            lambda r: home_prob_to_handicap_prob(
                r["home_win_prob"],
                bool(r["handicap_team_is_home"]),
                r.get("handicap_value", 0),
                sport=sport_code,
            ), axis=1
        )
        test_df["handicap_ev"] = test_df["pred_prob"].apply(calculate_handicap_ev)

        # フィルタ
        bet_df = test_df[test_df["handicap_ev"] >= ev_threshold]

        period_pnl = 0.0
        period_bets = 0
        period_wins = 0

        for _, row in bet_df.iterrows():
            payout = row.get("payout_rate", 0.0)
            if payout is None or pd.isna(payout):
                continue
            pnl = bet_size * (payout - 1.0)
            period_pnl += pnl
            period_bets += 1
            if payout >= 1.5:
                period_wins += 1

            result.bet_history.append({
                "date": str(row["date"]),
                "match_id": row["match_id"],
                "pred_prob": row["pred_prob"],
                "ev": row["handicap_ev"],
                "payout": payout,
                "pnl": pnl,
            })

            # ドローダウン追跡
            cumulative_pnl += pnl
            peak_pnl = max(peak_pnl, cumulative_pnl)
            dd = peak_pnl - cumulative_pnl
            max_dd = max(max_dd, dd)
            pnl_list.append(pnl)

        period_roi = period_pnl / (period_bets * bet_size) * 100 if period_bets > 0 else 0

        result.period_results.append({
            "period": period,
            "train_end": str(train_end_date),
            "test_range": f"{gap_date}~{test_end_date}",
            "bets": period_bets,
            "wins": period_wins,
            "pnl": period_pnl,
            "roi": period_roi,
        })

        result.total_bets += period_bets
        result.wins += period_wins
        result.total_pnl += period_pnl

        logger.info(
            f"Period {period}: {period_bets} bets, "
            f"ROI={period_roi:.1f}%, PnL={period_pnl:+,.0f}"
        )

    # 全体メトリクス
    if result.total_bets > 0:
        result.roi = result.total_pnl / (result.total_bets * bet_size) * 100
        result.max_drawdown = max_dd

        if pnl_list:
            pnl_arr = np.array(pnl_list)
            if pnl_arr.std() > 0:
                result.sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(252)

    logger.info(
        f"Walk-forward: {result.total_bets} bets, "
        f"ROI={result.roi:.1f}%, PnL={result.total_pnl:+,.0f}, "
        f"MaxDD={result.max_drawdown:,.0f}, Sharpe={result.sharpe:.2f}"
    )

    return result
