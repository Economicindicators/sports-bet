"""Eloレーティング特徴量 (5個)

対戦相手の強さを加味した動的レーティングシステム。
単純勝率と違い「強い相手に勝った」「弱い相手に負けた」を区別できる。
"""

import pandas as pd
import numpy as np
from collections import defaultdict


# Elo定数 (デフォルト)
INITIAL_RATING = 1500
K_FACTOR = 32
HOME_ADVANTAGE = 60  # ホームチームのEloボーナス
MOV_MULTIPLIER = 0.5  # 得点差によるK補正係数 (Margin of Victory)
SEASON_REGRESSION = 0.33  # シーズン間の平均回帰率

# スポーツ別Eloパラメータ (Optuna最適化済み)
SPORT_ELO_PARAMS = {
    "baseball": {
        "k_factor": 5.1,
        "home_advantage": 25.6,
        "mov_multiplier": 0.142,
        "season_regression": 0.661,
    },
    "soccer": {
        "k_factor": 13.2,
        "home_advantage": 10.0,
        "mov_multiplier": 0.832,
        "season_regression": 0.115,
    },
    "basketball": {
        "k_factor": 10.0,
        "home_advantage": 38.3,
        "mov_multiplier": 0.690,
        "season_regression": 0.493,
    },
}


def _get_elo_params(sport_code=None) -> tuple:
    """スポーツ別Eloパラメータを返す"""
    if sport_code and sport_code in SPORT_ELO_PARAMS:
        p = SPORT_ELO_PARAMS[sport_code]
        return p["k_factor"], p["home_advantage"], p["mov_multiplier"], p["season_regression"]
    return K_FACTOR, HOME_ADVANTAGE, MOV_MULTIPLIER, SEASON_REGRESSION


def _expected_score(rating_a: float, rating_b: float) -> float:
    """AがBに勝つ期待確率"""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _mov_multiplier_calc(score_diff: int, mov_mult: float) -> float:
    """得点差によるK補正 (大差の試合はレーティング変動を大きく)"""
    return np.log(abs(score_diff) + 1) * mov_mult + 1.0


def add_elo_features(df: pd.DataFrame, sport_code=None) -> pd.DataFrame:
    """
    Eloレーティング特徴量を追加。

    追加カラム:
        - home_elo: ホームチームのElo (試合前時点)
        - away_elo: アウェイチームのElo (試合前時点)
        - elo_diff: Elo差 (home - away)
        - elo_expected: ホーム勝利のElo期待値 (HA込み)
        - elo_surprise: ハンデ有利チームのElo期待値と実際のズレ

    必要カラム: home_team_id, away_team_id, home_score, away_score, date
    """
    k, ha, mov_m, season_reg = _get_elo_params(sport_code)

    df = df.sort_values("date").reset_index(drop=True)

    # チーム別Eloレーティング
    elo = defaultdict(lambda: INITIAL_RATING)

    # シーズン追跡 (年が変わったら平均回帰)
    last_year = defaultdict(lambda: None)

    home_elos = []
    away_elos = []
    elo_diffs = []
    elo_expected_list = []

    for _, row in df.iterrows():
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]
        home_score = row["home_score"]
        away_score = row["away_score"]

        # 年を取得
        if hasattr(row["date"], "year"):
            current_year = row["date"].year
        else:
            current_year = pd.Timestamp(row["date"]).year

        # シーズン変わりの平均回帰
        for team_id in [home_id, away_id]:
            if last_year[team_id] is not None and last_year[team_id] != current_year:
                elo[team_id] = (
                    elo[team_id] * (1 - season_reg)
                    + INITIAL_RATING * season_reg
                )
            last_year[team_id] = current_year

        # 試合前のレーティングを記録
        home_r = elo[home_id]
        away_r = elo[away_id]
        home_elos.append(home_r)
        away_elos.append(away_r)
        elo_diffs.append(home_r - away_r)

        # ホームアドバンテージ込みの期待値
        expected_home = _expected_score(home_r + ha, away_r)
        elo_expected_list.append(expected_home)

        # 試合結果でElo更新
        if pd.isna(home_score) or pd.isna(away_score):
            continue

        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        # 得点差補正
        mov = _mov_multiplier_calc(int(home_score - away_score), mov_m)

        # Elo更新
        delta = k * mov * (actual_home - expected_home)
        elo[home_id] += delta
        elo[away_id] -= delta

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = elo_diffs
    df["elo_expected"] = elo_expected_list

    # ハンデ有利チームのElo期待値
    # handicap_team_is_home=1ならhome_eloが有利チーム
    if "handicap_team_is_home" in df.columns:
        df["elo_surprise"] = np.where(
            df["handicap_team_is_home"] == 1,
            df["elo_expected"],  # ハンデ有利=ホーム → ホーム勝利期待値
            1 - df["elo_expected"],  # ハンデ有利=アウェイ → アウェイ勝利期待値
        )
    else:
        df["elo_surprise"] = 0.5

    return df
