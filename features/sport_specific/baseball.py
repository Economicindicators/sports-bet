"""野球固有特徴量: 投手成績、得点パターン、リーグ差、交互作用"""

import pandas as pd
import numpy as np

RECENT_WINDOW = 5


def add_baseball_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    野球固有の特徴量を追加。

    必要カラム: home_pitcher_id, away_pitcher_id, home_score, away_score, date
    """
    df = df.sort_values("date").reset_index(drop=True)

    # === 基本投手特徴量 ===
    df = _add_pitcher_career_stats(df)

    # === 投手の登板間隔・疲労 ===
    df = _add_pitcher_rest(df)

    # === チーム得点パターン ===
    df = _add_run_scoring_patterns(df)

    # === リーグ特性 ===
    df = _add_league_context(df)

    # === 球場タイプ (ドーム/屋外) ===
    df = _add_venue_type(df)

    # === 交互作用項 ===
    df = _add_interactions(df)

    return df


def _add_pitcher_career_stats(df: pd.DataFrame) -> pd.DataFrame:
    """先発投手の通算成績（ERA、勝率、登板数）"""
    # ホーム投手の勝率 (expanding, shift)
    df["_hp_win"] = (df["home_score"] > df["away_score"]).astype(float)
    df["home_pitcher_wr"] = (
        df.groupby("home_pitcher_id")["_hp_win"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.5)

    df["_ap_win"] = (df["away_score"] > df["home_score"]).astype(float)
    df["away_pitcher_wr"] = (
        df.groupby("away_pitcher_id")["_ap_win"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(0.5)

    # 投手力差（勝率ベース）
    df["pitcher_wr_diff"] = df["home_pitcher_wr"] - df["away_pitcher_wr"]

    # 先発投手の過去平均失点 (ERA的な指標)
    df["home_pitcher_era"] = (
        df.groupby("home_pitcher_id")["away_score"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(df["away_score"].mean())

    df["away_pitcher_era"] = (
        df.groupby("away_pitcher_id")["home_score"]
        .transform(lambda x: x.expanding().mean().shift(1))
    ).fillna(df["home_score"].mean())

    # ERA差 (ホーム投手のERAが低い=ホーム有利 → 正の値)
    df["pitcher_era_diff"] = df["away_pitcher_era"] - df["home_pitcher_era"]

    # 登板回数 (経験値)
    df["home_pitcher_starts"] = df.groupby("home_pitcher_id").cumcount()
    df["away_pitcher_starts"] = df.groupby("away_pitcher_id").cumcount()

    # 投手経験差
    df["pitcher_exp_diff"] = df["home_pitcher_starts"] - df["away_pitcher_starts"]

    df = df.drop(columns=["_hp_win", "_ap_win"], errors="ignore")
    return df


def _add_pitcher_rest(df: pd.DataFrame) -> pd.DataFrame:
    """投手の登板間隔（疲労指標）"""
    # 各投手の前回登板日からの日数
    pitcher_last_start: dict[int, pd.Timestamp] = {}
    home_rest = []
    away_rest = []

    for _, row in df.iterrows():
        game_date = pd.Timestamp(row["date"])
        hp = row.get("home_pitcher_id")
        ap = row.get("away_pitcher_id")

        # ホーム投手の登板間隔
        if pd.notna(hp):
            hp = int(hp)
            last = pitcher_last_start.get(hp)
            if last is not None:
                days = (game_date - last).days
                home_rest.append(min(days, 30))
            else:
                home_rest.append(7)  # 初登板はデフォルト7日
        else:
            home_rest.append(7)

        # アウェイ投手の登板間隔
        if pd.notna(ap):
            ap = int(ap)
            last = pitcher_last_start.get(ap)
            if last is not None:
                days = (game_date - last).days
                away_rest.append(min(days, 30))
            else:
                away_rest.append(7)
        else:
            away_rest.append(7)

        # 登板日を記録
        if pd.notna(row.get("home_pitcher_id")):
            pitcher_last_start[int(row["home_pitcher_id"])] = game_date
        if pd.notna(row.get("away_pitcher_id")):
            pitcher_last_start[int(row["away_pitcher_id"])] = game_date

    df["home_pitcher_rest_days"] = home_rest
    df["away_pitcher_rest_days"] = away_rest
    df["pitcher_rest_diff"] = df["home_pitcher_rest_days"] - df["away_pitcher_rest_days"]

    # 中4日以下は疲労フラグ
    df["home_pitcher_fatigued"] = (df["home_pitcher_rest_days"] <= 4).astype(int)
    df["away_pitcher_fatigued"] = (df["away_pitcher_rest_days"] <= 4).astype(int)

    return df


def _add_run_scoring_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """チームの得点パターン特徴量"""
    # リーグ全体の得点トレンド
    df["total_runs_avg"] = (
        (df["home_score"] + df["away_score"])
        .expanding().mean().shift(1)
    ).fillna(df["home_score"].mean() + df["away_score"].mean())

    # チーム別のロースコア/ハイスコア傾向
    team_runs: dict[int, list[float]] = {}
    team_allowed: dict[int, list[float]] = {}
    home_run_std = []
    away_run_std = []
    home_shutout_rate = []
    away_shutout_rate = []

    for _, row in df.iterrows():
        ht = int(row["home_team_id"])
        at = int(row["away_team_id"])
        hs = row.get("home_score")
        as_ = row.get("away_score")

        # 得点のばらつき（安定感）
        h_runs = team_runs.get(ht, [])
        a_runs = team_runs.get(at, [])
        home_run_std.append(np.std(h_runs[-10:]) if len(h_runs) >= 3 else 2.0)
        away_run_std.append(np.std(a_runs[-10:]) if len(a_runs) >= 3 else 2.0)

        # 完封率（投手力の指標）
        h_allowed = team_allowed.get(ht, [])
        a_allowed = team_allowed.get(at, [])
        home_shutout_rate.append(
            np.mean([1.0 if r == 0 else 0.0 for r in h_allowed[-20:]]) if h_allowed else 0.0
        )
        away_shutout_rate.append(
            np.mean([1.0 if r == 0 else 0.0 for r in a_allowed[-20:]]) if a_allowed else 0.0
        )

        # 記録
        if pd.notna(hs) and pd.notna(as_):
            team_runs.setdefault(ht, []).append(float(hs))
            team_runs.setdefault(at, []).append(float(as_))
            team_allowed.setdefault(ht, []).append(float(as_))
            team_allowed.setdefault(at, []).append(float(hs))

    df["home_run_volatility"] = home_run_std
    df["away_run_volatility"] = away_run_std
    df["home_shutout_rate"] = home_shutout_rate
    df["away_shutout_rate"] = away_shutout_rate

    return df


def _add_league_context(df: pd.DataFrame) -> pd.DataFrame:
    """リーグ特性（NPB vs MLB差）"""
    # NPBフラグ（リーグ差の学習用）
    df["is_npb"] = (df["league_code"] == "npb").astype(int)

    # NPB/MLBの平均得点差を反映（MLBの方がスコアが高い傾向）
    league_avg_runs: dict[str, list[float]] = {}
    league_run_feature = []

    for _, row in df.iterrows():
        lc = row["league_code"]
        hist = league_avg_runs.get(lc, [])
        league_run_feature.append(np.mean(hist[-50:]) if hist else 8.0)

        hs = row.get("home_score")
        as_ = row.get("away_score")
        if pd.notna(hs) and pd.notna(as_):
            league_avg_runs.setdefault(lc, []).append(float(hs) + float(as_))

    df["league_avg_total_runs"] = league_run_feature

    return df


def _add_venue_type(df: pd.DataFrame) -> pd.DataFrame:
    """球場タイプ: ドーム球場かどうか (得点傾向に影響)"""
    if "venue" not in df.columns:
        df["is_dome"] = 0
        return df

    def _check_dome(venue: object) -> int:
        if pd.isna(venue):
            return 0
        v = str(venue).lower()
        # "ドーム" covers 東京ドーム, PayPayドーム, バンテリンドーム, 京セラドーム, ベルーナドーム etc.
        # "dome" covers MLB dome stadiums
        # "エスコン" = エスコンフィールド (retractable roof, usually closed)
        if "ドーム" in v or "dome" in v or "エスコン" in v:
            return 1
        return 0

    df["is_dome"] = df["venue"].apply(_check_dome)
    return df


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """重要な交互作用項"""
    # 投手力 × チーム打力（投手が良くても打線が弱いと勝てない）
    df["pitcher_era_x_scored"] = (
        df.get("home_pitcher_era", pd.Series(4.0, index=df.index)) *
        df.get("home_scored_trend", pd.Series(4.0, index=df.index))
    )

    # 投手疲労 × ERA（疲れてる良い投手 vs 元気な普通の投手）
    df["pitcher_fatigue_x_era"] = (
        df["home_pitcher_fatigued"] * df.get("home_pitcher_era", pd.Series(4.0, index=df.index))
    )

    # Elo差 × ハンデ値（実力差とハンデの乖離＝価値のあるベット）
    if "elo_diff" in df.columns:
        df["elo_x_handicap"] = df["elo_diff"] * df["handicap_value"]

    # 投手力差 × 休養差
    df["pitcher_quality_x_rest"] = df["pitcher_era_diff"] * df["pitcher_rest_diff"]

    return df
