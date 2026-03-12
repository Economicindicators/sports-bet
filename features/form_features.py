"""直近フォーム特徴量 — チーム/選手の好調・不調を捕捉"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RECENT_WINDOW = 5  # 直近5試合


def add_form_features(df: pd.DataFrame, sport_code: str) -> pd.DataFrame:
    """好調/不調の特徴量を追加 (全スポーツ共通 + スポーツ別)"""
    if df.empty:
        return df

    df = df.sort_values("date").reset_index(drop=True)

    # --- 共通: チームの直近フォーム ---
    df = _add_team_recent_form(df)

    # --- 共通: 得失点トレンド ---
    df = _add_scoring_trend(df)

    # --- 共通: ハンデカバー率の直近推移 ---
    df = _add_recent_cover_rate(df)

    # --- 野球: 先発投手の直近フォーム ---
    if sport_code == "baseball":
        df = _add_pitcher_recent_form(df)

    logger.info(f"Added form features for {sport_code}")
    return df


def _add_team_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    """チーム別 直近N試合の勝率 (ホーム/アウェイ問わず全試合ベース)"""
    # 各チームの全試合結果を時系列で追跡
    team_results: dict[int, list[float]] = {}
    home_form = []
    away_form = []

    for _, row in df.iterrows():
        ht = int(row["home_team_id"])
        at = int(row["away_team_id"])

        # この試合前のフォーム
        h_hist = team_results.get(ht, [])
        a_hist = team_results.get(at, [])
        home_form.append(np.mean(h_hist[-RECENT_WINDOW:]) if h_hist else 0.5)
        away_form.append(np.mean(a_hist[-RECENT_WINDOW:]) if a_hist else 0.5)

        # 試合結果を記録
        hs = row.get("home_score")
        as_ = row.get("away_score")
        if pd.notna(hs) and pd.notna(as_):
            team_results.setdefault(ht, []).append(1.0 if hs > as_ else 0.0)
            team_results.setdefault(at, []).append(1.0 if as_ > hs else 0.0)

    df["home_recent_form"] = home_form
    df["away_recent_form"] = away_form
    df["form_diff"] = df["home_recent_form"] - df["away_recent_form"]
    return df


def _add_scoring_trend(df: pd.DataFrame) -> pd.DataFrame:
    """直近N試合の平均得点/失点 (フォームの得点力指標)"""
    team_scored: dict[int, list[float]] = {}
    team_conceded: dict[int, list[float]] = {}
    home_scored_trend = []
    away_scored_trend = []
    home_conceded_trend = []
    away_conceded_trend = []

    for _, row in df.iterrows():
        ht = int(row["home_team_id"])
        at = int(row["away_team_id"])

        hs_hist = team_scored.get(ht, [])
        as_hist = team_scored.get(at, [])
        hc_hist = team_conceded.get(ht, [])
        ac_hist = team_conceded.get(at, [])

        home_scored_trend.append(np.mean(hs_hist[-RECENT_WINDOW:]) if hs_hist else 0.0)
        away_scored_trend.append(np.mean(as_hist[-RECENT_WINDOW:]) if as_hist else 0.0)
        home_conceded_trend.append(np.mean(hc_hist[-RECENT_WINDOW:]) if hc_hist else 0.0)
        away_conceded_trend.append(np.mean(ac_hist[-RECENT_WINDOW:]) if ac_hist else 0.0)

        hs = row.get("home_score")
        as_ = row.get("away_score")
        if pd.notna(hs) and pd.notna(as_):
            team_scored.setdefault(ht, []).append(float(hs))
            team_scored.setdefault(at, []).append(float(as_))
            team_conceded.setdefault(ht, []).append(float(as_))
            team_conceded.setdefault(at, []).append(float(hs))

    df["home_scored_trend"] = home_scored_trend
    df["away_scored_trend"] = away_scored_trend
    df["home_conceded_trend"] = home_conceded_trend
    df["away_conceded_trend"] = away_conceded_trend
    # 攻守バランス: 得点トレンド - 失点トレンド
    df["home_net_trend"] = df["home_scored_trend"] - df["home_conceded_trend"]
    df["away_net_trend"] = df["away_scored_trend"] - df["away_conceded_trend"]
    return df


def _add_recent_cover_rate(df: pd.DataFrame) -> pd.DataFrame:
    """ハンデ有利チームの直近カバー率 (ハンデ勝負でのフォーム)"""
    team_covers: dict[int, list[float]] = {}
    home_cover = []
    away_cover = []

    for _, row in df.iterrows():
        ht = int(row["home_team_id"])
        at = int(row["away_team_id"])

        h_hist = team_covers.get(ht, [])
        a_hist = team_covers.get(at, [])
        home_cover.append(np.mean(h_hist[-RECENT_WINDOW:]) if h_hist else 0.5)
        away_cover.append(np.mean(a_hist[-RECENT_WINDOW:]) if a_hist else 0.5)

        pr = row.get("payout_rate")
        handi_team = row.get("handicap_team_id")
        if pd.notna(pr) and pd.notna(handi_team):
            covered = 1.0 if pr >= 1.5 else 0.0
            # ハンデ有利チーム側にカバー結果を記録
            if int(handi_team) == ht:
                team_covers.setdefault(ht, []).append(covered)
            elif int(handi_team) == at:
                team_covers.setdefault(at, []).append(covered)

    df["home_recent_cover"] = home_cover
    df["away_recent_cover"] = away_cover
    return df


def _add_pitcher_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    """先発投手の直近N登板のフォーム (野球のみ)"""
    pitcher_results: dict[int, list[float]] = {}
    pitcher_runs: dict[int, list[float]] = {}
    home_p_form = []
    away_p_form = []
    home_p_era_recent = []
    away_p_era_recent = []

    for _, row in df.iterrows():
        hp = row.get("home_pitcher_id")
        ap = row.get("away_pitcher_id")

        # ホーム投手
        if pd.notna(hp):
            hp = int(hp)
            h_hist = pitcher_results.get(hp, [])
            h_runs = pitcher_runs.get(hp, [])
            home_p_form.append(np.mean(h_hist[-RECENT_WINDOW:]) if h_hist else 0.5)
            home_p_era_recent.append(np.mean(h_runs[-RECENT_WINDOW:]) if h_runs else 4.0)
        else:
            home_p_form.append(0.5)
            home_p_era_recent.append(4.0)

        # アウェイ投手
        if pd.notna(ap):
            ap = int(ap)
            a_hist = pitcher_results.get(ap, [])
            a_runs = pitcher_runs.get(ap, [])
            away_p_form.append(np.mean(a_hist[-RECENT_WINDOW:]) if a_hist else 0.5)
            away_p_era_recent.append(np.mean(a_runs[-RECENT_WINDOW:]) if a_runs else 4.0)
        else:
            away_p_form.append(0.5)
            away_p_era_recent.append(4.0)

        # 結果を記録
        hs = row.get("home_score")
        as_ = row.get("away_score")
        if pd.notna(hs) and pd.notna(as_):
            if pd.notna(row.get("home_pitcher_id")):
                pid = int(row["home_pitcher_id"])
                pitcher_results.setdefault(pid, []).append(1.0 if hs > as_ else 0.0)
                pitcher_runs.setdefault(pid, []).append(float(as_))
            if pd.notna(row.get("away_pitcher_id")):
                pid = int(row["away_pitcher_id"])
                pitcher_results.setdefault(pid, []).append(1.0 if as_ > hs else 0.0)
                pitcher_runs.setdefault(pid, []).append(float(hs))

    df["home_pitcher_recent_form"] = home_p_form
    df["away_pitcher_recent_form"] = away_p_form
    df["home_pitcher_era_recent"] = home_p_era_recent
    df["away_pitcher_era_recent"] = away_p_era_recent
    df["pitcher_form_diff"] = df["home_pitcher_recent_form"] - df["away_pitcher_recent_form"]
    return df
