"""Advanced features: elo velocity, win streak, rest days, home/away splits, h2h cover rate."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 6 advanced features to the DataFrame."""
    if df.empty:
        return df

    df = df.sort_values("date").reset_index(drop=True)

    df = _add_elo_velocity(df)
    df = _add_win_streak(df)
    df = _add_rest_days(df)
    df = _add_home_away_win_rates(df)
    df = _add_h2h_cover_rate(df)

    logger.info("Added advanced features")
    return df


def _add_elo_velocity(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Elo change over last N games (momentum indicator)."""
    if "home_elo" not in df.columns:
        df["home_elo_velocity"] = 0.0
        df["away_elo_velocity"] = 0.0
        return df

    home_vel = []
    away_vel = []
    team_elo_history = {}  # team_id -> list of elo values

    for _, row in df.iterrows():
        ht = row["home_team_id"]
        at = row["away_team_id"]

        for tid, elo_col, vel_list in [
            (ht, "home_elo", home_vel),
            (at, "away_elo", away_vel),
        ]:
            history = team_elo_history.get(tid, [])
            if len(history) >= 2:
                recent = history[-window:] if len(history) >= window else history
                vel_list.append(recent[-1] - recent[0])
            else:
                vel_list.append(0.0)

            team_elo_history.setdefault(tid, []).append(row[elo_col])

    df["home_elo_velocity"] = home_vel
    df["away_elo_velocity"] = away_vel
    return df


def _add_win_streak(df: pd.DataFrame) -> pd.DataFrame:
    """Current win/loss streak for each team."""
    team_streak = {}  # team_id -> int (positive = wins, negative = losses)
    home_streaks = []
    away_streaks = []

    for _, row in df.iterrows():
        ht = row["home_team_id"]
        at = row["away_team_id"]

        home_streaks.append(team_streak.get(ht, 0))
        away_streaks.append(team_streak.get(at, 0))

        hs = row.get("home_score")
        as_ = row.get("away_score")
        if pd.notna(hs) and pd.notna(as_):
            if hs > as_:
                team_streak[ht] = max(0, team_streak.get(ht, 0)) + 1
                team_streak[at] = min(0, team_streak.get(at, 0)) - 1
            elif as_ > hs:
                team_streak[at] = max(0, team_streak.get(at, 0)) + 1
                team_streak[ht] = min(0, team_streak.get(ht, 0)) - 1
            else:
                team_streak[ht] = 0
                team_streak[at] = 0

    df["home_win_streak"] = home_streaks
    df["away_win_streak"] = away_streaks
    return df


def _add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Days since last game for each team."""
    team_last_date = {}
    home_rest = []
    away_rest = []

    for _, row in df.iterrows():
        ht = row["home_team_id"]
        at = row["away_team_id"]
        d = pd.Timestamp(row["date"])

        for tid, rest_list in [(ht, home_rest), (at, away_rest)]:
            last = team_last_date.get(tid)
            if last is not None:
                rest_list.append((d - last).days)
            else:
                rest_list.append(3)  # default
            team_last_date[tid] = d

    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest
    return df


def _add_home_away_win_rates(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling home win rate and away win rate for each team."""
    team_home_results = {}  # team_id -> list of 0/1
    team_away_results = {}
    home_wr = []
    away_wr = []

    for _, row in df.iterrows():
        ht = row["home_team_id"]
        at = row["away_team_id"]

        # Current rates before this game
        h_hist = team_home_results.get(ht, [])
        a_hist = team_away_results.get(at, [])
        home_wr.append(np.mean(h_hist[-window:]) if h_hist else 0.5)
        away_wr.append(np.mean(a_hist[-window:]) if a_hist else 0.5)

        hs = row.get("home_score")
        as_ = row.get("away_score")
        if pd.notna(hs) and pd.notna(as_):
            team_home_results.setdefault(ht, []).append(1 if hs > as_ else 0)
            team_away_results.setdefault(at, []).append(1 if as_ > hs else 0)

    df["home_home_win_rate"] = home_wr
    df["away_away_win_rate"] = away_wr
    return df


def _add_h2h_cover_rate(df: pd.DataFrame, min_games: int = 3) -> pd.DataFrame:
    """Head-to-head handicap cover rate (favorite covering the spread)."""
    h2h_results = {}  # (team_a, team_b) -> list of cover booleans
    cover_rates = []

    for _, row in df.iterrows():
        key = tuple(sorted([int(row["home_team_id"]), int(row["away_team_id"])]))
        hist = h2h_results.get(key, [])

        if len(hist) >= min_games:
            cover_rates.append(np.mean(hist))
        else:
            cover_rates.append(0.5)

        pr = row.get("payout_rate")
        if pd.notna(pr):
            h2h_results.setdefault(key, []).append(1 if pr >= 1.5 else 0)

    df["h2h_cover_rate"] = cover_rates
    return df
