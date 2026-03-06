"""リーグ順位・勝ち点特徴量 (5個)"""

import pandas as pd
import numpy as np


def add_league_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    順位・勝ち点の特徴量を追加。
    過去の試合結果から累積的に順位を算出 (未来リーク防止)。

    必要カラム: home_team_id, away_team_id, home_score, away_score, date, league_code
    """
    df = df.sort_values("date").reset_index(drop=True)

    # チームごとの累積成績を計算
    home_results = df[["date", "league_code", "home_team_id", "home_score", "away_score"]].copy()
    home_results.columns = ["date", "league_code", "team_id", "scored", "conceded"]
    away_results = df[["date", "league_code", "away_team_id", "away_score", "home_score"]].copy()
    away_results.columns = ["date", "league_code", "team_id", "scored", "conceded"]

    all_results = pd.concat([home_results, away_results]).sort_values("date").reset_index(drop=True)
    all_results["win"] = (all_results["scored"] > all_results["conceded"]).astype(int)
    all_results["draw"] = (all_results["scored"] == all_results["conceded"]).astype(int)
    all_results["points"] = all_results["win"] * 3 + all_results["draw"]  # サッカー式勝ち点

    # 累積勝ち点 (shift for no leak)
    all_results["cum_points"] = (
        all_results.groupby("team_id")["points"]
        .transform(lambda x: x.cumsum().shift(1))
    ).fillna(0)

    all_results["cum_wins"] = (
        all_results.groupby("team_id")["win"]
        .transform(lambda x: x.cumsum().shift(1))
    ).fillna(0)

    all_results["cum_games"] = (
        all_results.groupby("team_id")["win"]
        .transform(lambda x: x.expanding().count().shift(1))
    ).fillna(1)

    all_results["cum_winrate"] = all_results["cum_wins"] / all_results["cum_games"].clip(lower=1)

    # 各日付・リーグ内での順位を計算
    # → 各試合時点でのチームの順位情報を作成
    # 直近のチーム stats をlookup用に保持
    team_latest = {}  # team_id → {cum_points, cum_winrate, cum_games}
    home_ranks = []
    away_ranks = []
    home_pts = []
    away_pts = []
    home_wrs = []
    away_wrs = []

    for _, row in df.iterrows():
        h_stats = team_latest.get(row["home_team_id"], {"cum_points": 0, "cum_winrate": 0.5, "cum_games": 0})
        a_stats = team_latest.get(row["away_team_id"], {"cum_points": 0, "cum_winrate": 0.5, "cum_games": 0})

        home_pts.append(h_stats["cum_points"])
        away_pts.append(a_stats["cum_points"])
        home_wrs.append(h_stats["cum_winrate"])
        away_wrs.append(a_stats["cum_winrate"])

        # 順位は勝ち点/勝率差で代替
        home_ranks.append(0)  # placeholder
        away_ranks.append(0)

        # 試合後に更新
        h_win = 1 if (row["home_score"] or 0) > (row["away_score"] or 0) else 0
        a_win = 1 if (row["away_score"] or 0) > (row["home_score"] or 0) else 0
        draw = 1 if (row["home_score"] or 0) == (row["away_score"] or 0) else 0

        for tid, win, is_draw in [
            (row["home_team_id"], h_win, draw),
            (row["away_team_id"], a_win, draw),
        ]:
            prev = team_latest.get(tid, {"cum_points": 0, "cum_winrate": 0.5, "cum_games": 0, "cum_wins": 0})
            new_games = prev["cum_games"] + 1
            new_wins = prev.get("cum_wins", 0) + win
            team_latest[tid] = {
                "cum_points": prev["cum_points"] + win * 3 + is_draw,
                "cum_winrate": new_wins / max(new_games, 1),
                "cum_games": new_games,
                "cum_wins": new_wins,
            }

    # 1-2. 勝ち点
    df["home_points"] = home_pts
    df["away_points"] = away_pts

    # 3. 勝ち点差
    df["points_diff"] = df["home_points"] - df["away_points"]

    # 4-5. 累積勝率
    df["home_season_wr"] = home_wrs
    df["away_season_wr"] = away_wrs

    return df
