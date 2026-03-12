"""The Odds API からブックメーカーオッズを取得"""

import os
import logging
import requests
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("ODDS_API_KEY", "")
BASE_URL = "https://api.the-odds-api.com/v4"

# リーグコード → The Odds API のスポーツキー
LEAGUE_TO_ODDS_SPORT = {
    "nba": "basketball_nba",
    "mlb": "baseball_mlb",
    "npb": "baseball_npb",
    "premier": "soccer_epl",
    "laliga": "soccer_spain_la_liga",
    "seriea": "soccer_italy_serie_a",
    "bundesliga": "soccer_germany_bundesliga",
    "ligue1": "soccer_france_ligue_one",
    "eredivisie": "soccer_netherlands_eredivisie",
}

# ブックメーカー優先順位（日本から使いやすい順）
PREFERRED_BOOKMAKERS = [
    "pinnacle", "bet365", "unibet", "betfair_ex_eu",
    "williamhill", "marathonbet", "1xbet",
]


@dataclass
class BookmakerOdds:
    home_team: str
    away_team: str
    bookmaker: str
    home_odds: float
    away_odds: float
    draw_odds: Optional[float]  # サッカーのみ
    spread_home: Optional[float]  # ハンデ値
    spread_home_odds: Optional[float]
    spread_away_odds: Optional[float]


def get_odds(league_code: str, markets: str = "h2h,spreads") -> list[BookmakerOdds]:
    """指定リーグのオッズを取得"""
    if not API_KEY:
        logger.warning("ODDS_API_KEY not set")
        return []

    sport_key = LEAGUE_TO_ODDS_SPORT.get(league_code)
    if not sport_key:
        logger.warning(f"No Odds API mapping for league: {league_code}")
        return []

    try:
        resp = requests.get(
            f"{BASE_URL}/sports/{sport_key}/odds",
            params={
                "apiKey": API_KEY,
                "regions": "eu,uk",
                "markets": markets,
                "oddsFormat": "decimal",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"Odds API error for {league_code}: {e}")
        return []

    results = []
    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        for bm in game.get("bookmakers", []):
            bm_key = bm.get("key", "")

            h2h_home = None
            h2h_away = None
            h2h_draw = None
            spread_val = None
            spread_home_price = None
            spread_away_price = None

            for market in bm.get("markets", []):
                if market["key"] == "h2h":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home:
                            h2h_home = outcome["price"]
                        elif outcome["name"] == away:
                            h2h_away = outcome["price"]
                        elif outcome["name"] == "Draw":
                            h2h_draw = outcome["price"]

                elif market["key"] == "spreads":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home:
                            spread_val = outcome.get("point")
                            spread_home_price = outcome["price"]
                        elif outcome["name"] == away:
                            spread_away_price = outcome["price"]

            if h2h_home and h2h_away:
                results.append(BookmakerOdds(
                    home_team=home,
                    away_team=away,
                    bookmaker=bm_key,
                    home_odds=h2h_home,
                    away_odds=h2h_away,
                    draw_odds=h2h_draw,
                    spread_home=spread_val,
                    spread_home_odds=spread_home_price,
                    spread_away_odds=spread_away_price,
                ))

    return results


def get_best_odds(league_code: str) -> dict[str, BookmakerOdds]:
    """各試合のベストオッズ（Pinnacle優先）を返す。key = "home_team vs away_team" """
    all_odds = get_odds(league_code)
    if not all_odds:
        return {}

    # 試合ごとにグループ化
    by_game: dict[str, list[BookmakerOdds]] = {}
    for o in all_odds:
        key = f"{o.home_team} vs {o.away_team}"
        by_game.setdefault(key, []).append(o)

    best = {}
    for key, odds_list in by_game.items():
        # 優先ブックメーカーから選択
        selected = None
        for pref in PREFERRED_BOOKMAKERS:
            for o in odds_list:
                if o.bookmaker == pref:
                    selected = o
                    break
            if selected:
                break
        if not selected:
            selected = odds_list[0]
        best[key] = selected

    return best


def fetch_and_update_turso(league_code: str) -> int:
    """オッズを取得してTursoのpredictionsテーブルを更新"""
    import subprocess
    import tempfile

    best = get_best_odds(league_code)
    if not best:
        return 0

    lines = []
    for key, odds in best.items():
        # handicap_teamのオッズを使う（home or away）
        # predictionsテーブルのhome_team/away_teamと英語名をマッチする必要がある
        # TODO: チーム名マッチング（英語→日本語）
        # ひとまずh2hオッズの低い方（favoriteのオッズ）を保存
        fav_odds = min(odds.home_odds, odds.away_odds)
        underdog_odds = max(odds.home_odds, odds.away_odds)

        # Tursoに保存（英語チーム名でマッチ試行）
        lines.append(
            f"UPDATE predictions SET bookmaker_odds = {fav_odds} "
            f"WHERE status = 'pending' AND date >= date('now') "
            f"AND (home_team LIKE '%' || '{odds.home_team}' || '%' "
            f"OR away_team LIKE '%' || '{odds.away_team}' || '%');"
        )

    if not lines:
        return 0

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
        f.write("\n".join(lines))
        sql_path = f.name

    try:
        result = subprocess.run(
            ["turso", "db", "shell", "sports-bet"],
            stdin=open(sql_path),
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            logger.error(f"Turso error: {result.stderr}")
    except Exception as e:
        logger.error(f"Turso update failed: {e}")

    return len(lines)
