"""ブックメーカーオッズを取得してTursoのpendingな予測に紐付ける"""

import sys
import logging
import subprocess
import tempfile
from scraper.odds_api import get_best_odds, LEAGUE_TO_ODDS_SPORT
from scraper.team_name_map import find_ja_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def fetch_and_save(leagues: list[str] = None) -> int:
    if leagues is None:
        leagues = list(LEAGUE_TO_ODDS_SPORT.keys())

    sql_lines = []
    total = 0

    for league in leagues:
        best = get_best_odds(league)
        if not best:
            logger.info(f"{league}: no odds data")
            continue

        for key, odds in best.items():
            home_ja = find_ja_name(odds.home_team)
            away_ja = find_ja_name(odds.away_team)

            if not home_ja or not away_ja:
                logger.warning(f"Name not found: {odds.home_team} → {home_ja}, {odds.away_team} → {away_ja}")
                continue

            # handicap_teamのオッズを使いたいが、どちらがhandicap_teamかはpredictionsテーブルを見ないとわからない
            # → home_odds と away_odds の両方を使い、handicap_team に合うほうを UPDATE
            esc = lambda s: s.replace("'", "''")
            # home側がhandicap_teamの場合
            sql_lines.append(
                f"UPDATE predictions SET bookmaker_odds = {odds.home_odds} "
                f"WHERE status = 'pending' AND handicap_team = '{esc(home_ja)}' "
                f"AND home_team = '{esc(home_ja)}' AND away_team = '{esc(away_ja)}';"
            )
            # away側がhandicap_teamの場合
            sql_lines.append(
                f"UPDATE predictions SET bookmaker_odds = {odds.away_odds} "
                f"WHERE status = 'pending' AND handicap_team = '{esc(away_ja)}' "
                f"AND home_team = '{esc(home_ja)}' AND away_team = '{esc(away_ja)}';"
            )
            total += 1
            logger.info(f"  {home_ja} vs {away_ja}: {odds.home_odds} / {odds.away_odds} ({odds.bookmaker})")

    if not sql_lines:
        logger.info("No odds to update")
        return 0

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
        f.write("\n".join(sql_lines))
        sql_path = f.name

    try:
        result = subprocess.run(
            ["turso", "db", "shell", "sports-bet"],
            stdin=open(sql_path),
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            logger.error(f"Turso error: {result.stderr}")
            return 0
    except Exception as e:
        logger.error(f"Turso update failed: {e}")
        return 0

    logger.info(f"Updated {total} matches with bookmaker odds")
    return total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch bookmaker odds and update Turso")
    parser.add_argument("--league", nargs="+", default=None, help="Leagues to fetch")
    args = parser.parse_args()
    fetch_and_save(args.league)
