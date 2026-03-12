"""Auto-settle predictions by matching with actual results from local DB."""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from datetime import date, timedelta

from database.models import get_session, Match, HandicapData, Team
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TURSO_DB = "sports-bet"


def get_pending_predictions() -> list[dict]:
    """Get pending predictions from Turso."""
    result = subprocess.run(
        ["turso", "db", "shell", TURSO_DB],
        input="SELECT id, match_id, sport, home_team, away_team, date FROM predictions WHERE status = 'pending';\n",
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        logger.error(f"Turso error: {result.stderr}")
        return []

    rows = []
    for line in result.stdout.strip().split("\n"):
        if not line or line.startswith("id") or line.startswith("-"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 6:
            rows.append({
                "id": int(parts[0]),
                "match_id": int(parts[1]),
                "sport": parts[2],
                "home_team": parts[3],
                "away_team": parts[4],
                "date": parts[5],
            })
    return rows


def find_result_in_local_db(pred: dict) -> dict | None:
    """Look up actual result for a prediction in local SQLite."""
    session = get_session()

    # Try exact match_id first
    match = session.get(Match, pred["match_id"])
    if match and match.status == "finished":
        hd = session.query(HandicapData).filter_by(match_id=match.match_id).first()
        if hd and hd.result_type:
            result = {
                "result_type": hd.result_type,
                "payout_rate": hd.payout_rate or 0,
            }
            session.close()
            return result

    # Fuzzy match by team names and date
    if pred.get("date"):
        matches = session.query(Match).filter(
            Match.date == pred["date"],
            Match.status == "finished",
        ).all()

        for m in matches:
            home = session.get(Team, m.home_team_id)
            away = session.get(Team, m.away_team_id)
            if not home or not away:
                continue
            if (pred["home_team"] in home.name or home.name in pred["home_team"]) and \
               (pred["away_team"] in away.name or away.name in pred["away_team"]):
                hd = session.query(HandicapData).filter_by(match_id=m.match_id).first()
                if hd and hd.result_type:
                    result = {
                        "result_type": hd.result_type,
                        "payout_rate": hd.payout_rate or 0,
                    }
                    session.close()
                    return result

    session.close()
    return None


def settle_predictions() -> dict:
    """Settle all pending predictions."""
    pending = get_pending_predictions()
    logger.info(f"Found {len(pending)} pending predictions")

    settled = 0
    sql_lines = []

    for pred in pending:
        result = find_result_in_local_db(pred)
        if result:
            esc = lambda s: str(s).replace("'", "''")
            sql_lines.append(
                f"UPDATE predictions SET "
                f"status = 'settled', "
                f"result_type = '{esc(result['result_type'])}', "
                f"payout_rate = {result['payout_rate']} "
                f"WHERE id = {pred['id']};"
            )
            settled += 1
            logger.info(
                f"Settled: {pred['home_team']} vs {pred['away_team']} "
                f"-> {result['result_type']} ({result['payout_rate']}x)"
            )

    if sql_lines:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
            f.write("\n".join(sql_lines))
            sql_path = f.name

        result = subprocess.run(
            ["turso", "db", "shell", TURSO_DB],
            stdin=open(sql_path),
            capture_output=True, text=True, timeout=60,
        )

        if result.returncode != 0:
            logger.error(f"Turso update error: {result.stderr}")
        else:
            logger.info(f"Updated {settled} predictions in Turso")

    return {"pending": len(pending), "settled": settled}


if __name__ == "__main__":
    result = settle_predictions()
    print(f"Settled {result['settled']}/{result['pending']} predictions")
