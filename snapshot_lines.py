"""ハンデライン スナップショット取得 (試合開始時刻ベース)

各試合の開始時刻から逆算して3回スナップショットを取得:
  opening = 試合6時間前
  midday  = 試合3時間前
  closing = 試合1時間前

10分間隔でチェックし、該当タイミングの試合があればスクレイプ→記録。

Usage:
    python snapshot_lines.py              # デーモンモード (10分間隔で常駐)
    python snapshot_lines.py --once       # 1回だけ実行して終了
    python snapshot_lines.py --type opening --once  # 全試合を強制的にopeningで記録
"""

from __future__ import annotations

import argparse
import logging
import time as time_mod
from datetime import date, datetime, time, timedelta
from typing import Optional

from config.constants import ALL_LEAGUES
from database.models import init_db, get_session, Match, HandicapData
from database.repository import Repository
from scraper.manager import ScrapeManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# スナップショットのタイミング定義 (試合開始N時間前)
SNAPSHOT_SCHEDULE = [
    ("opening", 6),  # 6時間前
    ("midday", 3),   # 3時間前
    ("closing", 1),  # 1時間前
]

# 許容誤差 (分): この範囲内ならスナップショット対象
WINDOW_MINUTES = 30

CHECK_INTERVAL = 600  # 10分間隔


def get_match_kickoff(match: Match) -> Optional[datetime]:
    """試合のキックオフ日時を取得 (JST)"""
    if match.time is None:
        return None
    return datetime.combine(match.date, match.time)


def determine_snapshot_type_for_match(match: Match, now: datetime) -> Optional[str]:
    """現在時刻と試合開始時刻から、取るべきスナップショットタイプを判定。
    該当なしならNone。"""
    kickoff = get_match_kickoff(match)
    if kickoff is None:
        return None

    for snap_type, hours_before in SNAPSHOT_SCHEDULE:
        target_time = kickoff - timedelta(hours=hours_before)
        diff_minutes = abs((now - target_time).total_seconds()) / 60

        if diff_minutes <= WINDOW_MINUTES:
            return snap_type

    return None


def scrape_upcoming(target_date: date) -> dict:
    """今日＋明日のデータをスクレイプ"""
    tomorrow = target_date + timedelta(days=1)
    manager = ScrapeManager(use_cache=False)
    results = {}

    for league in ALL_LEAGUES:
        for d in [target_date, tomorrow]:
            try:
                count = manager.scrape_and_save(league, d)
                if count > 0:
                    results[f"{league}/{d}"] = count
                    logger.info(f"Scraped {league}/{d}: {count} matches")
            except Exception as e:
                logger.warning(f"Scrape failed {league}/{d}: {e}")

    return results


def take_snapshots_smart(force_type: Optional[str] = None) -> dict:
    """試合開始時刻ベースでスナップショットを取得。

    Args:
        force_type: 指定すると全試合をこのタイプで強制記録
    """
    now = datetime.now()
    td = now.date()
    tomorrow = td + timedelta(days=1)

    # Step 1: スクレイプして最新ハンデ値を取得
    scrape_results = scrape_upcoming(td)

    # Step 2: 今日＋明日のscheduled試合をチェック
    session = get_session()
    repo = Repository(session)

    matches = (
        session.query(Match)
        .filter(Match.date.in_([td, tomorrow]), Match.status == "scheduled")
        .all()
    )

    snapshot_count = 0
    details = []

    for match in matches:
        hd = session.query(HandicapData).filter_by(match_id=match.match_id).first()
        if not hd:
            continue

        if force_type:
            snap_type = force_type
        else:
            snap_type = determine_snapshot_type_for_match(match, now)

        if snap_type is None:
            continue

        # 既に同じスナップショットがあればスキップ（force以外）
        from database.models import HandicapSnapshot
        existing = (
            session.query(HandicapSnapshot)
            .filter_by(match_id=match.match_id, snapshot_type=snap_type)
            .first()
        )
        if existing and not force_type:
            continue

        home = session.get(type(hd.handicap_team), match.home_team_id)
        away = session.get(type(hd.handicap_team), match.away_team_id)

        repo.upsert_snapshot(
            match_id=match.match_id,
            handicap_team_id=hd.handicap_team_id,
            handicap_value=hd.handicap_value,
            snapshot_type=snap_type,
            handicap_display=hd.handicap_display,
        )
        snapshot_count += 1

        kickoff = get_match_kickoff(match)
        kickoff_str = kickoff.strftime("%H:%M") if kickoff else "?"
        logger.info(
            f"  [{snap_type}] match:{match.match_id} "
            f"kickoff:{kickoff_str} handi:{hd.handicap_value}"
        )
        details.append({
            "match_id": match.match_id,
            "type": snap_type,
            "kickoff": kickoff_str,
            "value": hd.handicap_value,
        })

    repo.commit()
    session.close()

    logger.info(f"Saved {snapshot_count} snapshots")
    return {
        "snapshots": snapshot_count,
        "details": details,
        "scrape": scrape_results,
    }


def export_snapshots_to_turso(target_date: Optional[date] = None) -> int:
    """スナップショットデータをTursoにエクスポート"""
    import asyncio
    import os
    from libsql_client import create_client

    td = target_date or date.today()
    tomorrow = td + timedelta(days=1)

    session = get_session()
    from database.models import HandicapSnapshot, Team

    snapshots = (
        session.query(HandicapSnapshot, Match, Team)
        .join(Match, HandicapSnapshot.match_id == Match.match_id)
        .join(Team, HandicapSnapshot.handicap_team_id == Team.team_id)
        .filter(Match.date.in_([td, tomorrow]))
        .all()
    )

    if not snapshots:
        session.close()
        return 0

    async def _upload():
        turso_url = os.environ.get("TURSO_DATABASE_URL", "")
        turso_token = os.environ.get("TURSO_AUTH_TOKEN", "")

        for env_name in [".env.local", ".env"]:
            env_path = os.path.expanduser(f"~/sports-bet-web/{env_name}")
            if not os.path.exists(env_path):
                continue
            for line in open(env_path):
                line = line.strip()
                if line.startswith("TURSO_DATABASE_URL=") and not turso_url:
                    turso_url = line.split("=", 1)[1].strip().strip('"')
                elif line.startswith("TURSO_AUTH_TOKEN=") and not turso_token:
                    turso_token = line.split("=", 1)[1].strip().strip('"')

        turso_url = turso_url.replace("libsql://", "https://")
        client = create_client(url=turso_url, auth_token=turso_token)

        await client.execute(
            "CREATE TABLE IF NOT EXISTS handicap_snapshots ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  match_id INTEGER NOT NULL,"
            "  home_team TEXT,"
            "  away_team TEXT,"
            "  handicap_team TEXT,"
            "  handicap_value REAL,"
            "  handicap_display TEXT,"
            "  snapshot_type TEXT NOT NULL,"
            "  match_date TEXT,"
            "  league TEXT,"
            "  captured_at TEXT,"
            "  UNIQUE(match_id, snapshot_type)"
            ")"
        )

        batch = []
        for snap, match, team in snapshots:
            home = session.get(Team, match.home_team_id)
            away = session.get(Team, match.away_team_id)
            esc = lambda s: str(s).replace("'", "''")
            sql = (
                f"INSERT OR REPLACE INTO handicap_snapshots "
                f"(match_id, home_team, away_team, handicap_team, handicap_value, "
                f"handicap_display, snapshot_type, match_date, league, captured_at) VALUES "
                f"({match.match_id}, '{esc(home.name if home else '')}', "
                f"'{esc(away.name if away else '')}', '{esc(team.name)}', "
                f"{snap.handicap_value}, '{esc(snap.handicap_display or '')}', "
                f"'{snap.snapshot_type}', '{match.date}', '{match.league_code}', "
                f"'{snap.captured_at}')"
            )
            batch.append(sql)

        for sql in batch:
            await client.execute(sql)

        await client.close()
        return len(batch)

    count = asyncio.run(_upload())
    session.close()
    logger.info(f"Exported {count} snapshots to Turso")
    return count


def run_daemon():
    """10分間隔で常駐し、試合開始時刻ベースでスナップショットを取得"""
    logger.info(f"Snapshot daemon started (interval={CHECK_INTERVAL}s)")
    logger.info(f"Schedule: opening=-6h, midday=-3h, closing=-1h (window=±{WINDOW_MINUTES}min)")

    while True:
        try:
            result = take_snapshots_smart()
            if result["snapshots"] > 0:
                exported = export_snapshots_to_turso()
                logger.info(f"Cycle done: {result['snapshots']} snapshots, {exported} exported")
            else:
                logger.debug("No snapshots needed this cycle")
        except Exception as e:
            logger.exception(f"Daemon cycle error: {e}")

        time_mod.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--type", choices=["opening", "midday", "closing"], default=None,
                        help="Force snapshot type for all matches")
    parser.add_argument("--export", action="store_true", help="Export to Turso (with --once)")
    args = parser.parse_args()

    init_db()

    if args.once:
        result = take_snapshots_smart(force_type=args.type)
        print(f"Result: {result}")
        if args.export or result["snapshots"] > 0:
            count = export_snapshots_to_turso()
            print(f"Exported {count} snapshots to Turso")
    else:
        run_daemon()
