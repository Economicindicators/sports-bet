"""選手統計の算出 — 試合データからPlayerStatsテーブルを更新"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from database.models import Match, Player, Team, get_session
from database.repository import Repository

logger = logging.getLogger(__name__)


def compute_player_stats(
    sport_code: str,
    season: Optional[str] = None,
    session: Optional[Session] = None,
) -> int:
    """
    指定スポーツの選手統計を試合データから算出してplayer_statsに保存。

    Args:
        sport_code: "baseball", "soccer", "basketball"
        season: シーズン文字列 (None=現在年)

    Returns:
        更新した統計レコード数
    """
    sess = session or get_session()
    repo = Repository(sess)

    if season is None:
        season = str(date.today().year)

    if sport_code == "baseball":
        count = _compute_baseball_pitcher_stats(repo, sess, season)
    elif sport_code == "soccer":
        count = _compute_team_stats(repo, sess, sport_code, season)
    elif sport_code == "basketball":
        count = _compute_team_stats(repo, sess, sport_code, season)
    else:
        logger.warning(f"Unknown sport_code: {sport_code}")
        return 0

    repo.commit()
    if not session:
        sess.close()

    logger.info(f"Computed {count} player stats for {sport_code}/{season}")
    return count


def _compute_baseball_pitcher_stats(repo: Repository, sess: Session, season: str) -> int:
    """野球投手の統計を算出"""
    count = 0

    # home_pitcher_idまたはaway_pitcher_idが設定されている全投手を取得
    pitcher_ids = set()
    rows = (
        sess.query(Match.home_pitcher_id)
        .filter(Match.sport_code == "baseball", Match.status == "finished")
        .filter(Match.home_pitcher_id.isnot(None))
        .distinct()
        .all()
    )
    pitcher_ids.update(r[0] for r in rows)

    rows = (
        sess.query(Match.away_pitcher_id)
        .filter(Match.sport_code == "baseball", Match.status == "finished")
        .filter(Match.away_pitcher_id.isnot(None))
        .distinct()
        .all()
    )
    pitcher_ids.update(r[0] for r in rows)

    for pid in pitcher_ids:
        # ホーム先発試合
        home_games = (
            sess.query(Match)
            .filter(Match.home_pitcher_id == pid, Match.status == "finished")
            .all()
        )
        # アウェイ先発試合
        away_games = (
            sess.query(Match)
            .filter(Match.away_pitcher_id == pid, Match.status == "finished")
            .all()
        )

        total_starts = len(home_games) + len(away_games)
        if total_starts == 0:
            continue

        # 勝率
        wins = 0
        total_runs_allowed = 0
        for m in home_games:
            if m.home_score is not None and m.away_score is not None:
                if m.home_score > m.away_score:
                    wins += 1
                total_runs_allowed += m.away_score  # 投手目線: 被得点
        for m in away_games:
            if m.home_score is not None and m.away_score is not None:
                if m.away_score > m.home_score:
                    wins += 1
                total_runs_allowed += m.home_score

        win_rate = wins / total_starts if total_starts > 0 else 0.5
        era = total_runs_allowed / total_starts if total_starts > 0 else 4.0

        # 直近5試合のフォーム
        all_games = sorted(home_games + away_games, key=lambda m: m.date, reverse=True)
        recent = all_games[:5]
        recent_wins = 0
        for m in recent:
            if m.home_score is not None and m.away_score is not None:
                is_home = m.home_pitcher_id == pid
                if is_home and m.home_score > m.away_score:
                    recent_wins += 1
                elif not is_home and m.away_score > m.home_score:
                    recent_wins += 1
        recent_form = recent_wins / len(recent) if recent else 0.5

        repo.upsert_player_stat(pid, season, "win_rate", round(win_rate, 4), total_starts)
        repo.upsert_player_stat(pid, season, "era", round(era, 2), total_starts)
        repo.upsert_player_stat(pid, season, "starts", float(total_starts), total_starts)
        repo.upsert_player_stat(pid, season, "recent_form", round(recent_form, 4), len(recent))
        count += 4

    return count


def _compute_team_stats(repo: Repository, sess: Session, sport_code: str, season: str) -> int:
    """サッカー/バスケのチームレベル統計を算出"""
    count = 0

    # 全チームを取得（そのスポーツに属するもの）
    from database.models import League
    league_codes = [
        r[0] for r in sess.query(League.league_code).filter_by(sport_code=sport_code).all()
    ]
    teams = sess.query(Team).filter(Team.league_code.in_(league_codes)).all()

    for team in teams:
        # ホーム試合
        home_matches = (
            sess.query(Match)
            .filter(
                Match.home_team_id == team.team_id,
                Match.status == "finished",
                Match.sport_code == sport_code,
            )
            .all()
        )
        # アウェイ試合
        away_matches = (
            sess.query(Match)
            .filter(
                Match.away_team_id == team.team_id,
                Match.status == "finished",
                Match.sport_code == sport_code,
            )
            .all()
        )

        total = len(home_matches) + len(away_matches)
        if total == 0:
            continue

        # チーム用の仮想Playerを作成 (チーム全体の統計を格納)
        player = repo.upsert_player(
            name=f"[TEAM] {team.name}",
            league_code=team.league_code,
            team_id=team.team_id,
            position="team",
        )

        # 得点/失点集計
        scored = 0
        conceded = 0
        wins = 0
        clean_sheets = 0

        for m in home_matches:
            if m.home_score is not None and m.away_score is not None:
                scored += m.home_score
                conceded += m.away_score
                if m.home_score > m.away_score:
                    wins += 1
                if m.away_score == 0:
                    clean_sheets += 1

        for m in away_matches:
            if m.home_score is not None and m.away_score is not None:
                scored += m.away_score
                conceded += m.home_score
                if m.away_score > m.home_score:
                    wins += 1
                if m.home_score == 0:
                    clean_sheets += 1

        avg_scored = scored / total
        avg_conceded = conceded / total
        win_rate = wins / total
        cs_rate = clean_sheets / total

        repo.upsert_player_stat(player.player_id, season, "avg_scored", round(avg_scored, 2), total)
        repo.upsert_player_stat(player.player_id, season, "avg_conceded", round(avg_conceded, 2), total)
        repo.upsert_player_stat(player.player_id, season, "win_rate", round(win_rate, 4), total)
        repo.upsert_player_stat(player.player_id, season, "clean_sheet_rate", round(cs_rate, 4), total)
        count += 4

        # 直近5試合フォーム
        all_matches = sorted(home_matches + away_matches, key=lambda m: m.date, reverse=True)
        recent = all_matches[:5]
        recent_wins = 0
        recent_scored = 0
        recent_conceded = 0
        for m in recent:
            if m.home_score is not None and m.away_score is not None:
                is_home = m.home_team_id == team.team_id
                if is_home:
                    recent_scored += m.home_score
                    recent_conceded += m.away_score
                    if m.home_score > m.away_score:
                        recent_wins += 1
                else:
                    recent_scored += m.away_score
                    recent_conceded += m.home_score
                    if m.away_score > m.home_score:
                        recent_wins += 1

        if recent:
            repo.upsert_player_stat(player.player_id, season, "recent_form", round(recent_wins / len(recent), 4), len(recent))
            repo.upsert_player_stat(player.player_id, season, "recent_avg_scored", round(recent_scored / len(recent), 2), len(recent))
            repo.upsert_player_stat(player.player_id, season, "recent_avg_conceded", round(recent_conceded / len(recent), 2), len(recent))
            count += 3

    return count


def export_player_stats_to_turso(season: Optional[str] = None) -> int:
    """PlayerStatsデータをTursoに送信"""
    import subprocess
    import tempfile

    sess = get_session()
    if season is None:
        season = str(date.today().year)

    from database.models import PlayerStats
    stats = (
        sess.query(PlayerStats, Player)
        .join(Player, PlayerStats.player_id == Player.player_id)
        .filter(PlayerStats.season == season)
        .all()
    )

    if not stats:
        sess.close()
        return 0

    lines = [
        "CREATE TABLE IF NOT EXISTS player_stats ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  player_name TEXT NOT NULL,"
        "  team_name TEXT,"
        "  league_code TEXT NOT NULL,"
        "  position TEXT,"
        "  season TEXT NOT NULL,"
        "  stat_type TEXT NOT NULL,"
        "  value REAL NOT NULL,"
        "  sample_size INTEGER NOT NULL,"
        "  UNIQUE(player_name, league_code, season, stat_type)"
        ");"
    ]

    for ps, player in stats:
        team = sess.get(Team, player.team_id) if player.team_id else None
        esc = lambda s: str(s).replace("'", "''")
        lines.append(
            f"INSERT OR REPLACE INTO player_stats "
            f"(player_name, team_name, league_code, position, season, stat_type, value, sample_size) VALUES "
            f"('{esc(player.name)}', '{esc(team.name if team else '')}', '{esc(player.league_code)}', "
            f"'{esc(player.position or '')}', '{season}', '{esc(ps.stat_type)}', {ps.value}, {ps.sample_size});"
        )

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
            return 0
    except Exception as e:
        logger.error(f"Turso export failed: {e}")
        return 0
    finally:
        sess.close()

    logger.info(f"Exported {len(stats)} player stats to Turso")
    return len(stats)
