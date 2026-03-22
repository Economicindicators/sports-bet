"""NBA Advanced Stats (stats.nba.com API)

チーム別の攻守効率・ペース等を取得:
  - Offensive Rating (OffRtg)
  - Defensive Rating (DefRtg)
  - Net Rating
  - Pace
  - AST%, TOV%
  - ORB%, DRB% (別エンドポイント)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import requests

from database.models import NbaTeamStats, Team, get_session

logger = logging.getLogger(__name__)

# NBA.com チーム名 → DB名マッピング
NBA_TEAM_MAP = {
    "Atlanta Hawks": "ホークス", "Boston Celtics": "セルティックス",
    "Brooklyn Nets": "ネッツ", "Charlotte Hornets": "ホーネッツ",
    "Chicago Bulls": "ブルズ", "Cleveland Cavaliers": "キャバリアーズ",
    "Dallas Mavericks": "マーベリックス", "Denver Nuggets": "ナゲッツ",
    "Detroit Pistons": "ピストンズ", "Golden State Warriors": "ウォリアーズ",
    "Houston Rockets": "ロケッツ", "Indiana Pacers": "ペイサーズ",
    "LA Clippers": "クリッパーズ", "Los Angeles Lakers": "レイカーズ",
    "Memphis Grizzlies": "グリズリーズ", "Miami Heat": "ヒート",
    "Milwaukee Bucks": "バックス", "Minnesota Timberwolves": "ティンバーウルブズ",
    "New Orleans Pelicans": "ペリカンズ", "New York Knicks": "ニックス",
    "Oklahoma City Thunder": "サンダー", "Orlando Magic": "マジック",
    "Philadelphia 76ers": "シクサーズ", "Phoenix Suns": "サンズ",
    "Portland Trail Blazers": "ブレイザーズ", "Sacramento Kings": "キングス",
    "San Antonio Spurs": "スパーズ", "Toronto Raptors": "ラプターズ",
    "Utah Jazz": "ジャズ", "Washington Wizards": "ウィザーズ",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json",
}


class NbaStatsScraper:
    """stats.nba.com API からチームアドバンスドスタッツを取得"""

    BASE_URL = "https://stats.nba.com/stats/leaguedashteamstats"

    def scrape_season(self, season: str = "2025-26") -> list[dict]:
        """
        シーズンのチームアドバンスドスタッツを取得。

        Args:
            season: シーズン文字列 (例: "2025-26")
        """
        results = []

        # Advanced Stats (OffRtg, DefRtg, NetRtg, Pace, AST%, TOV%)
        advanced = self._fetch_stats(season, "Advanced")
        if advanced:
            headers = advanced["resultSets"][0]["headers"]
            rows = advanced["resultSets"][0]["rowSet"]

            stat_cols = {
                "OFF_RATING": "off_rtg",
                "DEF_RATING": "def_rtg",
                "NET_RATING": "net_rtg",
                "PACE": "pace",
                "AST_PCT": "ast_pct",
                "AST_TO": "ast_to_ratio",
            }

            name_idx = headers.index("TEAM_NAME")
            for row in rows:
                team_name = row[name_idx]
                for col, stat_type in stat_cols.items():
                    if col in headers:
                        idx = headers.index(col)
                        val = row[idx]
                        if val is not None:
                            results.append({
                                "team_name": team_name,
                                "season": season,
                                "stat_type": stat_type,
                                "value": float(val),
                            })

        time.sleep(1)

        # Four Factors (ORB%, DRB%, TOV%)
        four_factors = self._fetch_stats(season, "Four Factors")
        if four_factors:
            headers = four_factors["resultSets"][0]["headers"]
            rows = four_factors["resultSets"][0]["rowSet"]

            stat_cols = {
                "OREB_PCT": "orb_pct",
                "DREB_PCT": "drb_pct",
                "TM_TOV_PCT": "tov_pct",
            }

            name_idx = headers.index("TEAM_NAME")
            for row in rows:
                team_name = row[name_idx]
                for col, stat_type in stat_cols.items():
                    if col in headers:
                        idx = headers.index(col)
                        val = row[idx]
                        if val is not None:
                            results.append({
                                "team_name": team_name,
                                "season": season,
                                "stat_type": stat_type,
                                "value": float(val),
                            })

        teams = set(r["team_name"] for r in results)
        logger.info(f"[nba_stats] {season}: {len(results)} entries from {len(teams)} teams")
        return results

    def _fetch_stats(self, season: str, measure_type: str) -> dict | None:
        """NBA.com stats API からデータ取得"""
        params = {
            "Conference": "", "DateFrom": "", "DateTo": "",
            "Division": "", "GameScope": "", "GameSegment": "",
            "Height": "", "ISTRound": "", "LastNGames": 0,
            "LeagueID": "00", "Location": "",
            "MeasureType": measure_type,
            "Month": 0, "OpponentTeamID": 0, "Outcome": "",
            "PORound": 0, "PaceAdjust": "N", "PerMode": "PerGame",
            "Period": 0, "PlayerExperience": "", "PlayerPosition": "",
            "PlusMinus": "N", "Rank": "N",
            "Season": season,
            "SeasonSegment": "", "SeasonType": "Regular Season",
            "ShotClockRange": "", "StarterBench": "",
            "TeamID": 0, "TwoWay": 0,
            "VsConference": "", "VsDivision": "",
        }
        try:
            r = requests.get(self.BASE_URL, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"NBA stats API error ({measure_type}): {e}")
            return None

    def save_to_db(self, stats: list[dict]) -> int:
        """スタッツをDBに保存"""
        session = get_session()
        saved = 0

        teams = session.query(Team).filter(Team.league_code == "nba").all()
        team_map = {t.name: t.team_id for t in teams}

        for entry in stats:
            jp_name = NBA_TEAM_MAP.get(entry["team_name"], entry["team_name"])
            team_id = team_map.get(jp_name)

            if not team_id:
                logger.debug(f"Team not found: {entry['team_name']} ({jp_name})")
                continue

            existing = session.query(NbaTeamStats).filter_by(
                team_id=team_id,
                season=entry["season"],
                stat_type=entry["stat_type"],
            ).first()

            if existing:
                existing.value = entry["value"]
                existing.updated_at = datetime.utcnow()
            else:
                session.add(NbaTeamStats(
                    team_id=team_id,
                    season=entry["season"],
                    stat_type=entry["stat_type"],
                    value=entry["value"],
                ))
            saved += 1

        session.commit()
        session.close()
        logger.info(f"Saved {saved} NBA team stats")
        return saved
