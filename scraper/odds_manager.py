"""オッズデータ統合マネージャー: OddsPortal/OddsAPI → DB保存"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from database.models import get_session, Match, Team
from database.repository import Repository

logger = logging.getLogger(__name__)


class OddsManager:
    """複数オッズソースからデータを収集してDBに保存"""

    def __init__(self):
        self._oddsportal_scraper = None
        self._repo = None

    def _get_repo(self) -> Repository:
        if self._repo is None:
            self._repo = Repository()
        return self._repo

    def _get_oddsportal(self):
        if self._oddsportal_scraper is None:
            from scraper.oddsportal import OddsPortalScraper
            self._oddsportal_scraper = OddsPortalScraper(headless=True)
        return self._oddsportal_scraper

    def scrape_oddsportal(
        self, league_code: str, target_date: Optional[date] = None
    ) -> int:
        """OddsPortalからオッズを取得してDB保存"""
        scraper = self._get_oddsportal()
        matches = scraper.scrape_league(league_code, target_date)

        if not matches:
            logger.info(f"No matches found on OddsPortal for {league_code}")
            return 0

        repo = self._get_repo()
        saved = 0

        for op_match in matches:
            try:
                # OddsPortal英語名 → 日本語名
                scraper = self._get_oddsportal()
                home_ja = scraper.resolve_team_name(op_match.home_team, league_code) or op_match.home_team
                away_ja = scraper.resolve_team_name(op_match.away_team, league_code) or op_match.away_team

                # DBの試合にマッチング
                db_match = self._find_db_match(
                    repo, home_ja, away_ja,
                    op_match.match_date, league_code,
                )
                if not db_match:
                    logger.debug(
                        f"No DB match for {op_match.home_team} vs {op_match.away_team} "
                        f"({op_match.match_date})"
                    )
                    continue

                # 平均オッズを保存
                if op_match.home_odds and op_match.away_odds:
                    repo.upsert_bookmaker_odds(
                        match_id=db_match.match_id,
                        bookmaker="average",
                        source="oddsportal",
                        home_odds=op_match.home_odds,
                        away_odds=op_match.away_odds,
                        draw_odds=op_match.draw_odds,
                    )
                    saved += 1

                # 個別ブックメーカーのオッズ
                for bm_odds in op_match.bookmaker_odds:
                    repo.upsert_bookmaker_odds(
                        match_id=db_match.match_id,
                        bookmaker=bm_odds["bookmaker"],
                        source="oddsportal",
                        home_odds=bm_odds.get("home_odds"),
                        away_odds=bm_odds.get("away_odds"),
                        draw_odds=bm_odds.get("draw_odds"),
                    )

            except Exception as e:
                logger.error(f"Failed to save odds for {op_match.home_team} vs {op_match.away_team}: {e}")

        repo.commit()
        logger.info(f"Saved odds for {saved} matches from OddsPortal ({league_code})")
        return saved

    def scrape_odds_api(self, league_code: str) -> int:
        """The Odds API からオッズを取得してDB保存"""
        from scraper.odds_api import get_odds

        all_odds = get_odds(league_code, markets="h2h,spreads,totals")
        if not all_odds:
            return 0

        repo = self._get_repo()
        saved = 0

        # 試合ごとにグループ化
        by_game: dict[str, list] = {}
        for o in all_odds:
            key = f"{o.home_team}|{o.away_team}"
            by_game.setdefault(key, []).append(o)

        for key, odds_list in by_game.items():
            home_en, away_en = key.split("|")
            # チーム名変換
            from scraper.team_name_map import find_ja_name
            from scraper.oddsportal import ODDSPORTAL_TEAM_MAP
            home_ja = find_ja_name(home_en) or ODDSPORTAL_TEAM_MAP.get(home_en)
            away_ja = find_ja_name(away_en) or ODDSPORTAL_TEAM_MAP.get(away_en)

            if not home_ja or not away_ja:
                logger.debug(f"Cannot resolve team names: {home_en} vs {away_en}")
                continue

            # DBマッチング (今日の予定試合)
            db_match = self._find_db_match(
                repo, home_ja, away_ja, date.today(), league_code,
            )
            if not db_match:
                continue

            for o in odds_list:
                repo.upsert_bookmaker_odds(
                    match_id=db_match.match_id,
                    bookmaker=o.bookmaker,
                    source="odds_api",
                    home_odds=o.home_odds,
                    away_odds=o.away_odds,
                    draw_odds=o.draw_odds,
                    home_spread=o.spread_home,
                    home_spread_odds=o.spread_home_odds,
                    away_spread_odds=o.spread_away_odds,
                )
                saved += 1

        repo.commit()
        logger.info(f"Saved {saved} bookmaker odds from Odds API ({league_code})")
        return saved

    def _find_db_match(
        self, repo: Repository, home_name: str, away_name: str,
        match_date: date, league_code: str,
    ) -> Optional[Match]:
        """DB内の試合を検索 (チーム名のファジーマッチ)"""
        session = repo.session

        # まず完全一致
        home_team = session.query(Team).filter(
            Team.league_code == league_code,
            Team.name == home_name,
        ).first()
        away_team = session.query(Team).filter(
            Team.league_code == league_code,
            Team.name == away_name,
        ).first()

        if home_team and away_team:
            match = session.query(Match).filter(
                Match.date == match_date,
                Match.home_team_id == home_team.team_id,
                Match.away_team_id == away_team.team_id,
            ).first()
            if match:
                return match

        # 部分一致で再検索
        home_team = session.query(Team).filter(
            Team.league_code == league_code,
            Team.name.contains(home_name),
        ).first()
        if not home_team:
            home_team = session.query(Team).filter(
                Team.league_code == league_code,
            ).all()
            home_team = next(
                (t for t in home_team if home_name in t.name or t.name in home_name),
                None,
            )

        away_team = session.query(Team).filter(
            Team.league_code == league_code,
            Team.name.contains(away_name),
        ).first()
        if not away_team:
            away_team = session.query(Team).filter(
                Team.league_code == league_code,
            ).all()
            away_team = next(
                (t for t in away_team if away_name in t.name or t.name in away_name),
                None,
            )

        if home_team and away_team:
            return session.query(Match).filter(
                Match.date == match_date,
                Match.home_team_id == home_team.team_id,
                Match.away_team_id == away_team.team_id,
            ).first()

        return None

    def close(self):
        if self._oddsportal_scraper:
            self._oddsportal_scraper.close()
        if self._repo:
            self._repo.close()
