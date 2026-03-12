"""スクレイピング統合マネージャー: GameData → DB保存"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from betting.handicap_resolver import resolve_handicap
from database.models import get_session
from database.repository import Repository
from scraper.handenomori import GameData, HandenomoriScraper
from scraper.football_hande import FootballHandeScraper
from scraper.b_handicap import BHandicapScraper
from scraper.date_utils import date_range
from config.constants import LEAGUE_TO_SPORT

logger = logging.getLogger(__name__)

# リーグ → スクレイパークラス
LEAGUE_TO_SCRAPER = {
    "npb": HandenomoriScraper,
    "mlb": HandenomoriScraper,
    "wbc": HandenomoriScraper,
    "jleague": FootballHandeScraper,
    "premier": FootballHandeScraper,
    "laliga": FootballHandeScraper,
    "seriea": FootballHandeScraper,
    "bundesliga": FootballHandeScraper,
    "ligue1": FootballHandeScraper,
    "eredivisie": FootballHandeScraper,
    "cl": FootballHandeScraper,
    "nba": BHandicapScraper,
    "bleague": BHandicapScraper,
}


class ScrapeManager:
    """スクレイピング → DB保存の統合管理"""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._scrapers: dict = {}

    def _get_scraper(self, league_code: str):
        """リーグに対応するスクレイパーを取得（キャッシュ）"""
        scraper_cls = LEAGUE_TO_SCRAPER.get(league_code)
        if not scraper_cls:
            raise ValueError(f"Unknown league: {league_code}")

        key = scraper_cls.__name__
        if key not in self._scrapers:
            self._scrapers[key] = scraper_cls(use_cache=self.use_cache)
        return self._scrapers[key]

    def scrape_and_save(
        self, league_code: str, d: date
    ) -> int:
        """指定日・リーグのデータをスクレイプしてDBに保存"""
        scraper = self._get_scraper(league_code)
        games = scraper.scrape_date(league_code, d)

        if not games:
            return 0

        sport_code = LEAGUE_TO_SPORT[league_code]
        repo = Repository()

        saved = 0
        for game in games:
            try:
                saved += self._save_game(repo, game, sport_code)
            except Exception as e:
                logger.error(f"Failed to save game {game.home_team} vs {game.away_team}: {e}")

        repo.commit()
        repo.close()
        return saved

    def scrape_range(
        self, league_code: str, start: date, end: date
    ) -> int:
        """日付範囲をスクレイプ"""
        total = 0
        for d in date_range(start, end):
            count = self.scrape_and_save(league_code, d)
            total += count
        return total

    def _save_game(self, repo: Repository, game: GameData, sport_code: str) -> int:
        """1試合をDBに保存"""
        # チームをupsert
        home_team = repo.upsert_team(game.home_team, game.league_code, game.venue)
        away_team = repo.upsert_team(game.away_team, game.league_code)

        # 投手をupsert (野球のみ)
        home_pitcher_id = None
        away_pitcher_id = None
        if game.home_pitcher:
            pitcher = repo.upsert_player(
                game.home_pitcher, game.league_code, home_team.team_id, "pitcher"
            )
            home_pitcher_id = pitcher.player_id
        if game.away_pitcher:
            pitcher = repo.upsert_player(
                game.away_pitcher, game.league_code, away_team.team_id, "pitcher"
            )
            away_pitcher_id = pitcher.player_id

        # ステータス判定
        status = "finished" if game.home_score is not None else "scheduled"

        # 試合時刻をtime型に変換
        match_time = None
        if game.time:
            try:
                from datetime import time as dt_time
                parts = game.time.split(":")
                match_time = dt_time(int(parts[0]), int(parts[1]))
            except (ValueError, IndexError):
                pass

        # 試合をupsert
        match = repo.upsert_match(
            sport_code=sport_code,
            league_code=game.league_code,
            match_date=game.date,
            home_team_id=home_team.team_id,
            away_team_id=away_team.team_id,
            home_score=game.home_score,
            away_score=game.away_score,
            match_time=match_time,
            venue=game.venue,
            status=status,
            home_pitcher_id=home_pitcher_id,
            away_pitcher_id=away_pitcher_id,
        )

        # ハンデチームのIDを特定
        if game.handicap_team == game.home_team:
            handicap_team_id = home_team.team_id
        else:
            handicap_team_id = away_team.team_id

        # ハンデ結果を計算 (スコアがある場合)
        result_type = None
        payout_rate = None
        if game.home_score is not None and game.away_score is not None:
            if game.handicap_team == game.home_team:
                fav_score = game.home_score
                underdog_score = game.away_score
            else:
                fav_score = game.away_score
                underdog_score = game.home_score

            result = resolve_handicap(fav_score, underdog_score, game.handicap_value)
            result_type = result.result_type
            payout_rate = result.payout_rate

        # ハンデデータをupsert
        repo.upsert_handicap(
            match_id=match.match_id,
            handicap_team_id=handicap_team_id,
            handicap_value=game.handicap_value,
            result_type=result_type,
            payout_rate=payout_rate,
            handicap_display=getattr(game, 'handicap_display', None),
        )

        return 1
