"""ハンデの森スクレイパー (NPB/MLB)

HTML構造:
  section > div.game-detail2
    div.detail-single-studium-time > span (時刻, 球場)
    div.detail-card > div.detail-card-team (ホーム/ビジター名)
                    > div.detail-card-vs > span (スコア, 色でハンデ有利側表示)
    table.single-handi > td.single-handi-handi (ハンデ値, 片側にのみ数値)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

from bs4 import BeautifulSoup, Tag

from config.constants import SOURCES
from scraper.base_scraper import BaseScraper
from scraper.date_utils import format_date

logger = logging.getLogger(__name__)


@dataclass
class GameData:
    """スクレイピング結果の試合データ"""

    date: date
    time: Optional[str]
    venue: Optional[str]
    home_team: str
    away_team: str
    home_score: Optional[int]
    away_score: Optional[int]
    handicap_team: str  # ハンデを背負うチーム名
    handicap_value: float
    league_code: str
    home_pitcher: Optional[str] = None  # 野球: 予告先発
    away_pitcher: Optional[str] = None


class HandenomoriScraper(BaseScraper):
    """ハンデの森: NPB/MLB"""

    BASE_URL = SOURCES["handenomori"]["base_url"]

    def _build_url(self, section: str, d: date) -> str:
        return f"{self.BASE_URL}/{section}/{format_date(d)}"

    def scrape_date(self, league_code: str, d: date) -> list[GameData]:
        """指定日の全試合データを取得"""
        section_map = {"npb": "jpb", "mlb": "mlb"}
        section = section_map.get(league_code)
        if not section:
            raise ValueError(f"Unknown league: {league_code}")

        url = self._build_url(section, d)
        try:
            soup = self.fetch_and_parse(url)
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return []

        return self._parse_page(soup, d, league_code)

    def _parse_page(
        self, soup: BeautifulSoup, d: date, league_code: str
    ) -> list[GameData]:
        """ページ全体をパースして試合データリストを返す"""
        games = []
        container = soup.find("div", class_="index-container")
        if not container:
            logger.warning(f"No index-container found for {d}")
            return games

        sections = container.find_all("section")
        for section in sections:
            game = self._parse_game_section(section, d, league_code)
            if game:
                games.append(game)

        logger.info(f"[handenomori/{league_code}] {d}: {len(games)} games")
        return games

    def _parse_game_section(
        self, section: Tag, d: date, league_code: str
    ) -> Optional[GameData]:
        """1試合のsectionをパース"""
        detail = section.find("div", class_="game-detail2")
        if not detail:
            return None

        # 時刻・球場
        time_venue = detail.find("div", class_="detail-single-studium-time")
        game_time = None
        venue = None
        if time_venue:
            spans = time_venue.find_all("span")
            if len(spans) >= 1:
                game_time = spans[0].get_text(strip=True)
            if len(spans) >= 2:
                venue = spans[1].get_text(strip=True)

        # チーム名
        card = detail.find("div", class_="detail-card")
        if not card:
            return None

        team_divs = card.find_all("div", class_="detail-card-team")
        if len(team_divs) < 2:
            return None

        home_team = team_divs[0].get_text(strip=True)
        away_team = team_divs[1].get_text(strip=True)

        # スコア
        vs_div = card.find("div", class_="detail-card-vs")
        home_score = None
        away_score = None
        if vs_div:
            score_spans = vs_div.find_all("span")
            if len(score_spans) >= 2:
                try:
                    home_score = int(score_spans[0].get_text(strip=True))
                    away_score = int(score_spans[1].get_text(strip=True))
                except (ValueError, TypeError):
                    pass

        # 予告先発 (投手名)
        home_pitcher = None
        away_pitcher = None
        pitcher_div = detail.find("div", class_="detail-single-pitcher")
        if pitcher_div:
            pitcher_divs = pitcher_div.find_all("div", class_="detail-team-pitcher")
            if len(pitcher_divs) >= 2:
                home_pitcher = pitcher_divs[0].get_text(strip=True) or None
                away_pitcher = pitcher_divs[1].get_text(strip=True) or None

        # ハンデ値
        handi_table = detail.find("table", class_="single-handi")
        handicap_team = ""
        handicap_value = 0.0
        if handi_table:
            tds = handi_table.find_all("td", class_="single-handi-handi")
            if len(tds) >= 2:
                home_handi = tds[0].get_text(strip=True)
                away_handi = tds[1].get_text(strip=True)

                if home_handi:
                    # ホーム側にハンデ値 → ホームが有利（ハンデを背負う）
                    handicap_team = home_team
                    try:
                        handicap_value = float(home_handi)
                    except ValueError:
                        return None
                elif away_handi:
                    # アウェイ側にハンデ値 → アウェイが有利
                    handicap_team = away_team
                    try:
                        handicap_value = float(away_handi)
                    except ValueError:
                        return None
                else:
                    return None  # ハンデなし

        if not handicap_team:
            return None

        return GameData(
            date=d,
            time=game_time,
            venue=venue,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            handicap_team=handicap_team,
            handicap_value=handicap_value,
            league_code=league_code,
            home_pitcher=home_pitcher,
            away_pitcher=away_pitcher,
        )

    def get_available_dates(self, league_code: str) -> list[date]:
        """過去データのインデックスページから利用可能な日付一覧を取得"""
        section_map = {"npb": "jpb", "mlb": "mlb"}
        section = section_map.get(league_code)
        url = f"{self.BASE_URL}/{section}/"

        soup = self.fetch_and_parse(url)
        dates = []

        for link in soup.find_all("a"):
            href = link.get("href", "")
            # /jpb/20250801 のようなパターンを探す
            match = re.search(r"/(?:jpb|mlb)/(\d{8})", href)
            if match:
                date_str = match.group(1)
                try:
                    from scraper.date_utils import parse_date

                    dates.append(parse_date(date_str))
                except ValueError:
                    pass

        return sorted(set(dates))
