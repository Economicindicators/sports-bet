"""ハンディマンスクレイパー (NBA/Bリーグ)

HTML構造:
  table.single-gamedata
    tr[0]: td.stadium-data (球場名, TIP-OFF時刻)
    tr[1]: td.team-data (ホーム + span.handhi)
           td.score-data (span.h-score, span.a-score)
           td.team-data (アウェイ + span.handhi)

ハンデ値: span.handhi にテキストで入る (片側のみ)
スコア: span.h-score / span.a-score
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Optional

from bs4 import BeautifulSoup, Tag

from config.constants import SOURCES
from scraper.base_scraper import BaseScraper
from scraper.date_utils import format_date
from scraper.handenomori import GameData

logger = logging.getLogger(__name__)


class BHandicapScraper(BaseScraper):
    """ハンディマン: NBA/Bリーグ"""

    BASE_URL = SOURCES["b_handicap"]["base_url"]

    def _build_url(self, section: str, d: date) -> str:
        return f"{self.BASE_URL}/{section}/{format_date(d)}"

    def scrape_date(self, league_code: str, d: date) -> list[GameData]:
        """指定日の全試合データを取得"""
        url = self._build_url(league_code, d)
        try:
            soup = self.fetch_and_parse(url)
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return []

        return self._parse_page(soup, d, league_code)

    def _parse_page(
        self, soup: BeautifulSoup, d: date, league_code: str
    ) -> list[GameData]:
        """ページ全体をパース"""
        games = []
        tables = soup.find_all("table", class_="single-gamedata")

        for table in tables:
            game = self._parse_game_table(table, d, league_code)
            if game:
                games.append(game)

        logger.info(f"[b_handicap/{league_code}] {d}: {len(games)} games")
        return games

    def _parse_game_table(
        self, table: Tag, d: date, league_code: str
    ) -> Optional[GameData]:
        """1試合のtableをパース"""
        rows = table.find_all("tr")
        if len(rows) < 2:
            return None

        # Row 0: スタジアム・時刻
        stadium_td = rows[0].find("td", class_="stadium-data")
        game_time = None
        venue = None
        if stadium_td:
            # スタジアム名
            stadium_span = stadium_td.find("span", class_="stadium-name")
            if stadium_span:
                # spanの後のテキストがスタジアム名
                venue_text = stadium_span.next_sibling
                if venue_text:
                    venue = str(venue_text).strip()

            # TIP-OFF時刻
            tipoff_span = stadium_td.find("span", class_="tipoff-time")
            if tipoff_span:
                time_text = tipoff_span.next_sibling
                if time_text:
                    time_match = re.search(r"(\d{1,2}:\d{2})", str(time_text))
                    if time_match:
                        game_time = time_match.group(1)

        # Row 1: チーム・スコア・ハンデ
        team_tds = rows[1].find_all("td", class_="team-data")
        score_td = rows[1].find("td", class_="score-data")

        if len(team_tds) < 2:
            return None

        # ホームチーム
        home_td = team_tds[0]
        away_td = team_tds[1]

        home_team = self._extract_team_name(home_td)
        away_team = self._extract_team_name(away_td)

        if not home_team or not away_team:
            return None

        # ハンデ値
        home_handi = home_td.find("span", class_="handhi")
        away_handi = away_td.find("span", class_="handhi")

        handicap_team = ""
        handicap_value = 0.0

        home_handi_text = home_handi.get_text(strip=True) if home_handi else ""
        away_handi_text = away_handi.get_text(strip=True) if away_handi else ""

        if home_handi_text:
            handicap_team = home_team
            try:
                handicap_value = float(home_handi_text)
            except ValueError:
                return None
        elif away_handi_text:
            handicap_team = away_team
            try:
                handicap_value = float(away_handi_text)
            except ValueError:
                return None
        else:
            # ハンデなし（まだ発表されていない）
            return None

        # スコア
        home_score = None
        away_score = None
        if score_td:
            h_score_span = score_td.find("span", class_="h-score")
            a_score_span = score_td.find("span", class_="a-score")

            if h_score_span:
                try:
                    home_score = int(h_score_span.get_text(strip=True))
                except (ValueError, TypeError):
                    pass
            if a_score_span:
                try:
                    away_score = int(a_score_span.get_text(strip=True))
                except (ValueError, TypeError):
                    pass

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
        )

    def _extract_team_name(self, td: Tag) -> str:
        """team-data tdからチーム名を取得 (span要素を除く)"""
        # captionとhandhiのspanを除いたテキスト
        text_parts = []
        for child in td.children:
            if hasattr(child, "name") and child.name == "span":
                continue
            text = str(child).strip()
            if text:
                text_parts.append(text)

        name = " ".join(text_parts).strip()
        return name

    def get_available_dates(self, league_code: str) -> list[date]:
        """利用可能な日付一覧を取得"""
        url = f"{self.BASE_URL}/{league_code}/"
        soup = self.fetch_and_parse(url)
        dates = []

        for link in soup.find_all("a"):
            href = link.get("href", "")
            match = re.search(rf"/{league_code}/(\d{{8}})", href)
            if match:
                from scraper.date_utils import parse_date

                try:
                    dates.append(parse_date(match.group(1)))
                except ValueError:
                    pass

        return sorted(set(dates))
