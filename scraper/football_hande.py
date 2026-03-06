"""ハンデの嵐スクレイパー (サッカー)

HTML構造:
  ul.index-game-list > li > table.index-handi-table
    tr[0]: th (節, 時刻, スタジアム)
    tr[1]: td (ハンデ結果) + td "HDCP" + td (ハンデ結果)
           - ハンデテキストがある側 = ハンデを背負うチーム (有利判定チーム)
           - win-cellクラス = 試合の勝者 (ハンデ勝負の勝者ではない)
    tr[2]: td (ホームチーム+スコア) + td (vs) + td (アウェイチーム+スコア)

ハンデ表記の解読:
  "0半3" → ハンデ0.5  (「半」= +0.5)
  "0/4"  → ハンデ0    (「/」= 整数ハンデ)
  "1.8"  → ハンデ1    (「.」= 整数ハンデ)
  "1半5" → ハンデ1.5
  "0"    → ハンデ0    (単独数字)
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
from scraper.handenomori import GameData

logger = logging.getLogger(__name__)


def parse_hande_result(text: str) -> Optional[tuple[float, str]]:
    """
    サッカーハンデ表記をパースする。

    Args:
        text: "0半3", "0/4", "1.8", "0" 等

    Returns:
        (handicap_value, result_code) or None

    ハンデ値の解読:
        "半" = +0.5, "/" or "." = 整数ハンデ
    """
    if not text or text.strip() == "":
        return None

    text = text.strip()

    # パターン1: 数字 + (半|/|.) + 数字
    match = re.match(r"(\d+)(半|/|\.)(\d+)", text)
    if match:
        base = int(match.group(1))
        separator = match.group(2)
        result_code = match.group(3)

        if separator == "半":
            handicap_value = base + 0.5
        else:  # "/" or "."
            handicap_value = float(base)

        return handicap_value, result_code

    # パターン2: 単独数字 ("0", "1", "2" 等)
    match = re.match(r"^(\d+)$", text)
    if match:
        return float(match.group(1)), ""

    return None


class FootballHandeScraper(BaseScraper):
    """ハンデの嵐: サッカー各リーグ"""

    BASE_URL = SOURCES["football_hande"]["base_url"]

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
        tables = soup.find_all("table", class_="index-handi-table")

        for table in tables:
            game = self._parse_game_table(table, d, league_code)
            if game:
                games.append(game)

        logger.info(f"[football_hande/{league_code}] {d}: {len(games)} games")
        return games

    def _parse_game_table(
        self, table: Tag, d: date, league_code: str
    ) -> Optional[GameData]:
        """1試合のtableをパース"""
        rows = table.find_all("tr")
        if len(rows) < 3:
            return None

        # Row 0: 節・時刻・スタジアム
        header = rows[0].find("th")
        game_time = None
        venue = None
        if header:
            header_text = header.get_text(strip=True)
            # "第37節 14:00 味の素スタジアム" のようなフォーマット
            time_match = re.search(r"(\d{1,2}:\d{2})", header_text)
            if time_match:
                game_time = time_match.group(1)
            # 時刻の後ろがスタジアム名
            venue_match = re.search(r"\d{1,2}:\d{2}\s*(.*)", header_text)
            if venue_match:
                venue = venue_match.group(1).strip()

        # Row 1: ハンデ結果
        result_cells = rows[1].find_all("td")
        if len(result_cells) < 3:
            return None

        # ハンデテキストを両側から探す
        # テキストがある側 = ハンデを背負うチーム (有利判定チーム)
        # win-cell = 試合の勝者 (ハンデ勝負とは無関係)
        home_text = result_cells[0].get_text(strip=True)
        away_text = result_cells[2].get_text(strip=True)

        hande_text = ""
        handicap_on_home = False

        # ホーム側にハンデテキストがあるか
        if home_text and parse_hande_result(home_text):
            hande_text = home_text
            handicap_on_home = True
        # アウェイ側にハンデテキストがあるか
        elif away_text and parse_hande_result(away_text):
            hande_text = away_text
            handicap_on_home = False

        parsed = parse_hande_result(hande_text)
        if not parsed:
            return None

        handicap_value, result_code = parsed

        # Row 2: チーム名・スコア
        team_cells = rows[2].find_all("td")
        if len(team_cells) < 3:
            return None

        # ホームチーム
        home_team_div = team_cells[0].find("div", class_="handi-table-team")
        home_score_div = team_cells[0].find("div", class_="handi-table-score")
        # アウェイチーム
        away_team_div = team_cells[2].find("div", class_="handi-table-team")
        away_score_div = team_cells[2].find("div", class_="handi-table-score")

        if not all([home_team_div, away_team_div]):
            return None

        # チーム名抽出 (home_away-txt spanを除く)
        home_team = self._extract_team_name(home_team_div)
        away_team = self._extract_team_name(away_team_div)

        # スコア
        home_score = self._extract_score(home_score_div)
        away_score = self._extract_score(away_score_div)

        # ハンデを背負うチーム = テキストがある側
        if handicap_on_home:
            handicap_team = home_team
        else:
            handicap_team = away_team

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

    def _extract_team_name(self, div: Tag) -> str:
        """チーム名divからhome_away-txtを除いて名前を取得"""
        # spanを一時的に除去してテキスト取得
        for span in div.find_all("span", class_="home_away-txt"):
            span.decompose()
        name = div.get_text(strip=True)
        # "vs" は中央カラムなので除外
        if name == "vs":
            return ""
        return name

    def _extract_score(self, div: Optional[Tag]) -> Optional[int]:
        """スコアdivから数値を取得"""
        if not div:
            return None
        text = div.get_text(strip=True)
        if text == "-" or text == "":
            return None
        try:
            return int(text)
        except ValueError:
            return None

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
