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
import os
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
    handicap_display: str = ""  # 原文表記 (例: "0半5", "0/3")
    home_pitcher: Optional[str] = None  # 野球: 予告先発
    away_pitcher: Optional[str] = None


class HandenomoriScraper(BaseScraper):
    """ハンデの森: NPB/MLB"""

    BASE_URL = SOURCES["handenomori"]["base_url"]

    LOGIN_URL = "https://handenomori.com/membership-login/"

    def _ensure_login(self) -> None:
        """ハンデの森にログイン（環境変数から認証情報取得）"""
        user = os.environ.get("HANDENOMORI_USER")
        pw = os.environ.get("HANDENOMORI_PASS")
        if not user or not pw:
            logger.warning("HANDENOMORI_USER/PASS not set — handicaps will be hidden (member-only)")
            return
        if user and pw:
            self.login(self.LOGIN_URL, {
                "swpm_login_origination_flag": "1",
                "swpm_user_name": user,
                "swpm_password": pw,
                "rememberme": "on",
                "swpm-login": "Log In",
            })

    def _build_url(self, section: str, d: date) -> str:
        return f"{self.BASE_URL}/{section}/{format_date(d)}"

    @staticmethod
    def parse_handicap_value(text: str) -> float:
        """ハンデ値テキストを数値に変換。'半'=.5, '/'=.25/.75 対応"""
        text = text.strip()
        if not text:
            raise ValueError("empty handicap")
        # "1半5" → 1.5, "0半5" → 0.5, "2半2" → 2.5
        if "半" in text:
            parts = text.split("半")
            base = int(parts[0]) if parts[0] else 0
            return base + 0.5
        # "0/3" → 0.75 (Asian quarter line: X/Y = (X+Y)/2)
        if "/" in text:
            parts = text.split("/")
            return (float(parts[0]) + float(parts[1])) / 2
        return float(text)

    def scrape_date(self, league_code: str, d: date) -> list[GameData]:
        """指定日の全試合データを取得"""
        section_map = {"npb": "jpb", "mlb": "mlb", "wbc": "wbc"}
        section = section_map.get(league_code)
        if not section:
            raise ValueError(f"Unknown league: {league_code}")

        self._ensure_login()

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
        handicap_display = ""
        if handi_table:
            tds = handi_table.find_all("td", class_="single-handi-handi")
            if len(tds) >= 2:
                home_handi = tds[0].get_text(strip=True)
                away_handi = tds[1].get_text(strip=True)

                if home_handi:
                    handicap_team = home_team
                    handicap_display = home_handi
                    try:
                        handicap_value = self.parse_handicap_value(home_handi)
                    except ValueError:
                        return None
                elif away_handi:
                    handicap_team = away_team
                    handicap_display = away_handi
                    try:
                        handicap_value = self.parse_handicap_value(away_handi)
                    except ValueError:
                        return None
                else:
                    pass  # ハンデ未発表 — 試合情報のみ保存

        # ハンデ未発表でもhome_teamをデフォルトに設定して試合を保存
        if not handicap_team:
            handicap_team = home_team
            handicap_value = 0.0

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
            handicap_display=handicap_display,
            home_pitcher=home_pitcher,
            away_pitcher=away_pitcher,
        )

    def get_available_dates(self, league_code: str) -> list[date]:
        """過去データのインデックスページから利用可能な日付一覧を取得"""
        section_map = {"npb": "jpb", "mlb": "mlb", "wbc": "wbc"}
        section = section_map.get(league_code)
        url = f"{self.BASE_URL}/{section}/"

        soup = self.fetch_and_parse(url)
        dates = []

        for link in soup.find_all("a"):
            href = link.get("href", "")
            # /jpb/20250801 のようなパターンを探す
            match = re.search(r"/(?:jpb|mlb|wbc)/(\d{8})", href)
            if match:
                date_str = match.group(1)
                try:
                    from scraper.date_utils import parse_date

                    dates.append(parse_date(date_str))
                except ValueError:
                    pass

        return sorted(set(dates))
