"""OddsPortal スクレイパー (Playwright) — 複数ブックメーカーのオッズを取得"""

from __future__ import annotations

import logging
import re
import time
import random
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

# OddsPortal のリーグURL
LEAGUE_URLS = {
    # Baseball
    "npb": "/baseball/japan/npb/",
    "mlb": "/baseball/usa/mlb/",
    # Basketball
    "nba": "/basketball/usa/nba/",
    "bleague": "/basketball/japan/b-league/",
    # Soccer
    "jleague": "/football/japan/j1-league/",
    "premier": "/football/england/premier-league/",
    "laliga": "/football/spain/laliga/",
    "seriea": "/football/italy/serie-a/",
    "bundesliga": "/football/germany/bundesliga/",
    "ligue1": "/football/france/ligue-1/",
    "eredivisie": "/football/netherlands/eredivisie/",
    "cl": "/football/europe/champions-league/",
}

# チーム名正規化マップ (OddsPortal英語名 → DB日本語名)
# NPB
ODDSPORTAL_NPB_TEAMS = {
    "Yomiuri Giants": "巨人",
    "Hanshin Tigers": "阪神",
    "Yokohama DeNA BayStars": "DeNA",
    "Hiroshima Toyo Carp": "広島",
    "Tokyo Yakult Swallows": "ヤクルト",
    "Chunichi Dragons": "中日",
    "Orix Buffaloes": "オリックス",
    "SoftBank Hawks": "ソフトバンク",
    "Fukuoka SoftBank Hawks": "ソフトバンク",
    "Saitama Seibu Lions": "西武",
    "Tohoku Rakuten Golden Eagles": "楽天",
    "Rakuten Gold Eagles": "楽天",
    "Chiba Lotte Marines": "ロッテ",
    "Hokkaido Nippon-Ham Fighters": "日本ハム",
    "Nippon-Ham Fighters": "日本ハム",
}

# MLB (主要チーム)
ODDSPORTAL_MLB_TEAMS = {
    "New York Yankees": "ヤンキース",
    "Los Angeles Dodgers": "ドジャース",
    "Houston Astros": "アストロズ",
    "Atlanta Braves": "ブレーブス",
    "Philadelphia Phillies": "フィリーズ",
    "Baltimore Orioles": "オリオールズ",
    "Texas Rangers": "レンジャーズ",
    "Arizona Diamondbacks": "ダイヤモンドバックス",
    "Minnesota Twins": "ツインズ",
    "Tampa Bay Rays": "レイズ",
    "Milwaukee Brewers": "ブリュワーズ",
    "Toronto Blue Jays": "ブルージェイズ",
    "Chicago Cubs": "カブス",
    "Seattle Mariners": "マリナーズ",
    "Miami Marlins": "マーリンズ",
    "San Diego Padres": "パドレス",
    "Boston Red Sox": "レッドソックス",
    "Cleveland Guardians": "ガーディアンズ",
    "Cincinnati Reds": "レッズ",
    "Pittsburgh Pirates": "パイレーツ",
    "Kansas City Royals": "ロイヤルズ",
    "San Francisco Giants": "ジャイアンツ",
    "Detroit Tigers": "タイガース",
    "St. Louis Cardinals": "カーディナルス",
    "New York Mets": "メッツ",
    "Los Angeles Angels": "エンゼルス",
    "Chicago White Sox": "ホワイトソックス",
    "Washington Nationals": "ナショナルズ",
    "Colorado Rockies": "ロッキーズ",
    "Oakland Athletics": "アスレチックス",
}

# 全マッピング統合
ODDSPORTAL_TEAM_MAP = {
    **ODDSPORTAL_NPB_TEAMS,
    **ODDSPORTAL_MLB_TEAMS,
}


@dataclass
class OddsPortalMatch:
    """OddsPortalの試合オッズデータ"""
    home_team: str
    away_team: str
    match_date: date
    match_time: Optional[str] = None
    # h2h odds (best/average)
    home_odds: Optional[float] = None
    away_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    # 複数ブックメーカーのオッズ
    bookmaker_odds: list[dict] = field(default_factory=list)
    # score (finished matches)
    home_score: Optional[int] = None
    away_score: Optional[int] = None


class OddsPortalScraper:
    """Playwright を使ったOddsPortalスクレイパー"""

    BASE_URL = "https://www.oddsportal.com"

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._browser = None
        self._context = None
        self._page = None

    def _ensure_browser(self):
        """Playwright ブラウザの起動（遅延初期化）"""
        if self._browser is not None:
            return

        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless, channel="chrome"
        )
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        self._page = self._context.new_page()
        # Cookie consent (GDPR)
        self._page.goto(self.BASE_URL, wait_until="networkidle", timeout=30000)
        try:
            self._page.click("button#onetrust-accept-btn-handler", timeout=5000)
        except Exception:
            pass  # consent button may not appear
        time.sleep(random.uniform(1, 2))

    def close(self):
        if self._browser:
            self._browser.close()
            self._playwright.stop()
            self._browser = None

    def scrape_league(
        self, league_code: str, target_date: Optional[date] = None
    ) -> list[OddsPortalMatch]:
        """
        リーグの試合オッズを取得。

        Args:
            league_code: npb, mlb, nba, etc.
            target_date: 特定日のみ取得する場合

        Returns:
            list[OddsPortalMatch]
        """
        path = LEAGUE_URLS.get(league_code)
        if not path:
            logger.warning(f"No OddsPortal URL for league: {league_code}")
            return []

        self._ensure_browser()

        # 結果ページ or 今日のページ
        if target_date and target_date < date.today():
            # 過去の結果: /results/ ページ
            url = f"{self.BASE_URL}{path}results/"
        else:
            url = f"{self.BASE_URL}{path}"

        logger.info(f"Scraping OddsPortal: {url}")

        try:
            self._page.goto(url, wait_until="networkidle", timeout=30000)
            time.sleep(random.uniform(2, 4))
        except Exception as e:
            logger.error(f"Failed to load {url}: {e}")
            return []

        matches = self._parse_matches_page(league_code, target_date)
        logger.info(f"Found {len(matches)} matches from OddsPortal for {league_code}")
        return matches

    def scrape_match_detail(self, match_url: str) -> list[dict]:
        """
        個別試合ページから全ブックメーカーのオッズを取得。

        Returns:
            list of {"bookmaker": str, "home_odds": float, "away_odds": float, "draw_odds": float|None}
        """
        self._ensure_browser()

        try:
            self._page.goto(
                f"{self.BASE_URL}{match_url}",
                wait_until="networkidle",
                timeout=30000,
            )
            time.sleep(random.uniform(2, 3))
        except Exception as e:
            logger.error(f"Failed to load match detail {match_url}: {e}")
            return []

        odds_list = []
        try:
            # オッズテーブルの各行を取得
            rows = self._page.query_selector_all("div.border-black-borders")
            if not rows:
                rows = self._page.query_selector_all("tr[data-bid]")

            for row in rows:
                try:
                    # ブックメーカー名
                    bm_el = row.query_selector("a.leading-6, p.height-content, a[href*='bookmaker']")
                    if not bm_el:
                        continue
                    bm_name = bm_el.inner_text().strip()

                    # オッズ値
                    odds_els = row.query_selector_all("p.height-content, td.table-main__detail-odds")
                    odds_values = []
                    for el in odds_els:
                        text = el.inner_text().strip()
                        try:
                            odds_values.append(float(text))
                        except ValueError:
                            continue

                    if len(odds_values) >= 2:
                        entry = {
                            "bookmaker": self._normalize_bookmaker(bm_name),
                            "home_odds": odds_values[0],
                        }
                        if len(odds_values) == 3:
                            entry["draw_odds"] = odds_values[1]
                            entry["away_odds"] = odds_values[2]
                        else:
                            entry["draw_odds"] = None
                            entry["away_odds"] = odds_values[1]

                        odds_list.append(entry)
                except Exception:
                    continue

        except Exception as e:
            logger.error(f"Failed to parse match detail: {e}")

        return odds_list

    def _parse_matches_page(
        self, league_code: str, target_date: Optional[date]
    ) -> list[OddsPortalMatch]:
        """メインの試合一覧ページを解析"""
        # スポーツカテゴリのURLパターン
        sport_path = LEAGUE_URLS[league_code].split("/")[1]  # baseball, basketball, football

        # JavaScript で試合データを抽出
        match_data = self._page.evaluate(f'''() => {{
            const results = [];
            // 試合リンクを全取得
            const links = document.querySelectorAll('a[href*="/{sport_path}/"]');
            const seen = new Set();

            for (const link of links) {{
                const href = link.getAttribute('href') || '';
                // 試合URLのパターン: /sport/country/league/team1-team2-HASH/
                if (href.split('/').length < 6 || href.endsWith('/results/') || href.endsWith('/outrights/'))
                    continue;
                const parts = href.split('/');
                const slug = parts[parts.length - 2] || parts[parts.length - 1];
                if (!slug || slug.length < 10 || !slug.includes('-'))
                    continue;
                if (seen.has(href)) continue;
                seen.add(href);

                // 親のrow要素を探す
                let row = link.closest('div');
                // さらに親を遡って試合行全体を見つける
                for (let i = 0; i < 5 && row; i++) {{
                    const text = row.innerText || '';
                    if (text.match(/\\d\\.\\d{{2}}/)) break;  // オッズ値が見つかったらストップ
                    row = row.parentElement;
                }}

                if (!row) continue;

                const rowText = row.innerText || '';
                const linkText = link.innerText || '';

                // オッズ値を取得 (p.height-content)
                const oddsParagraphs = row.querySelectorAll('p.height-content');
                const oddsValues = [];
                for (const p of oddsParagraphs) {{
                    const v = parseFloat(p.textContent.trim());
                    if (!isNaN(v) && v > 1.0 && v < 100) oddsValues.push(v);
                }}

                results.push({{
                    href: href,
                    linkText: linkText.trim(),
                    rowText: rowText.trim().substring(0, 500),
                    odds: oddsValues,
                }});
            }}
            return results;
        }}''')

        matches = []
        seen_teams = set()

        for data in match_data:
            try:
                match = self._parse_extracted_data(data, league_code, target_date)
                if match:
                    key = f"{match.home_team}|{match.away_team}"
                    if key not in seen_teams:
                        seen_teams.add(key)
                        matches.append(match)
            except Exception as e:
                logger.debug(f"Failed to parse match data: {e}")

        return matches

    def _parse_extracted_data(
        self, data: dict, league_code: str, target_date: Optional[date]
    ) -> Optional[OddsPortalMatch]:
        """JavaScriptで抽出した生データを解析"""
        link_text = data.get("linkText", "")
        row_text = data.get("rowText", "")
        odds = data.get("odds", [])

        # リンクテキストからチーム名を抽出: "Team1 vs Team2 - DD/MM/YYYY"
        # or "HH:MM\nTeam1\n...\nTeam2"
        vs_match = re.match(r'^(.+?)\s+vs\s+(.+?)(?:\s*-\s*(\d{2}/\d{2}/\d{4}))?$', link_text)
        if vs_match:
            home_name = vs_match.group(1).strip()
            away_name = vs_match.group(2).strip()
            date_str = vs_match.group(3)
        else:
            # 改行区切りのテキストから抽出を試行
            lines = [l.strip() for l in link_text.split('\n') if l.strip()]
            # 数字だけの行(時間、スコア)を除去
            team_lines = [l for l in lines if not re.match(r'^[\d:–\-/]+$', l) and len(l) > 2]
            if len(team_lines) >= 2:
                home_name = team_lines[0]
                away_name = team_lines[1]
                date_str = None
            else:
                return None

        if not home_name or not away_name or home_name == away_name:
            return None

        # 日付解析
        match_date = target_date or date.today()
        if not target_date:
            # rowTextから日付を探す
            date_match = re.search(r'(\d{2})/(\d{2})/(\d{4})', link_text + " " + row_text)
            if date_match:
                day, month, year = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
                match_date = date(year, month, day)

        # スコア抽出
        score_match = re.search(r'(\d+)\s*[–\-]\s*(\d+)', row_text)
        home_score = int(score_match.group(1)) if score_match else None
        away_score = int(score_match.group(2)) if score_match else None

        # 時間抽出
        time_match = re.search(r'(\d{2}:\d{2})', row_text)
        match_time = time_match.group(1) if time_match else None

        match = OddsPortalMatch(
            home_team=home_name,
            away_team=away_name,
            match_date=match_date,
            match_time=match_time,
            home_score=home_score,
            away_score=away_score,
        )

        # オッズ割り当て
        if len(odds) >= 2:
            match.home_odds = odds[0]
            if len(odds) == 3:
                # サッカー: home, draw, away
                match.draw_odds = odds[1]
                match.away_odds = odds[2]
            else:
                match.away_odds = odds[1]

        return match

    def _normalize_bookmaker(self, name: str) -> str:
        """ブックメーカー名を正規化"""
        name_lower = name.lower().strip()
        mapping = {
            "pinnacle": "pinnacle",
            "pinnacle sports": "pinnacle",
            "bet365": "bet365",
            "1xbet": "1xbet",
            "william hill": "williamhill",
            "williamhill": "williamhill",
            "unibet": "unibet",
            "betfair": "betfair",
            "betfair exchange": "betfair_ex",
            "marathonbet": "marathonbet",
            "marathon": "marathonbet",
            "bwin": "bwin",
            "betway": "betway",
            "888sport": "888sport",
            "sportingbet": "sportingbet",
        }
        for key, val in mapping.items():
            if key in name_lower:
                return val
        return name_lower.replace(" ", "_")

    def resolve_team_name(self, en_name: str, league_code: str) -> Optional[str]:
        """OddsPortalの英語名 → DB日本語名"""
        # 直接マッチ
        if en_name in ODDSPORTAL_TEAM_MAP:
            return ODDSPORTAL_TEAM_MAP[en_name]

        # team_name_map からも検索
        from scraper.team_name_map import find_ja_name
        ja = find_ja_name(en_name)
        if ja:
            return ja

        # 部分一致
        for en, ja in ODDSPORTAL_TEAM_MAP.items():
            if en_name in en or en in en_name:
                return ja

        return None
