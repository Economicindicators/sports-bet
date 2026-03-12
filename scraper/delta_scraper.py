"""1point02.jp (DELTA Inc) スクレイパー — NPBセイバーメトリクス取得"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import requests
from bs4 import BeautifulSoup

from scraper.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

BASE_URL = "https://1point02.jp/op/gnav"

# チーム略称 → DB名
DELTA_TEAM_MAP = {
    "G": "巨人", "T": "阪神", "DB": "DeNA", "C": "広島",
    "S": "ヤクルト", "D": "中日",
    "Bs": "オリックス", "H": "ソフトバンク", "L": "西武",
    "E": "楽天", "M": "ロッテ", "F": "日本ハム",
}

# チーム日本語名 → DB名 (チーム成績用、正式名称からの変換)
DELTA_TEAM_FULLNAME = {
    "読売": "巨人", "阪神": "阪神", "ＤｅＮＡ": "DeNA", "DeNA": "DeNA",
    "広島": "広島", "広島東洋": "広島", "ヤクルト": "ヤクルト",
    "東京ヤクルト": "ヤクルト", "中日": "中日",
    "オリックス": "オリックス", "ソフトバンク": "ソフトバンク",
    "福岡ソフトバンク": "ソフトバンク",
    "西武": "西武", "埼玉西武": "西武",
    "楽天": "楽天", "東北楽天": "楽天",
    "ロッテ": "ロッテ", "千葉ロッテ": "ロッテ",
    "日本ハム": "日本ハム", "北海道日本ハム": "日本ハム",
    "横浜DeNA": "DeNA", "横浜ＤｅＮＡ": "DeNA",
}


@dataclass
class PlayerBattingStats:
    """打撃成績"""
    name: str
    team_code: str
    team_name: str  # DB名
    games: int = 0
    pa: int = 0          # 打席
    ab: int = 0          # 打数
    hits: int = 0        # 安打
    doubles: int = 0     # 二塁打
    triples: int = 0     # 三塁打
    hr: int = 0          # 本塁打
    runs: int = 0        # 得点
    rbi: int = 0         # 打点
    bb: int = 0          # 四球
    so: int = 0          # 三振
    hbp: int = 0         # 死球
    sb: int = 0          # 盗塁
    cs: int = 0          # 盗塁刺
    avg: float = 0.0     # 打率
    # 計算値
    obp: float = 0.0     # 出塁率
    slg: float = 0.0     # 長打率
    ops: float = 0.0     # OPS
    iso: float = 0.0     # ISO (長打力)
    bb_rate: float = 0.0 # 四球率
    k_rate: float = 0.0  # 三振率


@dataclass
class PlayerPitchingStats:
    """投手成績"""
    name: str
    team_code: str
    team_name: str
    wins: int = 0
    losses: int = 0
    era: float = 0.0
    games: int = 0
    gs: int = 0          # 先発
    cg: int = 0          # 完投
    sho: int = 0         # 完封
    saves: int = 0
    holds: int = 0
    ip: float = 0.0      # 投球回
    hits_allowed: int = 0
    runs_allowed: int = 0
    er: int = 0          # 自責点
    hr_allowed: int = 0
    bb: int = 0
    so: int = 0
    # 計算値
    whip: float = 0.0
    k9: float = 0.0      # 奪三振率
    bb9: float = 0.0     # 与四球率
    hr9: float = 0.0     # 被本塁打率
    k_bb: float = 0.0    # K-BB%
    fip: float = 0.0     # FIP (推定)


@dataclass
class TeamStats:
    """チーム成績"""
    team_name: str       # DB名
    games: int = 0
    pa: int = 0
    avg: float = 0.0
    hr: int = 0
    runs: int = 0
    sb: int = 0
    bb: int = 0
    so: int = 0
    # 計算値
    obp: float = 0.0
    slg: float = 0.0
    ops: float = 0.0
    runs_per_game: float = 0.0


class DeltaScraper(BaseScraper):
    """1point02.jp NPBセイバーメトリクス スクレイパー"""

    def scrape_batting(self, season: int = 2025, league: int = 0) -> list[PlayerBattingStats]:
        """打撃成績を取得 (無料ページ)"""
        url = f"{BASE_URL}/leaders/pl/pbs_standard.aspx?sn={season}&lg={league}&pn=-1"
        html = self.fetch(url)
        soup = self.parse(html)

        table = soup.find("table", id="gvMaster")
        if not table:
            logger.warning("Batting table not found")
            return []

        rows = table.find_all("tr")
        if len(rows) < 2:
            return []

        # ヘッダー解析
        headers = [th.text.strip() for th in rows[0].find_all(["th", "td"])]
        col_idx = {h: i for i, h in enumerate(headers)}

        results = []
        for row in rows[1:]:
            cells = [td.text.strip() for td in row.find_all("td")]
            if len(cells) < 10:
                continue

            try:
                team_code = cells[col_idx.get("球団", 1)]
                team_name = DELTA_TEAM_MAP.get(team_code, team_code)
                name = cells[col_idx.get("選手", 3)]

                # 全角数字対応
                def to_int(s):
                    return int(s.replace(",", "")) if s and s != "-" else 0
                def to_float(s):
                    return float(s) if s and s != "-" else 0.0

                games = to_int(cells[col_idx.get("試合", 4)])
                pa = to_int(cells[col_idx.get("打席", 5)])
                ab = to_int(cells[col_idx.get("打数", 6)])
                hits = to_int(cells[col_idx.get("安打", 7)])
                doubles = to_int(cells[col_idx.get("二塁打", 9)])
                triples = to_int(cells[col_idx.get("三塁打", 10)])
                hr = to_int(cells[col_idx.get("本塁打", 11)])
                runs = to_int(cells[col_idx.get("得点", 12)])
                rbi = to_int(cells[col_idx.get("打点", 13)])
                bb = to_int(cells[col_idx.get("四球", 14)])
                so = to_int(cells[col_idx.get("三振", 16)])
                hbp = to_int(cells[col_idx.get("死球", 17)])
                sb = to_int(cells[col_idx.get("盗塁", 21)])
                cs = to_int(cells[col_idx.get("盗塁刺", 22)])
                avg = to_float(cells[col_idx.get("打率", 23)])

                # 計算値
                obp = (hits + bb + hbp) / pa if pa > 0 else 0
                tb = hits + doubles + 2 * triples + 3 * hr  # singles already in hits
                slg = tb / ab if ab > 0 else 0
                ops = obp + slg
                iso = slg - avg
                bb_rate = bb / pa if pa > 0 else 0
                k_rate = so / pa if pa > 0 else 0

                stats = PlayerBattingStats(
                    name=name, team_code=team_code, team_name=team_name,
                    games=games, pa=pa, ab=ab, hits=hits,
                    doubles=doubles, triples=triples, hr=hr,
                    runs=runs, rbi=rbi, bb=bb, so=so, hbp=hbp,
                    sb=sb, cs=cs, avg=avg,
                    obp=round(obp, 3), slg=round(slg, 3),
                    ops=round(ops, 3), iso=round(iso, 3),
                    bb_rate=round(bb_rate, 3), k_rate=round(k_rate, 3),
                )
                results.append(stats)
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse batting row: {e}")
                continue

        logger.info(f"Scraped {len(results)} batting records for {season}")
        return results

    def scrape_pitching(self, season: int = 2025, league: int = 0) -> list[PlayerPitchingStats]:
        """投手成績を取得 (無料ページ)"""
        url = f"{BASE_URL}/leaders/pl/pps_standard.aspx?sn={season}&lg={league}&pn=-1"
        html = self.fetch(url)
        soup = self.parse(html)

        table = soup.find("table", id="gvMaster")
        if not table:
            logger.warning("Pitching table not found")
            return []

        rows = table.find_all("tr")
        if len(rows) < 2:
            return []

        headers = [th.text.strip() for th in rows[0].find_all(["th", "td"])]
        col_idx = {h: i for i, h in enumerate(headers)}

        results = []
        for row in rows[1:]:
            cells = [td.text.strip() for td in row.find_all("td")]
            if len(cells) < 10:
                continue

            try:
                team_code = cells[col_idx.get("球団", 1)]
                team_name = DELTA_TEAM_MAP.get(team_code, team_code)
                name = cells[col_idx.get("選手", 3)]

                def to_int(s):
                    return int(s.replace(",", "")) if s and s != "-" else 0
                def to_float(s):
                    return float(s) if s and s != "-" else 0.0

                wins = to_int(cells[col_idx.get("勝", 4)])
                losses = to_int(cells[col_idx.get("敗", 5)])
                era = to_float(cells[col_idx.get("防御率", 6)])
                games = to_int(cells[col_idx.get("登板", 7)])
                gs = to_int(cells[col_idx.get("先発", 8)])
                cg = to_int(cells[col_idx.get("完投", 9)])
                sho = to_int(cells[col_idx.get("完封", 10)])
                saves = to_int(cells[col_idx.get("セーブ", 11)])
                holds = to_int(cells[col_idx.get("ホールド", 12)])

                # 投球回 (例: "150.1" → 150.333)
                ip_raw = cells[col_idx.get("投球回", 13)]
                if "." in ip_raw:
                    ip_parts = ip_raw.split(".")
                    ip = int(ip_parts[0]) + int(ip_parts[1]) / 3
                else:
                    ip = to_float(ip_raw)

                hits_allowed = to_int(cells[col_idx.get("被安打", 15)])
                runs_allowed = to_int(cells[col_idx.get("失点", 16)])
                er = to_int(cells[col_idx.get("自責点", 17)])
                hr_allowed = to_int(cells[col_idx.get("本塁打", 18)])
                bb = to_int(cells[col_idx.get("四球", 19)])
                so = to_int(cells[col_idx.get("三振", 24)])

                # 計算値
                whip = (hits_allowed + bb) / ip if ip > 0 else 0
                k9 = (so * 9) / ip if ip > 0 else 0
                bb9 = (bb * 9) / ip if ip > 0 else 0
                hr9 = (hr_allowed * 9) / ip if ip > 0 else 0
                # FIP = ((13*HR + 3*BB - 2*K) / IP) + constant (~3.10 for NPB)
                fip_constant = 3.10
                fip = ((13 * hr_allowed + 3 * bb - 2 * so) / ip + fip_constant) if ip > 0 else 0

                batters_faced = to_int(cells[col_idx.get("打者", 14)])
                k_bb = (so - bb) / batters_faced if batters_faced > 0 else 0

                stats = PlayerPitchingStats(
                    name=name, team_code=team_code, team_name=team_name,
                    wins=wins, losses=losses, era=era,
                    games=games, gs=gs, cg=cg, sho=sho,
                    saves=saves, holds=holds, ip=round(ip, 1),
                    hits_allowed=hits_allowed, runs_allowed=runs_allowed,
                    er=er, hr_allowed=hr_allowed, bb=bb, so=so,
                    whip=round(whip, 3), k9=round(k9, 2),
                    bb9=round(bb9, 2), hr9=round(hr9, 2),
                    k_bb=round(k_bb, 3), fip=round(fip, 2),
                )
                results.append(stats)
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse pitching row: {e}")
                continue

        logger.info(f"Scraped {len(results)} pitching records for {season}")
        return results

    def scrape_team_batting(self, season: int = 2025, league: int = 0) -> list[TeamStats]:
        """チーム打撃成績を取得"""
        url = f"{BASE_URL}/leaders/tm/tbs_standard.aspx?sn={season}&lg={league}"
        html = self.fetch(url)
        soup = self.parse(html)

        table = soup.find("table", id="gvMaster")
        if not table:
            logger.warning("Team batting table not found")
            return []

        rows = table.find_all("tr")
        if len(rows) < 2:
            return []

        headers = [th.text.strip() for th in rows[0].find_all(["th", "td"])]
        col_idx = {h: i for i, h in enumerate(headers)}

        results = []
        for row in rows[1:]:
            cells = [td.text.strip() for td in row.find_all("td")]
            if len(cells) < 10:
                continue

            try:
                raw_name = cells[col_idx.get("チーム", 1)]
                team_name = DELTA_TEAM_FULLNAME.get(raw_name, raw_name)

                def to_int(s):
                    return int(s.replace(",", "")) if s and s != "-" else 0
                def to_float(s):
                    return float(s) if s and s != "-" else 0.0

                games = to_int(cells[col_idx.get("試合", 2)])
                pa = to_int(cells[col_idx.get("打席", 3)])
                ab = to_int(cells[col_idx.get("打数", 4)])
                hits = to_int(cells[col_idx.get("安打", 5)])
                doubles = to_int(cells[col_idx.get("二塁打", 7)])
                triples = to_int(cells[col_idx.get("三塁打", 8)])
                hr = to_int(cells[col_idx.get("本塁打", 9)])
                runs = to_int(cells[col_idx.get("得点", 10)])
                bb = to_int(cells[col_idx.get("四球", 12)])
                so = to_int(cells[col_idx.get("三振", 14)])
                hbp = to_int(cells[col_idx.get("死球", 15)])
                sb = to_int(cells[col_idx.get("盗塁", 19)])
                avg = to_float(cells[col_idx.get("打率", 21)])

                obp = (hits + bb + hbp) / pa if pa > 0 else 0
                tb = hits + doubles + 2 * triples + 3 * hr
                slg = tb / ab if ab > 0 else 0
                ops = obp + slg
                rpg = runs / games if games > 0 else 0

                stats = TeamStats(
                    team_name=team_name,
                    games=games, pa=pa, avg=avg, hr=hr,
                    runs=runs, sb=sb, bb=bb, so=so,
                    obp=round(obp, 3), slg=round(slg, 3),
                    ops=round(ops, 3), runs_per_game=round(rpg, 2),
                )
                results.append(stats)
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse team batting row: {e}")
                continue

        logger.info(f"Scraped {len(results)} team batting records for {season}")
        return results

    def get_team_aggregate(self, season: int = 2025) -> dict[str, dict]:
        """
        チーム別の打撃・投手の集約データを返す。

        Returns:
            {team_name: {"batting": {...}, "pitching": {...}}}
        """
        batting = self.scrape_batting(season)
        pitching = self.scrape_pitching(season)
        team_batting = self.scrape_team_batting(season)

        # チーム別に集約
        result = {}

        for ts in team_batting:
            result[ts.team_name] = {
                "team_ops": ts.ops,
                "team_avg": ts.avg,
                "team_runs_per_game": ts.runs_per_game,
                "team_hr": ts.hr,
                "team_sb": ts.sb,
                "team_bb": ts.bb,
                "team_so": ts.so,
                "team_obp": ts.obp,
                "team_slg": ts.slg,
            }

        # 投手集約: チーム別の先発投手平均ERA, WHIP, K9
        from collections import defaultdict
        team_pitchers = defaultdict(list)
        for p in pitching:
            if p.gs >= 5:  # 先発5試合以上
                team_pitchers[p.team_name].append(p)

        for team, pitchers in team_pitchers.items():
            if team not in result:
                result[team] = {}
            if pitchers:
                avg_era = sum(p.era for p in pitchers) / len(pitchers)
                avg_whip = sum(p.whip for p in pitchers) / len(pitchers)
                avg_k9 = sum(p.k9 for p in pitchers) / len(pitchers)
                avg_fip = sum(p.fip for p in pitchers) / len(pitchers)
                result[team]["rotation_era"] = round(avg_era, 2)
                result[team]["rotation_whip"] = round(avg_whip, 3)
                result[team]["rotation_k9"] = round(avg_k9, 2)
                result[team]["rotation_fip"] = round(avg_fip, 2)
                result[team]["rotation_depth"] = len(pitchers)

        return result
