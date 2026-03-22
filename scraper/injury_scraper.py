"""NBA欠場情報スクレイパー — Rotowire Lineups"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from scraper.team_name_map import find_ja_name

logger = logging.getLogger(__name__)

ROTOWIRE_URL = "https://www.rotowire.com/basketball/nba-lineups.php"

# 欠場ステータスの重要度
STATUS_IMPACT = {
    "OUT": 1.0,      # 確定欠場
    "OFS": 1.0,      # Out For Season
    "Doub": 0.8,     # Doubtful（80%欠場）
    "Ques": 0.5,     # Questionable（50%欠場）
    "Prob": 0.2,     # Probable（20%欠場）
    "GTD": 0.5,      # Game-Time Decision
}


@dataclass
class InjuredPlayer:
    name: str
    team: str          # 英語チーム名
    team_ja: str       # 日本語チーム名
    status: str        # OUT, Ques, Prob, GTD, OFS, Doub
    impact: float      # 0-1 (欠場確率)
    is_star: bool      # スター選手か


@dataclass
class GameInjuries:
    home_team: str
    away_team: str
    home_team_ja: str
    away_team_ja: str
    home_injuries: List[InjuredPlayer] = field(default_factory=list)
    away_injuries: List[InjuredPlayer] = field(default_factory=list)
    home_impact_score: float = 0.0  # チーム全体の欠場影響度
    away_impact_score: float = 0.0


def scrape_nba_injuries() -> List[GameInjuries]:
    """Rotowireから今日のNBA欠場情報を取得"""
    r = requests.get(ROTOWIRE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    games = []
    game_elements = soup.select(".lineup__matchup")

    for game_el in game_elements:
        # ゲーム全体のコンテナを取得
        game_container = game_el.find_parent(class_="lineup")
        if not game_container:
            continue

        # チーム名取得
        teams = game_container.select(".lineup__abbr")
        if len(teams) < 2:
            continue

        away_abbr = teams[0].text.strip()
        home_abbr = teams[1].text.strip()

        # フルネーム取得（レコード等のゴミを除去）
        team_names = game_container.select(".lineup__team-name, .lineup__mteam")
        away_name = re.sub(r'\s*\([\d\-]+\)\s*$', '', team_names[0].text.strip()) if len(team_names) >= 2 else away_abbr
        home_name = re.sub(r'\s*\([\d\-]+\)\s*$', '', team_names[1].text.strip()) if len(team_names) >= 2 else home_abbr
        away_name = re.sub(r'\s+', ' ', away_name).strip()
        home_name = re.sub(r'\s+', ' ', home_name).strip()

        away_ja = find_ja_name(away_name) or away_name
        home_ja = find_ja_name(home_name) or home_name

        game = GameInjuries(
            home_team=home_name, away_team=away_name,
            home_team_ja=home_ja, away_team_ja=away_ja,
        )

        # 欠場選手を取得（MAY NOT PLAY セクション）
        injury_sections = game_container.select(".lineup__player-highlight, .lineup__status")

        # 全テキストから欠場情報をパース
        game_text = game_container.get_text()
        _parse_injuries_from_text(game_text, game, away_name, home_name)

        games.append(game)

    return games


def _parse_injuries_from_text(text: str, game: GameInjuries, away: str, home: str):
    """テキストから欠場情報をパース"""
    # "MAY NOT PLAY" セクションを探す
    sections = text.split("MAY NOT PLAY")

    if len(sections) < 2:
        return

    # 各チームの欠場セクション
    for i, section in enumerate(sections[1:], 1):
        # ステータス付きのプレイヤーを抽出
        # パターン: "F\nJ. Howard\nQues" or "G\nA. Black\nOUT"
        matches = re.findall(
            r'([FGCS])\s*\n\s*([A-Z][\w.\'\- ]+?)\s*\n?\s*(OUT|OFS|Doub|Ques|Prob|GTD)',
            section[:500],  # セクションの先頭500文字だけ
        )

        for pos, name, status in matches:
            impact = STATUS_IMPACT.get(status, 0.5)
            player = InjuredPlayer(
                name=name.strip(),
                team=away if i % 2 == 1 else home,  # 奇数=アウェイ、偶数=ホーム (推定)
                team_ja=game.away_team_ja if i % 2 == 1 else game.home_team_ja,
                status=status,
                impact=impact,
                is_star=False,  # 後で判定
            )

            if i % 2 == 1:
                game.away_injuries.append(player)
                game.away_impact_score += impact
            else:
                game.home_injuries.append(player)
                game.home_impact_score += impact


def get_injury_alerts(min_impact: float = 0.5) -> List[dict]:
    """欠場アラートを生成。片方のチームだけ主力欠場 → ハンデ乖離チャンス"""
    games = scrape_nba_injuries()
    alerts = []

    for game in games:
        # 影響度の差
        impact_diff = game.home_impact_score - game.away_impact_score

        if abs(impact_diff) < min_impact:
            continue

        if impact_diff > 0:
            weakened = "home"
            weakened_team = game.home_team_ja
            advantage_team = game.away_team_ja
        else:
            weakened = "away"
            weakened_team = game.away_team_ja
            advantage_team = game.home_team_ja

        injured_players = (
            game.home_injuries if weakened == "home" else game.away_injuries
        )
        out_names = [
            f"{p.name}({p.status})" for p in injured_players if p.impact >= 0.5
        ]

        alerts.append({
            "home_team": game.home_team_ja,
            "away_team": game.away_team_ja,
            "weakened_team": weakened_team,
            "advantage_team": advantage_team,
            "impact_diff": round(abs(impact_diff), 1),
            "injured_players": out_names,
            "home_impact": round(game.home_impact_score, 1),
            "away_impact": round(game.away_impact_score, 1),
        })

    alerts.sort(key=lambda a: a["impact_diff"], reverse=True)
    return alerts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    print("=== NBA 欠場情報 ===\n")
    games = scrape_nba_injuries()
    for g in games:
        print(f"{g.away_team_ja} @ {g.home_team_ja}")
        if g.away_injuries:
            print(f"  {g.away_team_ja} 欠場: {', '.join(f'{p.name}({p.status})' for p in g.away_injuries)}")
        if g.home_injuries:
            print(f"  {g.home_team_ja} 欠場: {', '.join(f'{p.name}({p.status})' for p in g.home_injuries)}")
        if not g.away_injuries and not g.home_injuries:
            print("  欠場なし")
        print()

    print("\n=== 欠場アラート ===\n")
    alerts = get_injury_alerts(min_impact=0.5)
    for a in alerts:
        print(f"⚠️ {a['home_team']} vs {a['away_team']}")
        print(f"   {a['weakened_team']} 弱体化 (影響度: {a['impact_diff']})")
        print(f"   欠場: {', '.join(a['injured_players'])}")
        print(f"   → {a['advantage_team']} 有利")
        print()
