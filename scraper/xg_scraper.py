"""サッカーxGデータ取得 — FotMob API"""

import logging
import requests
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# FotMob league IDs
LEAGUE_IDS = {
    "premier": 47,
    "laliga": 87,
    "seriea": 55,
    "bundesliga": 54,
    "ligue1": 53,
    "eredivisie": 57,
}

# FotMob team name → 日本語マッピング（team_name_map.pyと連携）
FOTMOB_TO_JA = {
    "Arsenal": "アーセナル", "Manchester City": "マンチェスター・C", "Chelsea": "チェルシー",
    "Liverpool": "リヴァプール", "Tottenham": "トッテナム", "Manchester United": "マンチェスター・U",
    "Newcastle": "ニューカッスル", "Brighton": "ブライトン", "Aston Villa": "アストン・ヴィラ",
    "West Ham": "ウェストハム", "Brentford": "ブレントフォード", "Crystal Palace": "クリスタル・パレス",
    "Fulham": "フラム", "Wolverhampton": "ウルブズ", "Everton": "エヴァートン",
    "Nottm Forest": "ノッティンガム", "Bournemouth": "ボーンマス",
    "Barcelona": "バルセロナ", "Real Madrid": "レアル・マドリード", "Atletico Madrid": "アトレティコ",
    "Inter": "インテル", "Milan": "ACミラン", "Napoli": "ナポリ", "Juventus": "ユヴェントス",
    "Bayern Munich": "バイエルン", "Dortmund": "ドルトムント", "Leverkusen": "レバークーゼン",
    "PSG": "パリSG", "Marseille": "マルセイユ", "Monaco": "モナコ",
}


def get_xg_table(league_code: str) -> list:
    """リーグのxGテーブルを取得

    Returns:
        [{team_name, xg, xg_conceded, xg_diff, xpoints, xpoints_diff, played, ...}, ...]
    """
    league_id = LEAGUE_IDS.get(league_code)
    if not league_id:
        return []

    try:
        # チームAPIからxGテーブルを取得
        # まず任意のチームIDを取得
        r = requests.get(f"https://www.fotmob.com/api/tltable?leagueId={league_id}", timeout=10)
        if r.status_code != 200:
            return []

        data = r.json()
        if not data or not isinstance(data, list):
            return []

        table = data[0].get("data", {}).get("table", {})
        xg_table = table.get("xg", [])

        if not xg_table:
            # xgテーブルがない場合、通常テーブルから基本データだけ取得
            all_table = table.get("all", [])
            return [{
                "team_name": t.get("name", ""),
                "team_ja": FOTMOB_TO_JA.get(t.get("shortName", ""), t.get("name", "")),
                "played": t.get("played", 0),
                "wins": t.get("wins", 0),
                "draws": t.get("draws", 0),
                "losses": t.get("losses", 0),
                "pts": t.get("pts", 0),
                "goal_diff": t.get("goalConDiff", 0),
            } for t in all_table]

        result = []
        for t in xg_table:
            played = t.get("played", 1) or 1
            result.append({
                "team_name": t.get("teamName", t.get("name", "")),
                "team_ja": FOTMOB_TO_JA.get(t.get("shortName", ""), t.get("teamName", "")),
                "played": played,
                "xg": round(t.get("xg", 0), 2),
                "xg_conceded": round(t.get("xgConceded", 0), 2),
                "xg_per_game": round(t.get("xg", 0) / played, 3),
                "xg_conceded_per_game": round(t.get("xgConceded", 0) / played, 3),
                "xg_diff": round(t.get("xgDiff", 0), 2),
                "xg_conceded_diff": round(t.get("xgConcededDiff", 0), 2),
                "xpoints": round(t.get("xPoints", 0), 1),
                "xpoints_diff": round(t.get("xPointsDiff", 0), 1),
                "xposition": t.get("xPosition", 0),
                "position": t.get("position", 0),
                "pts": t.get("pts", 0),
            })

        return result

    except Exception as e:
        logger.warning(f"xG fetch failed for {league_code}: {e}")
        return []


def get_all_xg() -> Dict[str, list]:
    """全リーグのxGデータを取得"""
    result = {}
    for league in LEAGUE_IDS:
        table = get_xg_table(league)
        if table:
            result[league] = table
            logger.info(f"{league}: {len(table)} teams with xG data")
    return result


def get_team_xg(team_name_ja: str, league_code: str) -> Optional[dict]:
    """特定チームのxGデータを取得"""
    table = get_xg_table(league_code)
    for t in table:
        if t["team_ja"] == team_name_ja or team_name_ja in t["team_name"]:
            return t
    # 部分一致
    for t in table:
        if team_name_ja[:3] in t["team_ja"] or team_name_ja[:3] in t["team_name"]:
            return t
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    all_xg = get_all_xg()
    for league, table in all_xg.items():
        print(f"\n=== {league} xG Top 5 ===")
        for t in sorted(table, key=lambda x: x.get("xg_per_game", 0), reverse=True)[:5]:
            print(f"  {t['team_ja']:20s} xG/試合:{t.get('xg_per_game',0):.2f} xGA/試合:{t.get('xg_conceded_per_game',0):.2f} xGdiff:{t.get('xg_diff',0):+.1f}")
