"""ライン乖離検出 — ハンデサイト vs ブックメーカーの差を検出してアラート

ハンデの森/嵐/マンが発表したハンデ値と、BMが提示するスプレッドを比較。
ハンデサイトの方が「甘い」（有利な条件）試合を検出する。
"""

import json
import logging
import os
from datetime import date, timedelta
from dataclasses import dataclass
from typing import List, Optional

from database.models import get_session, Match, Team, HandicapData, BookmakerOdds
from scraper.odds_api import get_best_odds
from scraper.team_name_map import find_ja_name

logger = logging.getLogger(__name__)


@dataclass
class LineEdge:
    """ライン乖離アラート"""
    match_id: int
    date: str
    home_team: str
    away_team: str
    league: str
    # ハンデサイト
    handi_team: str          # ハンデ有利チーム
    handi_value: float       # ハンデサイトのハンデ値
    handi_display: str       # 表示文字列
    # ブックメーカー
    bm_spread: float         # BMのスプレッド値
    bm_bookmaker: str        # ブックメーカー名
    bm_spread_odds: float    # BMのスプレッドオッズ
    bm_h2h_odds: float       # BMのh2hオッズ
    # 乖離
    line_diff: float         # ハンデ差（正=ハンデサイトが甘い）
    edge_type: str           # "ハンデ甘い" / "ハンデ厳しい" / "中立"
    edge_level: str          # "大" / "中" / "小"


def detect_line_edges(
    leagues: List[str] = None,
    min_diff: float = 0.5,
) -> List[LineEdge]:
    """ハンデサイト vs BMのライン乖離を検出。

    Args:
        leagues: 対象リーグ（Noneなら全リーグ）
        min_diff: 検出する最小乖離幅

    Returns:
        乖離が検出された試合のリスト
    """
    from config.constants import ALL_LEAGUES, LEAGUE_TO_SPORT
    if leagues is None:
        leagues = ALL_LEAGUES

    session = get_session()
    edges = []

    # 今日と明日の試合を対象
    today = date.today()
    tomorrow = today + timedelta(days=1)

    for league in leagues:
        # BMオッズを取得
        try:
            best_odds = get_best_odds(league)
        except Exception as e:
            logger.warning(f"Odds fetch failed for {league}: {e}")
            continue

        if not best_odds:
            continue

        for key, odds in best_odds.items():
            home_ja = find_ja_name(odds.home_team) or odds.home_team
            away_ja = find_ja_name(odds.away_team) or odds.away_team

            # BMのスプレッド（ハンデ値）
            bm_spread = odds.spread_home
            if bm_spread is None:
                continue

            # DBから対応する試合のハンデデータを探す
            matches = session.query(Match).filter(
                Match.league_code == league,
                Match.date >= today,
                Match.date <= tomorrow + timedelta(days=1),
            ).all()

            for match in matches:
                home = session.get(Team, match.home_team_id)
                away = session.get(Team, match.away_team_id)
                if not home or not away:
                    continue

                # チーム名マッチング（部分一致）
                h_name = home.name
                a_name = away.name
                if not (_fuzzy_match(h_name, home_ja) and _fuzzy_match(a_name, away_ja)):
                    if not (_fuzzy_match(h_name, away_ja) and _fuzzy_match(a_name, home_ja)):
                        continue

                # ハンデデータ取得
                hd = session.query(HandicapData).filter_by(match_id=match.match_id).first()
                if not hd or hd.handicap_value is None:
                    continue

                handi_team = session.get(Team, hd.handicap_team_id)
                if not handi_team:
                    continue

                handi_value = abs(hd.handicap_value or 0)

                # ハンデ有利チームがホームかアウェイか判定
                handi_is_home = hd.handicap_team_id == match.home_team_id

                # BMのspread_homeの意味:
                #   spread_home < 0: ホームが有利（ハンデを背負う側）
                #   spread_home > 0: ホームが不利（ハンデをもらう側）
                # → ハンデ有利チーム基準のBMスプレッドに変換
                if handi_is_home:
                    # ハンデ有利=ホーム → BM spread_home < 0 なら同じ方向
                    bm_handi_value = abs(bm_spread) if bm_spread < 0 else 0
                else:
                    # ハンデ有利=アウェイ → BM spread_home > 0 なら同じ方向
                    bm_handi_value = abs(bm_spread) if bm_spread > 0 else 0

                if bm_handi_value == 0:
                    continue  # BMとハンデサイトで有利チームが違う → 比較不能

                # 乖離計算
                # BMの方がハンデ値が大きい = ハンデサイトが甘い = 有利
                line_diff = bm_handi_value - handi_value

                if abs(line_diff) < min_diff:
                    continue

                if line_diff > 0:
                    edge_type = "ハンデ甘い"  # ハンデサイトの方が有利な条件
                else:
                    edge_type = "ハンデ厳しい"

                if abs(line_diff) >= 2.0:
                    edge_level = "大"
                elif abs(line_diff) >= 1.0:
                    edge_level = "中"
                else:
                    edge_level = "小"

                edges.append(LineEdge(
                    match_id=match.match_id,
                    date=str(match.date),
                    home_team=h_name,
                    away_team=a_name,
                    league=league,
                    handi_team=handi_team.name,
                    handi_value=handi_value,
                    handi_display=hd.handicap_display or "",
                    bm_spread=bm_handi_value,  # 有利チーム基準に統一
                    bm_bookmaker=odds.bookmaker,
                    bm_spread_odds=odds.spread_home_odds or 0,
                    bm_h2h_odds=odds.home_odds or 0,
                    line_diff=round(line_diff, 2),
                    edge_type=edge_type,
                    edge_level=edge_level,
                ))

    # 乖離が大きい順にソート
    edges.sort(key=lambda e: abs(e.line_diff), reverse=True)
    return edges


def _fuzzy_match(name1: str, name2: str) -> bool:
    """チーム名の部分一致"""
    n1 = name1.replace("・", "").replace(" ", "").replace("-", "").lower()[:4]
    n2 = name2.replace("・", "").replace(" ", "").replace("-", "").lower()[:4]
    return n1 == n2 or n1 in name2.lower() or n2 in name1.lower()


def save_edges_to_turso(edges: List[LineEdge]) -> int:
    """乖離アラートをTursoに保存"""
    if not edges:
        return 0

    import libsql_client

    turso_url = os.environ.get("TURSO_DATABASE_URL", "")
    turso_token = os.environ.get("TURSO_AUTH_TOKEN", "")
    if not turso_url or not turso_token:
        for env_path in [
            os.path.expanduser("~/sports-bet/.env"),
            os.path.expanduser("~/sports-bet-web/.env"),
        ]:
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("TURSO_DATABASE_URL"):
                            turso_url = turso_url or line.split("=", 1)[1].strip().strip('"')
                        elif line.startswith("TURSO_AUTH_TOKEN"):
                            turso_token = turso_token or line.split("=", 1)[1].strip().strip('"')

    http_url = turso_url.replace("libsql://", "https://")
    client = libsql_client.create_client_sync(url=http_url, auth_token=turso_token)

    # テーブル作成
    client.execute(
        "CREATE TABLE IF NOT EXISTS line_edges ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  match_id INTEGER,"
        "  date TEXT,"
        "  home_team TEXT,"
        "  away_team TEXT,"
        "  league TEXT,"
        "  handi_team TEXT,"
        "  handi_value REAL,"
        "  handi_display TEXT,"
        "  bm_spread REAL,"
        "  bm_bookmaker TEXT,"
        "  bm_spread_odds REAL,"
        "  bm_h2h_odds REAL,"
        "  line_diff REAL,"
        "  edge_type TEXT,"
        "  edge_level TEXT,"
        "  created_at TEXT DEFAULT (datetime('now')),"
        "  UNIQUE(match_id, bm_bookmaker)"
        ")"
    )

    stmts = [libsql_client.Statement(
        "DELETE FROM line_edges WHERE date >= ?", [str(date.today())]
    )]

    for e in edges:
        stmts.append(libsql_client.Statement(
            "INSERT OR REPLACE INTO line_edges "
            "(match_id, date, home_team, away_team, league, "
            "handi_team, handi_value, handi_display, "
            "bm_spread, bm_bookmaker, bm_spread_odds, bm_h2h_odds, "
            "line_diff, edge_type, edge_level) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                e.match_id, e.date, e.home_team, e.away_team, e.league,
                e.handi_team, e.handi_value, e.handi_display,
                e.bm_spread, e.bm_bookmaker, e.bm_spread_odds, e.bm_h2h_odds,
                e.line_diff, e.edge_type, e.edge_level,
            ],
        ))

    client.batch(stmts)
    client.close()
    return len(edges)


def run_detection(min_diff: float = 0.5) -> dict:
    """乖離検出を実行してTursoに保存"""
    logger.info("=== Line Edge Detection ===")
    edges = detect_line_edges(min_diff=min_diff)
    logger.info(f"Detected {len(edges)} line edges (min_diff={min_diff})")

    for e in edges[:10]:
        icon = "🔥" if e.edge_type == "ハンデ甘い" else "⚠️"
        logger.info(
            f"  {icon} [{e.edge_level}] {e.home_team} vs {e.away_team} "
            f"ハンデ:{e.handi_display or e.handi_value} vs BM:{e.bm_spread} "
            f"差:{e.line_diff:+.1f} ({e.edge_type})"
        )

    saved = save_edges_to_turso(edges)
    logger.info(f"Saved {saved} edges to Turso")
    return {"detected": len(edges), "saved": saved, "edges": [vars(e) for e in edges]}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = run_detection(min_diff=0.5)
    print(json.dumps(result, ensure_ascii=False, indent=2))
