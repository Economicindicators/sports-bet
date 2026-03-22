"""フィードバック分析 — 過去の結果からどこにエッジがあるか特定"""

import json
import logging
import os
from collections import defaultdict
from database.models import get_session, Match, HandicapData, Team

logger = logging.getLogger(__name__)


def analyze_feedback() -> dict:
    """全結果データを分析してエッジのある条件を特定"""
    session = get_session()

    hds = session.query(HandicapData).filter(
        HandicapData.result_type != None,
        HandicapData.result_type != '',
        HandicapData.payout_rate != None,
    ).all()

    # リーグ別
    by_league = defaultdict(lambda: {"total": 0, "win": 0, "payout_sum": 0})
    # ハンデ値帯別
    by_handi = defaultdict(lambda: {"total": 0, "win": 0, "payout_sum": 0})
    # リーグ×ハンデ帯
    by_league_handi = defaultdict(lambda: {"total": 0, "win": 0, "payout_sum": 0})

    for hd in hds:
        match = session.get(Match, hd.match_id)
        if not match:
            continue

        league = match.league_code
        payout = hd.payout_rate or 0
        handi_val = abs(hd.handicap_value or 0)
        is_win = payout >= 1.5

        # ハンデ帯
        if handi_val == 0:
            bucket = "0"
        elif handi_val <= 1:
            bucket = "0.3-1"
        elif handi_val <= 2:
            bucket = "1-2"
        else:
            bucket = "2+"

        for d, key in [
            (by_league, league),
            (by_handi, bucket),
            (by_league_handi, f"{league}_{bucket}"),
        ]:
            d[key]["total"] += 1
            d[key]["payout_sum"] += payout
            if is_win:
                d[key]["win"] += 1

    def _stats(d):
        result = {}
        for key, s in d.items():
            t = s["total"]
            if t < 20:
                continue
            result[key] = {
                "total": t,
                "win_rate": round(s["win"] / t, 3),
                "avg_payout": round(s["payout_sum"] / t, 3),
                "roi": round((s["payout_sum"] / t - 1) * 100, 1),
            }
        return result

    league_stats = _stats(by_league)
    handi_stats = _stats(by_handi)
    combo_stats = _stats(by_league_handi)

    # エッジのある条件を特定（ROI > 5%）
    edges = []
    for key, s in sorted(combo_stats.items(), key=lambda x: -x[1]["roi"]):
        if s["roi"] > 5 and s["total"] >= 50:
            league, handi = key.rsplit("_", 1)
            edges.append({
                "league": league,
                "handicap_range": handi,
                "total": s["total"],
                "win_rate": s["win_rate"],
                "roi": s["roi"],
            })

    result = {
        "total_matches": len(hds),
        "by_league": league_stats,
        "by_handicap": handi_stats,
        "profitable_conditions": edges,
    }

    return result


def save_feedback_to_turso(analysis: dict) -> int:
    """分析結果をTursoに保存"""
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

    # configテーブルに保存
    client.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES ('feedback_analysis', ?)",
        [json.dumps(analysis, ensure_ascii=False)],
    )
    client.close()
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    analysis = analyze_feedback()

    print(f"分析対象: {analysis['total_matches']}試合")
    print(f"\n=== リーグ別ROI ===")
    for league, s in sorted(analysis["by_league"].items(), key=lambda x: -x[1]["roi"]):
        print(f"  {league:12s} {s['total']:5d}試合 勝率:{s['win_rate']:.1%} ROI:{s['roi']:+.1f}%")

    print(f"\n=== 利益が出る条件 (ROI>5%, 50試合以上) ===")
    for e in analysis["profitable_conditions"]:
        print(f"  {e['league']} ハンデ{e['handicap_range']}: {e['total']}試合 勝率:{e['win_rate']:.1%} ROI:{e['roi']:+.1f}%")

    saved = save_feedback_to_turso(analysis)
    print(f"\nTurso保存: {saved}")
