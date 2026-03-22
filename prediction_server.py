"""予測APIサーバー - ダッシュボードからワンクリックで予測を実行"""

import json
import logging
import os
import subprocess

from dotenv import load_dotenv
load_dotenv()
import tempfile
import pandas as pd
from datetime import date, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from features.feature_pipeline import load_matches_df, build_features
from models.training import load_model
from betting.handicap_ev import calculate_handicap_ev, calculate_contrarian_ev, home_prob_to_handicap_prob
from betting.kelly import kelly_fraction
from config.constants import (
    CONTRARIAN_MIN_EV_THRESHOLD,
    LEAGUE_EV_THRESHOLD,
    LEAGUE_TO_SPORT,
    MIN_EV_THRESHOLD,
    PROB_PICK_THRESHOLD,
    SPORT_TO_LEAGUES,
    SPORT_MODEL_VERSION,
)
from database.models import get_session, Match, Team, HandicapData
from scraper.manager import ScrapeManager
from scraper.date_utils import format_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TURSO_DB = "sports-bet"

# ── EV 3段階判定 ──
# スポーツ別の閾値倍率でhigh/mid/low/passを判定
def _ev_tier(ev: float, sport: str) -> str:
    """EV値からtierを返す: high, mid, low, pass"""
    # スポーツに属するリーグのEV閾値を取得
    leagues = SPORT_TO_LEAGUES.get(sport, [])
    base = MIN_EV_THRESHOLD
    for lg in leagues:
        if lg in LEAGUE_EV_THRESHOLD:
            base = LEAGUE_EV_THRESHOLD[lg]
            break
    # high: base*1.5以上, mid: base以上, low: base*0.5以上, pass: それ未満
    if ev >= base * 1.5:
        return "high"
    elif ev >= base:
        return "mid"
    elif ev >= base * 0.5:
        return "low"
    return "pass"


def _ev_tier_label(ev: float, sport: str) -> str:
    """EV値から表示ラベルを返す"""
    tier = _ev_tier(ev, sport)
    if tier == "high":
        return "🔥 自信あり（期待値高）"
    elif tier == "mid":
        return "⚡ BET推奨（期待値あり）"
    elif tier == "low":
        return "🔍 様子見（期待値小）"
    return "❌ PASS（見送り推奨）"
TURSO_BIN = os.environ.get("TURSO_BIN", os.path.expanduser("~/.turso/turso"))


def scrape_today(leagues: list[str], target_date: date = None) -> dict:
    """指定リーグの今日のデータをスクレイプ"""
    td = target_date or date.today()
    manager = ScrapeManager(use_cache=False)
    results = {}
    for league in leagues:
        try:
            count = manager.scrape_and_save(league, td)
            results[league] = {"status": "ok", "matches": count}
            logger.info(f"Scraped {league}/{format_date(td)}: {count} matches")
        except Exception as e:
            results[league] = {"status": "error", "error": str(e)}
            logger.error(f"Scrape error {league}: {e}")
    return results


def extract_insights(row, sport: str, home_name: str, away_name: str, h_team: str) -> dict:
    """予測の根拠となるフィーチャー値を人間可読な形式で抽出"""
    def g(key, default=0):
        v = row.get(key, default)
        try:
            return round(float(v), 3) if v is not None else default
        except (ValueError, TypeError):
            return default

    insights = {
        "elo": {
            "home": g("home_elo"),
            "away": g("away_elo"),
            "diff": g("elo_diff"),
            "expected": g("elo_expected"),
        },
        "recent_form": {
            "home_wr5": g("home_wr_5"),
            "away_wr5": g("away_wr_5"),
            "home_wr10": g("home_wr_10"),
            "away_wr10": g("away_wr_10"),
            "home_streak": g("home_streak"),
            "away_streak": g("away_streak"),
        },
        "h2h": {
            "home_wr": g("h2h_home_wr"),
            "away_wr": g("h2h_away_wr"),
            "score_diff": g("h2h_score_diff"),
            "count": g("h2h_count"),
        },
        "schedule": {
            "home_rest": g("home_rest_days"),
            "away_rest": g("away_rest_days"),
            "home_b2b": g("home_back_to_back"),
            "away_b2b": g("away_back_to_back"),
        },
        "form_trend": {
            "home_recent": g("home_recent_form"),
            "away_recent": g("away_recent_form"),
            "home_scored": g("home_scored_trend"),
            "away_scored": g("away_scored_trend"),
            "home_conceded": g("home_conceded_trend"),
            "away_conceded": g("away_conceded_trend"),
        },
        "home_name": home_name,
        "away_name": away_name,
        "handicap_team": h_team,
    }

    if sport == "baseball":
        insights["pitcher"] = {
            "home_wr": g("home_pitcher_wr"),
            "away_wr": g("away_pitcher_wr"),
            "home_era": g("home_pitcher_era"),
            "away_era": g("away_pitcher_era"),
            "era_diff": g("pitcher_era_diff"),
        }
    elif sport == "soccer":
        insights["soccer"] = {
            "home_cs": g("home_clean_sheet_rate"),
            "away_cs": g("away_clean_sheet_rate"),
            "home_draw": g("home_draw_rate"),
            "away_draw": g("away_draw_rate"),
        }
    elif sport == "basketball":
        insights["basketball"] = {
            "home_pace": g("home_pace"),
            "away_pace": g("away_pace"),
            "close_game_rate": g("home_close_game_rate"),
        }
        insights["injuries"] = {
            "home_impact": g("home_injury_impact"),
            "away_impact": g("away_injury_impact"),
            "impact_diff": g("injury_impact_diff"),
        }

    return insights


def generate_commentary(insights: dict, pred_prob: float, handicap_ev: float, sport: str) -> str:
    """数値insightsから日本語の解説コメンタリーを生成"""
    home = insights.get("home_name", "ホーム")
    away = insights.get("away_name", "アウェイ")
    h_team = insights.get("handicap_team", "")
    lines = []

    # ── 総合判定 (3段階) ──
    conf = "高" if pred_prob >= 0.6 else "中" if pred_prob >= 0.5 else "やや低"
    ev_label = _ev_tier_label(handicap_ev, sport)
    lines.append(f"【総合判定】{ev_label}")
    lines.append(f"{h_team}のハンデ勝利予測: 確信度{conf}（{pred_prob*100:.1f}%）、EV +{handicap_ev*100:.1f}%")

    # ── Elo ──
    elo = insights.get("elo", {})
    h_elo, a_elo = elo.get("home", 0), elo.get("away", 0)
    diff = elo.get("diff", 0)
    if h_elo > 0 and a_elo > 0:
        stronger = home if diff > 0 else away
        weaker = away if diff > 0 else home
        abs_diff = abs(diff)
        if abs_diff > 100:
            lines.append(f"【実力差】Eloレーティングでは{stronger}が{weaker}を大きく上回っており（差{abs_diff:.0f}）、地力の差は明確。")
        elif abs_diff > 30:
            lines.append(f"【実力差】{stronger}がEloで優勢（差{abs_diff:.0f}）だが、拮抗した範囲。番狂わせも十分あり得る。")
        else:
            lines.append(f"【実力差】両チームのEloはほぼ互角（差{abs_diff:.0f}）。実力的には五分五分の対戦。")

    # ── 直近フォーム ──
    form = insights.get("recent_form", {})
    h_wr5, a_wr5 = form.get("home_wr5", 0), form.get("away_wr5", 0)
    h_wr10, a_wr10 = form.get("home_wr10", 0), form.get("away_wr10", 0)
    h_streak, a_streak = form.get("home_streak", 0), form.get("away_streak", 0)

    if h_wr5 > 0 or a_wr5 > 0:
        parts = []
        # ホームの調子
        if h_wr5 >= 0.8:
            parts.append(f"{home}は直近5試合で{h_wr5*100:.0f}%の高勝率と絶好調")
        elif h_wr5 >= 0.6:
            parts.append(f"{home}は直近5試合勝率{h_wr5*100:.0f}%と好調を維持")
        elif h_wr5 <= 0.2:
            parts.append(f"{home}は直近5試合勝率{h_wr5*100:.0f}%と深刻な不振")
        elif h_wr5 <= 0.4:
            parts.append(f"{home}は直近5試合勝率{h_wr5*100:.0f}%とやや低迷")

        # アウェイの調子
        if a_wr5 >= 0.8:
            parts.append(f"{away}は{a_wr5*100:.0f}%と波に乗っている")
        elif a_wr5 >= 0.6:
            parts.append(f"{away}は{a_wr5*100:.0f}%と安定")
        elif a_wr5 <= 0.2:
            parts.append(f"{away}は{a_wr5*100:.0f}%と苦しい状況")
        elif a_wr5 <= 0.4:
            parts.append(f"{away}は{a_wr5*100:.0f}%と調子を落としている")

        if parts:
            lines.append("【チーム状態】" + "。".join(parts) + "。")

        # 連勝/連敗
        streak_parts = []
        if h_streak >= 3:
            streak_parts.append(f"{home}は{int(h_streak)}連勝中")
        elif h_streak <= -3:
            streak_parts.append(f"{home}は{int(abs(h_streak))}連敗中")
        if a_streak >= 3:
            streak_parts.append(f"{away}は{int(a_streak)}連勝中")
        elif a_streak <= -3:
            streak_parts.append(f"{away}は{int(abs(a_streak))}連敗中")
        if streak_parts:
            lines.append("【勢い】" + "、".join(streak_parts) + "。")

    # ── 得失点トレンド ──
    trend = insights.get("form_trend", {})
    h_scored, a_scored = trend.get("home_scored", 0), trend.get("away_scored", 0)
    h_conceded, a_conceded = trend.get("home_conceded", 0), trend.get("away_conceded", 0)
    if h_scored > 0 or a_scored > 0:
        parts = []
        if h_scored > a_scored * 1.3:
            parts.append(f"{home}の攻撃力が{away}を上回る（得点傾向 {h_scored:.1f} vs {a_scored:.1f}）")
        elif a_scored > h_scored * 1.3:
            parts.append(f"{away}の攻撃力が{home}を上回る（得点傾向 {a_scored:.1f} vs {h_scored:.1f}）")
        if h_conceded < a_conceded * 0.7:
            parts.append(f"{home}の守備が堅い（失点傾向 {h_conceded:.1f} vs {a_conceded:.1f}）")
        elif a_conceded < h_conceded * 0.7:
            parts.append(f"{away}の守備が堅い（失点傾向 {a_conceded:.1f} vs {h_conceded:.1f}）")
        if parts:
            lines.append("【攻守バランス】" + "。".join(parts) + "。")

    # ── 対戦成績 ──
    h2h = insights.get("h2h", {})
    h2h_count = h2h.get("count", 0)
    h2h_home_wr, h2h_away_wr = h2h.get("home_wr", 0), h2h.get("away_wr", 0)
    if h2h_count >= 3:
        if h2h_home_wr >= 0.7:
            lines.append(f"【相性】過去{int(h2h_count)}回の対戦で{home}が{h2h_home_wr*100:.0f}%勝利と圧倒的な相性の良さ。")
        elif h2h_away_wr >= 0.7:
            lines.append(f"【相性】過去{int(h2h_count)}回の対戦で{away}が{h2h_away_wr*100:.0f}%勝利。{home}にとって苦手な相手。")
        elif h2h_count >= 5:
            lines.append(f"【相性】過去{int(h2h_count)}回の対戦は{home} {h2h_home_wr*100:.0f}% vs {away} {h2h_away_wr*100:.0f}%と拮抗。")
    elif h2h_count > 0:
        lines.append(f"【相性】対戦実績が少なく（{int(h2h_count)}試合）、相性の判断は難しい。")

    # ── 日程 ──
    sched = insights.get("schedule", {})
    h_rest, a_rest = sched.get("home_rest", 0), sched.get("away_rest", 0)
    h_b2b, a_b2b = sched.get("home_b2b", 0), sched.get("away_b2b", 0)
    sched_parts = []
    if h_rest > 0 and a_rest > 0:
        if h_rest >= a_rest + 2:
            sched_parts.append(f"{home}は中{int(h_rest)}日で余裕あり、{away}は中{int(a_rest)}日で疲労懸念")
        elif a_rest >= h_rest + 2:
            sched_parts.append(f"{away}は中{int(a_rest)}日で余裕あり、{home}は中{int(h_rest)}日で疲労懸念")
    if h_b2b > 0:
        sched_parts.append(f"{home}は連戦中")
    if a_b2b > 0:
        sched_parts.append(f"{away}は連戦中")
    if sched_parts:
        lines.append("【日程】" + "。".join(sched_parts) + "。")

    # ── スポーツ固有 ──
    if sport == "baseball" and "pitcher" in insights:
        p = insights["pitcher"]
        h_era, a_era = p.get("home_era", 0), p.get("away_era", 0)
        h_pwr, a_pwr = p.get("home_wr", 0), p.get("away_wr", 0)
        if h_era > 0 and a_era > 0:
            parts = []
            if h_era < a_era * 0.7:
                parts.append(f"{home}先発の防御率{h_era:.2f}が{away}先発{a_era:.2f}を大きく上回る")
            elif a_era < h_era * 0.7:
                parts.append(f"{away}先発の防御率{a_era:.2f}が{home}先発{h_era:.2f}を大きく上回る")
            if h_pwr >= 0.65:
                parts.append(f"{home}先発は勝率{h_pwr*100:.0f}%の好投手")
            if a_pwr >= 0.65:
                parts.append(f"{away}先発は勝率{a_pwr*100:.0f}%の好投手")
            if parts:
                lines.append("【先発投手】" + "。".join(parts) + "。")

    elif sport == "soccer" and "soccer" in insights:
        s = insights["soccer"]
        h_cs, a_cs = s.get("home_cs", 0), s.get("away_cs", 0)
        h_draw, a_draw = s.get("home_draw", 0), s.get("away_draw", 0)
        parts = []
        if h_cs >= 0.4:
            parts.append(f"{home}はクリーンシート率{h_cs*100:.0f}%と堅守")
        if a_cs >= 0.4:
            parts.append(f"{away}はクリーンシート率{a_cs*100:.0f}%と堅守")
        if h_draw >= 0.35 or a_draw >= 0.35:
            parts.append("引き分けの可能性にも注意が必要")
        if parts:
            lines.append("【サッカー特性】" + "。".join(parts) + "。")

    elif sport == "basketball" and "basketball" in insights:
        b = insights["basketball"]
        h_pace, a_pace = b.get("home_pace", 0), b.get("away_pace", 0)
        if h_pace > 0 and a_pace > 0:
            if h_pace > 100 and a_pace > 100:
                lines.append("【展開予想】両チームともハイペースで、ハイスコアゲームの展開が予想される。")
            elif h_pace < 95 and a_pace < 95:
                lines.append("【展開予想】両チームともロースコア傾向。接戦になる可能性が高い。")

        # 欠場影響
        inj = insights.get("injuries", {})
        h_inj, a_inj = inj.get("home_impact", 0), inj.get("away_impact", 0)
        if h_inj > 0 or a_inj > 0:
            if h_inj > a_inj + 0.5:
                lines.append(f"【欠場情報】{home}に主力欠場あり（影響度{h_inj:.1f}）。{away}有利の材料。")
            elif a_inj > h_inj + 0.5:
                lines.append(f"【欠場情報】{away}に主力欠場あり（影響度{a_inj:.1f}）。{home}有利の材料。")
            else:
                lines.append(f"【欠場情報】両チームとも欠場あり（{home}:{h_inj:.1f}, {away}:{a_inj:.1f}）。")

    # ── 確率ベースPICK補足 ──
    prob_thresh = PROB_PICK_THRESHOLD.get(sport, 0.55)
    if pred_prob >= prob_thresh and handicap_ev < 0:
        lines.append(f"【注目】確率は高い（{pred_prob*100:.1f}%）がハンデ構造上EVマイナス。参考情報として推奨。")

    # ── 確信度 ──
    confidence = "high" if pred_prob >= 0.60 else "medium" if pred_prob >= 0.50 else "low"
    conf_label = {"high": "高確信", "medium": "中確信", "low": "低確信"}[confidence]
    lines.append(f"【確信度】{conf_label}")

    # ── まとめ ──
    tier = _ev_tier(handicap_ev, sport)
    if tier == "high":
        lines.append(f"【結論】{h_team}ハンデ有利と判断。EV +{handicap_ev*100:.1f}%で自信を持って推奨。")
    elif tier == "mid":
        lines.append(f"【結論】{h_team}ハンデ有利と判断。EV +{handicap_ev*100:.1f}%で賭ける価値あり。")
    elif tier == "low":
        lines.append(f"【結論】{h_team}ハンデ有利だが期待値は小さい。慎重な判断を推奨。")
    else:
        if pred_prob >= prob_thresh:
            lines.append(f"【結論】EVは低いが確率{pred_prob*100:.1f}%で{h_team}ハンデ有利の可能性。参考推奨。")
        else:
            lines.append(f"【結論】{h_team}ハンデは期待値が低く、見送りが無難。")

    return "\n".join(lines), confidence


def generate_sns_text(pred: dict) -> str:
    """SNS投稿用テキストを生成"""
    home = pred["home_team"]
    away = pred["away_team"]
    h_team = pred["handicap_team"]
    prob = pred["pred_prob"]
    ev = pred["handicap_ev"]
    kelly = pred["kelly_fraction"]
    sport = pred["sport"]
    match_date = pred.get("date", "")
    game_time = pred.get("game_time", "")
    league = pred.get("league", "")
    h_val = pred.get("handicap_value", 0)
    ins = pred.get("insights", {})

    # スポーツ絵文字
    emoji = {"baseball": "⚾", "soccer": "⚽", "basketball": "🏀"}.get(sport, "🏆")

    # リーグ名
    league_names = {
        "npb": "NPB", "mlb": "MLB", "wbc": "WBC", "premier": "プレミアリーグ",
        "laliga": "ラ・リーガ", "seriea": "セリエA", "bundesliga": "ブンデスリーガ",
        "ligue1": "リーグ・アン", "eredivisie": "エールディヴィジ",
        "cl": "チャンピオンズリーグ", "nba": "NBA",
    }
    lg_name = league_names.get(league, league.upper())

    # 日付フォーマット
    date_str = ""
    if match_date:
        parts = match_date.split("-")
        if len(parts) == 3:
            date_str = f"{int(parts[1])}/{int(parts[2])}"
    time_str = f" {game_time}" if game_time else ""

    # 確信度
    confidence = pred.get("confidence", "low")
    conf_emoji = {"high": "🔥", "medium": "⚡", "low": "🔍"}.get(confidence, "🔍")

    # 判定 (3段階 + 確率PICK)
    tier = _ev_tier(ev, sport)
    prob_thresh = PROB_PICK_THRESHOLD.get(sport, 0.55)
    if tier == "high":
        verdict = "🔥 自信あり"
        conf_bar = "★★★"
    elif tier == "mid":
        verdict = "⚡ 期待値あり"
        conf_bar = "★★☆"
    elif tier == "low":
        verdict = "🔍 様子見"
        conf_bar = "★☆☆"
    elif prob >= prob_thresh:
        verdict = f"{conf_emoji} 確率推奨"
        conf_bar = "★★☆" if confidence == "high" else "★☆☆"
    else:
        verdict = "❌ PASS"
        conf_bar = "☆☆☆"

    # ハンデ表示
    h_sign = "+" if h_val > 0 else ""
    handi_str = f"{h_team} ({h_sign}{h_val})"

    lines = [
        f"{emoji} {lg_name}",
        f"📅 {date_str}{time_str}",
        f"",
        f"🏟 {home} vs {away}",
        f"",
        f"📊 AI予測",
        f"┣ ハンデ: {handi_str}",
        f"┣ 勝率: {prob*100:.1f}%",
        f"┣ 期待値: +{ev*100:.1f}%",
        f"┣ 確信度: {conf_emoji} {confidence.upper()}",
        f"┗ 判定: {verdict} {conf_bar}",
    ]

    # 根拠を簡潔に
    reasons = []
    form = ins.get("recent_form", {})
    h_wr5, a_wr5 = form.get("home_wr5", 0), form.get("away_wr5", 0)
    h_streak = form.get("home_streak", 0)
    a_streak = form.get("away_streak", 0)

    if h_wr5 >= 0.8:
        reasons.append(f"✅ {home}直近5試合{h_wr5*100:.0f}%と絶好調")
    elif h_wr5 <= 0.2:
        reasons.append(f"⚠️ {home}直近5試合{h_wr5*100:.0f}%と不振")
    if a_wr5 >= 0.8:
        reasons.append(f"✅ {away}直近5試合{a_wr5*100:.0f}%と絶好調")
    elif a_wr5 <= 0.2:
        reasons.append(f"⚠️ {away}直近5試合{a_wr5*100:.0f}%と不振")

    if h_streak >= 3:
        reasons.append(f"🔥 {home}{int(h_streak)}連勝中")
    elif h_streak <= -3:
        reasons.append(f"📉 {home}{int(abs(h_streak))}連敗中")
    if a_streak >= 3:
        reasons.append(f"🔥 {away}{int(a_streak)}連勝中")
    elif a_streak <= -3:
        reasons.append(f"📉 {away}{int(abs(a_streak))}連敗中")

    elo = ins.get("elo", {})
    elo_diff = abs(elo.get("diff", 0))
    if elo_diff > 100:
        stronger = home if elo.get("diff", 0) > 0 else away
        reasons.append(f"💪 {stronger}が地力で大きくリード")

    h2h = ins.get("h2h", {})
    if h2h.get("count", 0) >= 5:
        h_h2h = h2h.get("home_wr", 0)
        if h_h2h >= 0.7:
            reasons.append(f"📈 {home}が対戦成績で圧倒")
        elif h_h2h <= 0.3:
            reasons.append(f"📈 {away}が対戦成績で圧倒")

    if sport == "baseball" and "pitcher" in ins:
        p = ins["pitcher"]
        h_era, a_era = p.get("home_era", 0), p.get("away_era", 0)
        if h_era > 0 and a_era > 0:
            if h_era < a_era * 0.7:
                reasons.append(f"⚾ {home}先発の防御率{h_era:.2f}が優秀")
            elif a_era < h_era * 0.7:
                reasons.append(f"⚾ {away}先発の防御率{a_era:.2f}が優秀")

    if reasons:
        lines.append("")
        lines.append("💡 注目ポイント")
        for r in reasons[:4]:
            lines.append(r)

    # 逆張り
    if pred.get("contrarian_team"):
        c_ev = pred.get("contrarian_ev", 0)
        lines.append("")
        lines.append(f"🔄 逆張り: {pred['contrarian_team']}も期待値あり（+{c_ev*100:.1f}%）")

    lines.append("")
    lines.append("#スポーツベット #AI予測 #ハンデ予測")

    return "\n".join(lines)


def generate_predictions(sport: str, version: str = None, days_back: int = 7) -> list[dict]:
    """予測を生成して辞書リストで返す"""
    if version is None:
        version = SPORT_MODEL_VERSION.get(sport, "v1")

    # バスケ予測時は欠場キャッシュをクリアして最新情報を取得
    if sport == "basketball":
        from features.injury_features import clear_injury_cache
        clear_injury_cache()

    model = load_model(sport, version)
    df = load_matches_df(sport_code=sport, include_scheduled=True)

    if df.empty:
        return []

    df, feature_cols = build_features(df, sport)

    if sport == "basketball":
        # バスケ: ホーム勝率予測 → ハンデ有利チーム勝率に変換
        df["home_win_prob"] = model.predict_proba(df[feature_cols])
        df["handicap_team_is_home"] = (df["handicap_team_id"] == df["home_team_id"]).astype(int)
        df["pred_prob"] = df.apply(
            lambda r: home_prob_to_handicap_prob(
                r["home_win_prob"],
                bool(r["handicap_team_is_home"]),
                r.get("handicap_value", 0),
                sport=sport,
            ), axis=1
        )
    else:
        # 野球・サッカー: ハンデカバー確率を直接予測
        df["pred_prob"] = model.predict_proba(df[feature_cols])
    # EV計算: モデルがカバー確率を直接予測するので
    # ハンデ表示を取得してハンデ構造に基づく正確なEVを算出
    session_ev = get_session()
    def _calc_ev_row(row):
        mid = int(row["match_id"])
        hd = session_ev.query(HandicapData).filter_by(match_id=mid).first()
        h_display = hd.handicap_display or "" if hd else ""
        h_value = hd.handicap_value or 0 if hd else 0
        return calculate_handicap_ev(row["pred_prob"], h_value, h_display, sport)
    df["handicap_ev"] = df.apply(_calc_ev_row, axis=1)
    df["kelly_frac"] = df["pred_prob"].apply(lambda p: kelly_fraction(p))

    cutoff = date.today() - timedelta(days=days_back)
    # date列の型を統一（datetime64 or date object 両対応）
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        cutoff = pd.Timestamp(cutoff)
    recent = df[df["date"] >= cutoff].sort_values("date", ascending=False)

    if recent.empty:
        return []

    session = get_session()
    predictions = []

    for _, row in recent.iterrows():
        mid = int(row["match_id"])
        home = session.get(Team, int(row["home_team_id"]))
        away = session.get(Team, int(row["away_team_id"]))

        hd = session.query(HandicapData).filter_by(match_id=mid).first()
        h_team, h_val, result_type, payout, status = "", 0, "", 0, "pending"

        if hd:
            ht = session.get(Team, hd.handicap_team_id)
            h_team = ht.name if ht else ""
            h_val = hd.handicap_value or 0
            result_type = hd.result_type or ""
            payout = hd.payout_rate or 0
            if result_type:
                status = "settled"

        match = session.get(Match, mid)
        match_date = str(match.date) if match else ""
        game_time = str(match.time)[:5] if match and match.time else ""

        prob = float(row["pred_prob"])
        ev = float(row["handicap_ev"])

        league_code = match.league_code if match else ""
        league_ev_thresh = LEAGUE_EV_THRESHOLD.get(league_code, MIN_EV_THRESHOLD)
        # PICK if EV >= threshold OR prob >= prob_threshold
        ev_pick = ev >= league_ev_thresh
        prob_pick = prob >= PROB_PICK_THRESHOLD.get(sport, 0.55)
        is_recommended = 1 if (ev_pick or prob_pick) else 0

        h_display = hd.handicap_display or "" if hd else ""

        pred = {
            "match_id": mid,
            "model_version": f"{sport}_{version}",
            "sport": sport,
            "league": league_code,
            "date": match_date,
            "home_team": home.name if home else "",
            "away_team": away.name if away else "",
            "handicap_team": h_team,
            "handicap_value": h_val,
            "handicap_display": h_display,
            "pred_prob": round(prob, 6),
            "handicap_ev": round(ev, 6),
            "kelly_fraction": round(float(row["kelly_frac"]), 6),
            "result_type": result_type,
            "payout_rate": payout,
            "status": status,
            "game_time": game_time,
            "ev_threshold": league_ev_thresh,
            "ev_tier": _ev_tier(ev, sport),
            "is_recommended": is_recommended,
            "contrarian_team": "",
            "contrarian_ev": 0,
            "contrarian_kelly": 0,
            "insights": extract_insights(
                row, sport,
                home.name if home else "",
                away.name if away else "",
                h_team,
            ),
            "commentary": "",
        }
        # 解説コメンタリーを生成
        commentary_text, confidence = generate_commentary(pred["insights"], prob, ev, sport)
        pred["commentary"] = commentary_text
        pred["confidence"] = confidence
        pred["sns_text"] = generate_sns_text(pred)

        # 逆張り: 通常EVが基準未満（PASS）なら逆側を同じレコードに追加
        if ev < CONTRARIAN_MIN_EV_THRESHOLD:
            contrarian_ev = calculate_contrarian_ev(prob, h_val, h_display, sport)
            if contrarian_ev >= CONTRARIAN_MIN_EV_THRESHOLD:
                if h_team == (home.name if home else ""):
                    contra_team = away.name if away else ""
                else:
                    contra_team = home.name if home else ""
                contra_prob = 1.0 - prob - 0.02
                pred["contrarian_team"] = contra_team
                pred["contrarian_ev"] = round(contrarian_ev, 6)
                pred["contrarian_kelly"] = round(kelly_fraction(contra_prob), 6)

        predictions.append(pred)

    session.close()
    return predictions


def save_to_turso(predictions: list[dict]) -> int:
    """予測結果をTursoにlibsql_client経由で保存（turso CLI不要）"""
    if not predictions:
        return 0

    import libsql_client

    turso_url = os.environ.get("TURSO_DATABASE_URL", "")
    turso_token = os.environ.get("TURSO_AUTH_TOKEN", "")

    # .envから読み込み（フォールバック）
    if not turso_url or not turso_token:
        for env_path in [
            os.path.join(os.path.dirname(__file__), ".env"),
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

    # libsql_client needs https:// URL (not libsql://) for HTTP mode
    http_url = turso_url.replace("libsql://", "https://")
    client = libsql_client.create_client_sync(url=http_url, auth_token=turso_token)

    # Deduplicate: keep only highest EV per match (date + home + away)
    dedup_map = {}
    for p in predictions:
        h, a = sorted([p["home_team"], p["away_team"]])
        key = f"{p['date']}_{h}_{a}"
        existing = dedup_map.get(key)
        if not existing or p["handicap_ev"] > existing["handicap_ev"]:
            dedup_map[key] = p
    predictions = list(dedup_map.values())

    # Build batch statements
    stmts = []

    # Delete old pending predictions for the dates+sport being updated
    dates_to_clean = set(p["date"] for p in predictions)
    sports_to_clean = set(p["sport"] for p in predictions)
    for d in dates_to_clean:
        for s in sports_to_clean:
            stmts.append(libsql_client.Statement(
                "DELETE FROM predictions WHERE date = ? AND sport = ? AND status = 'pending'",
                [d, s],
            ))

    # Insert predictions
    for p in predictions:
        stmts.append(libsql_client.Statement(
            "INSERT OR REPLACE INTO predictions "
            "(match_id, model_version, sport, league, date, home_team, away_team, "
            "handicap_team, handicap_value, handicap_display, pred_prob, handicap_ev, kelly_fraction, "
            "result_type, payout_rate, status, game_time, "
            "ev_threshold, ev_tier, is_recommended, "
            "contrarian_team, contrarian_ev, contrarian_kelly, insights, commentary, sns_text) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                p["match_id"], p["model_version"], p["sport"], p.get("league", ""),
                p["date"], p["home_team"], p["away_team"],
                p["handicap_team"], p["handicap_value"], p.get("handicap_display", ""),
                p["pred_prob"], p["handicap_ev"], p["kelly_fraction"],
                p["result_type"], p["payout_rate"], p["status"], p.get("game_time", ""),
                p.get("ev_threshold", 0.08), p.get("ev_tier", "pass"), p.get("is_recommended", 0),
                p.get("contrarian_team", ""), p.get("contrarian_ev", 0), p.get("contrarian_kelly", 0),
                json.dumps(p.get("insights", {}), ensure_ascii=False),
                p.get("commentary", ""), p.get("sns_text", ""),
            ],
        ))

    # Execute all in one batch (single HTTP request)
    try:
        client.batch(stmts)
        saved = len(predictions)
    except Exception as e:
        logger.error(f"Batch save failed: {e}")
        saved = 0

    client.close()
    return saved


class PredictionHandler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json({"status": "ok", "server": "prediction_server"})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/predict":
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len else {}

            sport = body.get("sport", "basketball")
            leagues = SPORT_TO_LEAGUES.get(sport, [])
            date_str = body.get("date")
            target_date = None
            if date_str:
                from scraper.date_utils import parse_date
                target_date = parse_date(date_str.replace("-", ""))

            try:
                # Step 1: Scrape
                logger.info(f"Step 1: Scraping {leagues} ...")
                scrape_results = scrape_today(leagues, target_date)

                # Step 1.5: Fetch bookmaker odds
                logger.info(f"Step 1.5: Fetching bookmaker odds ...")
                try:
                    from scraper.odds_manager import OddsManager
                    odds_mgr = OddsManager()
                    odds_count = 0
                    for lg in leagues:
                        try:
                            odds_count += odds_mgr.scrape_odds_api(lg)
                        except Exception as e:
                            logger.warning(f"Odds API failed for {lg}: {e}")
                    odds_mgr.close()
                    logger.info(f"Fetched {odds_count} bookmaker odds")
                except Exception as e:
                    logger.warning(f"Odds fetch skipped: {e}")

                # Step 2: Generate predictions
                logger.info(f"Step 2: Generating predictions for {sport} ...")
                predictions = generate_predictions(sport, days_back=30)

                # Step 3: Save to Turso
                logger.info(f"Step 3: Saving {len(predictions)} predictions to Turso ...")
                saved = save_to_turso(predictions)

                self._json({
                    "status": "ok",
                    "sport": sport,
                    "scrape": scrape_results,
                    "predictions_count": len(predictions),
                    "saved_to_turso": saved,
                    "pending": sum(1 for p in predictions if p["status"] == "pending"),
                    "settled": sum(1 for p in predictions if p["status"] == "settled"),
                })

            except Exception as e:
                logger.exception("Prediction error")
                self._json({"status": "error", "error": str(e)}, 500)
        elif parsed.path == "/settle":
            try:
                from settle_predictions import settle_predictions
                result = settle_predictions()
                self._json({"status": "ok", **result})
            except Exception as e:
                logger.exception("Settle error")
                self._json({"status": "error", "error": str(e)}, 500)

        elif parsed.path == "/refresh":
            # ハンデスクレイプ + オッズ取得 + 予測 + 乖離検出を一括実行
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(content_len)) if content_len else {}
                sport = body.get("sport")

                if sport:
                    sports_list = [sport]
                else:
                    sports_list = list(SPORT_TO_LEAGUES.keys())

                # Step 1: ハンデスクレイプ
                all_leagues = []
                for s in sports_list:
                    all_leagues.extend(SPORT_TO_LEAGUES.get(s, []))

                scrape_results = scrape_today(all_leagues)

                # Step 2: オッズ取得
                odds_count = 0
                try:
                    from scraper.odds_manager import OddsManager
                    odds_mgr = OddsManager()
                    for lg in all_leagues:
                        try:
                            odds_count += odds_mgr.scrape_odds_api(lg)
                        except Exception:
                            pass
                    odds_mgr.close()
                except Exception:
                    pass

                # Step 3: 予測生成 + 保存
                total_preds = 0
                total_saved = 0
                for s in sports_list:
                    try:
                        preds = generate_predictions(s, days_back=7)
                        total_preds += len(preds)
                        saved = save_to_turso(preds)
                        total_saved += saved
                    except Exception as e:
                        logger.warning(f"Predict error for {s}: {e}")

                # Step 4: 乖離検出
                edge_count = 0
                try:
                    from line_edge_detector import run_detection
                    edge_result = run_detection(min_diff=0.5)
                    edge_count = edge_result.get("detected", 0)
                except Exception:
                    pass

                self._json({
                    "status": "ok",
                    "scrape": scrape_results,
                    "odds": odds_count,
                    "predictions": total_preds,
                    "saved": total_saved,
                    "edges": edge_count,
                })
            except Exception as e:
                logger.exception("Refresh error")
                self._json({"status": "error", "error": str(e)}, 500)

        elif parsed.path == "/injuries":
            try:
                from scraper.injury_scraper import get_injury_alerts
                alerts = get_injury_alerts(min_impact=0.5)
                self._json({"status": "ok", "alerts": alerts, "count": len(alerts)})
            except Exception as e:
                logger.exception("Injury scrape error")
                self._json({"status": "error", "error": str(e)}, 500)

        elif parsed.path == "/edges":
            try:
                from line_edge_detector import run_detection
                content_len = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(content_len)) if content_len else {}
                min_diff = body.get("min_diff", 0.5)
                result = run_detection(min_diff=min_diff)
                self._json({"status": "ok", **result})
            except Exception as e:
                logger.exception("Edge detection error")
                self._json({"status": "error", "error": str(e)}, 500)

        elif parsed.path == "/snapshot":
            try:
                from snapshot_lines import take_snapshots, export_snapshots_to_turso
                content_len = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(content_len)) if content_len else {}
                snap_type = body.get("type")
                result = take_snapshots(snap_type)
                exported = export_snapshots_to_turso()
                self._json({"status": "ok", **result, "exported_to_turso": exported})
            except Exception as e:
                logger.exception("Snapshot error")
                self._json({"status": "error", "error": str(e)}, 500)

        else:
            self._json({"error": "not found"}, 404)


def main():
    port = 8888
    server = HTTPServer(("0.0.0.0", port), PredictionHandler)
    logger.info(f"Prediction server running on http://localhost:{port}")
    logger.info("POST /predict  {{\"sport\": \"basketball\"}} to generate predictions")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
