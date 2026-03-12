"""ハンデデータ貼り付け → 予測生成スクリプト

使い方:
  python predict_from_paste.py < paste.txt
  python predict_from_paste.py --file paste.txt
  python predict_from_paste.py  # 対話モード（Ctrl+Dで終了）

入力フォーマット（ハンデの嵐のLINE共有形式）:
  <セリエA>
  ボローニャ<0半5>
  23:00
  ヴェローナ

  フィオレンティーナ<0半3>
  23:00
  パルマ
"""

import sys
import re
import logging
import argparse
import subprocess
import tempfile
import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Optional

from features.feature_pipeline import load_matches_df, build_features
from models.training import load_model
from betting.handicap_ev import calculate_handicap_ev, calculate_contrarian_ev
from betting.kelly import kelly_fraction
from config.constants import CONTRARIAN_MIN_EV_THRESHOLD
from database.models import get_session, Match, Team, HandicapData
from scraper.football_hande import parse_hande_result

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# リーグ名 → リーグコード
LEAGUE_ALIASES = {
    "セリエA": "seriea", "セリエ": "seriea",
    "リーガ": "laliga", "ラリーガ": "laliga",
    "ブンデス": "bundesliga", "ブンデスリーガ": "bundesliga",
    "エール": "eredivisie", "エールディビジ": "eredivisie",
    "リグアン": "ligue1", "リーグアン": "ligue1",
    "プレミア": "premier", "プレミアリーグ": "premier",
    "NPB": "npb", "プロ野球": "npb",
    "MLB": "mlb",
    "NBA": "nba",
    "Bリーグ": "bleague",
}

LEAGUE_TO_SPORT = {
    "npb": "baseball", "mlb": "baseball",
    "premier": "soccer", "laliga": "soccer", "seriea": "soccer",
    "bundesliga": "soccer", "ligue1": "soccer", "eredivisie": "soccer",
    "nba": "basketball", "bleague": "basketball",
}


@dataclass
class ParsedMatch:
    league_code: str
    home_team: str
    away_team: str
    handicap_team_side: str  # "home" or "away"
    handicap_display: str    # 原文: "0半5", "0/3" etc.
    handicap_value: float    # 数値: 0.5, 0.0 etc.
    border_pct: float = 50   # ボーダー%: 3→30, 5→50, 7→70, なし→0
    time: Optional[str] = None


def parse_paste(text: str, target_date: date = None) -> list[ParsedMatch]:
    """貼り付けテキストをパースして試合リストを返す"""
    lines = [l.strip() for l in text.strip().split("\n")]
    matches = []
    current_league = None
    i = 0

    while i < len(lines):
        line = lines[i]

        # 空行スキップ
        if not line:
            i += 1
            continue

        # 締切時間の行をスキップ
        if "締切" in line or "時間厳守" in line or "→" in line or line.startswith("＊") or line.startswith("🏀"):
            i += 1
            continue

        # リーグヘッダー: <セリエA> or ＜セリエA＞ or [NBA]
        league_match = re.match(r"[<＜\[](.+?)[>＞\]]", line)
        if league_match:
            league_name = league_match.group(1)
            for alias, code in LEAGUE_ALIASES.items():
                if alias in league_name:
                    current_league = code
                    break
            else:
                logger.warning(f"Unknown league: {league_name}")
                current_league = None
            i += 1
            continue

        if not current_league:
            i += 1
            continue

        # 試合パターン: チーム名<ハンデ> / 時刻 / チーム名<ハンデ>
        # Line 1: ホームチーム（ハンデ付きの場合あり）
        team1_line = line
        team1_name, team1_hande = _extract_team_hande(team1_line)

        if not team1_name:
            i += 1
            continue

        # Line 2: 時刻 (optional)
        game_time = None
        if i + 1 < len(lines):
            time_match = re.match(r"^(\d{1,2}:\d{2})$", lines[i + 1].strip())
            if time_match:
                game_time = time_match.group(1)
                i += 1

        # Line 3: アウェイチーム
        if i + 1 < len(lines):
            team2_line = lines[i + 1].strip()
            team2_name, team2_hande = _extract_team_hande(team2_line)
            i += 1
        else:
            i += 1
            continue

        if not team2_name:
            i += 1
            continue

        # ハンデ情報を決定
        if team1_hande:
            hande_text = team1_hande
            hande_side = "home"
        elif team2_hande:
            hande_text = team2_hande
            hande_side = "away"
        else:
            i += 1
            continue

        # ハンデ値をパース
        parsed = parse_hande_result(hande_text)
        if not parsed:
            logger.warning(f"Cannot parse handicap: {hande_text}")
            i += 1
            continue

        hande_value, result_code = parsed
        # ボーダー%: 末尾数字 × 10 (3→30, 5→50, 7→70)、なし→0
        border_pct = int(result_code) * 10 if result_code.isdigit() else 0

        matches.append(ParsedMatch(
            league_code=current_league,
            home_team=team1_name,
            away_team=team2_name,
            handicap_team_side=hande_side,
            handicap_display=hande_text,
            handicap_value=hande_value,
            border_pct=border_pct,
            time=game_time,
        ))

        i += 1

    return matches


def _extract_team_hande(text: str) -> tuple[str, Optional[str]]:
    """チーム名とハンデ表記を分離: 'ボローニャ<0半5>' → ('ボローニャ', '0半5')"""
    text = text.strip()
    if not text:
        return "", None

    # パターン: チーム名<ハンデ> or チーム名＜ハンデ＞
    m = re.match(r"^(.+?)[<＜]([^>＞]+)[>＞]$", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # ハンデなし
    # 時刻だけの行を除外
    if re.match(r"^\d{1,2}:\d{2}$", text):
        return "", None

    return text.strip(), None


def _similarity(a: str, b: str) -> float:
    """2文字列の類似度（0.0〜1.0）。共通部分文字列ベース"""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # 共通文字数 / 長い方の長さ
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    common = sum(1 for c in shorter if c in longer)
    return common / len(longer)


def find_team_in_db(session, name: str, league_code: str) -> Optional[Team]:
    """DB内のチームを名前で検索（完全一致→部分一致→類似度マッチ）"""
    # 完全一致
    team = session.query(Team).filter_by(name=name, league_code=league_code).first()
    if team:
        return team

    all_teams = session.query(Team).filter_by(league_code=league_code).all()

    # 部分一致（DB側の名前が入力に含まれる、または逆）
    for t in all_teams:
        if t.name in name or name in t.name:
            return t

    # 類似度マッチ（閾値0.6以上で最も高いものを採用）
    best_team = None
    best_score = 0.0
    for t in all_teams:
        score = _similarity(name, t.name)
        if score > best_score:
            best_score = score
            best_team = t
    if best_score >= 0.5:
        logger.info(f"Fuzzy match: '{name}' → '{best_team.name}' (score={best_score:.2f})")
        return best_team

    return None


def predict_matches(parsed: list[ParsedMatch], target_date: date, version: str = "v2") -> list[dict]:
    """パースした試合を予測"""
    session = get_session()
    results = []

    # スポーツごとにグループ化
    by_sport = {}
    for m in parsed:
        sport = LEAGUE_TO_SPORT.get(m.league_code, "soccer")
        by_sport.setdefault(sport, []).append(m)

    for sport, sport_matches in by_sport.items():
        try:
            model = load_model(sport, version)
        except Exception as e:
            logger.warning(f"No model for {sport}/{version}: {e}")
            continue

        df = load_matches_df(sport_code=sport, include_scheduled=True)
        if df.empty:
            continue

        df, feature_cols = build_features(df, sport)
        df["pred_prob"] = model.predict_proba(df[feature_cols])

        for pm in sport_matches:
            home = find_team_in_db(session, pm.home_team, pm.league_code)
            away = find_team_in_db(session, pm.away_team, pm.league_code)

            if not home or not away:
                results.append({
                    "league": pm.league_code,
                    "home_team": pm.home_team,
                    "away_team": pm.away_team,
                    "handicap_team": pm.home_team if pm.handicap_team_side == "home" else pm.away_team,
                    "handicap_display": pm.handicap_display,
                    "handicap_value": pm.handicap_value,
                    "pred_prob": None,
                    "handicap_ev": None,
                    "kelly_frac": None,
                    "error": f"Team not found: {pm.home_team if not home else pm.away_team}",
                })
                continue

            # DBの試合データからこのチーム組み合わせの最新行を探す
            mask = (
                ((df["home_team_id"] == home.team_id) & (df["away_team_id"] == away.team_id)) |
                ((df["home_team_id"] == away.team_id) & (df["away_team_id"] == home.team_id))
            )
            team_rows = df[mask].sort_values("date", ascending=False)

            if team_rows.empty:
                # チームの最新行を使う（特徴量の近似値として）
                home_mask = (df["home_team_id"] == home.team_id) | (df["away_team_id"] == home.team_id)
                home_rows = df[home_mask].sort_values("date", ascending=False)
                if home_rows.empty:
                    results.append({
                        "league": pm.league_code,
                        "home_team": pm.home_team,
                        "away_team": pm.away_team,
                        "handicap_team": pm.home_team if pm.handicap_team_side == "home" else pm.away_team,
                        "handicap_display": pm.handicap_display,
                        "handicap_value": pm.handicap_value,
                        "pred_prob": None,
                        "handicap_ev": None,
                        "kelly_frac": None,
                        "error": "No match history",
                    })
                    continue
                row = home_rows.iloc[0]
            else:
                row = team_rows.iloc[0]

            prob = float(row["pred_prob"])
            ev = calculate_handicap_ev(prob, border_pct=pm.border_pct)
            kelly = kelly_fraction(prob)
            h_team = pm.home_team if pm.handicap_team_side == "home" else pm.away_team

            results.append({
                "league": pm.league_code,
                "home_team": pm.home_team,
                "away_team": pm.away_team,
                "handicap_team": h_team,
                "handicap_display": pm.handicap_display,
                "handicap_value": pm.handicap_value,
                "pred_prob": round(prob, 6),
                "handicap_ev": round(ev, 6),
                "kelly_frac": round(kelly, 6),
                "bet_type": "normal",
                "game_time": pm.time,
                "error": None,
            })

            # 逆張り判定: 通常EVが基準未満（PASS）なら逆側を検討
            if ev < CONTRARIAN_MIN_EV_THRESHOLD:
                contrarian_ev = calculate_contrarian_ev(prob, border_pct=pm.border_pct)
                if contrarian_ev >= CONTRARIAN_MIN_EV_THRESHOLD:
                    # 逆側のチーム
                    contra_team = pm.away_team if pm.handicap_team_side == "home" else pm.home_team
                    contra_prob = 1.0 - prob - 0.02  # unfavorable probability
                    contra_kelly = kelly_fraction(contra_prob)
                    results.append({
                        "league": pm.league_code,
                        "home_team": pm.home_team,
                        "away_team": pm.away_team,
                        "handicap_team": contra_team,
                        "handicap_display": pm.handicap_display,
                        "handicap_value": pm.handicap_value,
                        "pred_prob": round(contra_prob, 6),
                        "handicap_ev": round(contrarian_ev, 6),
                        "kelly_frac": round(contra_kelly, 6),
                        "bet_type": "contrarian",
                        "game_time": pm.time,
                        "error": None,
                    })

    session.close()
    return results


def save_to_turso(results: list[dict], target_date: date, version: str = "v2") -> int:
    """予測結果をTursoに保存"""
    valid = [r for r in results if r["pred_prob"] is not None and (
        (r.get("bet_type") != "contrarian" and (r["handicap_ev"] or 0) >= 0.05) or
        (r.get("bet_type") == "contrarian" and (r["handicap_ev"] or 0) >= CONTRARIAN_MIN_EV_THRESHOLD)
    )]
    if not valid:
        return 0

    lines = []
    for r in valid:
        sport = LEAGUE_TO_SPORT.get(r["league"], "soccer")
        esc = lambda s: str(s).replace("'", "''")
        is_contrarian = r.get("bet_type") == "contrarian"
        version_suffix = f"{version}_contrarian" if is_contrarian else version
        # チーム名+日付+bet_typeから一意なmatch_idを生成（重複保存を防止）
        key = f"{target_date}_{r['home_team']}_{r['away_team']}{'_contra' if is_contrarian else ''}"
        match_id = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 900000 + 100000
        game_time = r.get('game_time') or ''
        lines.append(
            f"INSERT OR REPLACE INTO predictions "
            f"(match_id, model_version, sport, date, home_team, away_team, "
            f"handicap_team, handicap_value, pred_prob, handicap_ev, kelly_fraction, "
            f"result_type, payout_rate, status, league, handicap_display, game_time) VALUES "
            f"({match_id}, '{sport}_{version_suffix}', '{sport}', "
            f"'{target_date}', '{esc(r['home_team'])}', '{esc(r['away_team'])}', "
            f"'{esc(r['handicap_team'])}', {r['handicap_value']}, "
            f"{r['pred_prob']}, {r['handicap_ev']}, {r['kelly_frac']}, "
            f"'', 0, 'pending', '{r['league']}', '{esc(r['handicap_display'])}', '{esc(game_time)}');"
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
        f.write("\n".join(lines))
        sql_path = f.name

    try:
        result = subprocess.run(
            ["turso", "db", "shell", "sports-bet"],
            stdin=open(sql_path),
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print(f"Turso error: {result.stderr}", file=sys.stderr)
            return 0
    except Exception as e:
        print(f"Turso save failed: {e}", file=sys.stderr)
        return 0

    return len(valid)


def format_output(results: list[dict]) -> str:
    """結果をきれいに表示"""
    if not results:
        return "予測結果なし"

    # リーグごとにグループ化
    by_league = {}
    for r in results:
        by_league.setdefault(r["league"], []).append(r)

    league_names = {
        "seriea": "セリエA", "laliga": "リーガ", "bundesliga": "ブンデス",
        "eredivisie": "エール", "ligue1": "リグアン", "premier": "プレミア",
        "npb": "NPB", "mlb": "MLB", "nba": "NBA",
    }

    lines = []
    for league, matches in by_league.items():
        name = league_names.get(league, league)
        lines.append(f"\n【{name}】")

        # EV降順でソート
        matches.sort(key=lambda x: x["handicap_ev"] or -999, reverse=True)

        for r in matches:
            if r["error"]:
                lines.append(f"  ⚠ {r['home_team']} vs {r['away_team']} → {r['error']}")
                continue

            ev = r["handicap_ev"]
            prob = r["pred_prob"]
            is_contrarian = r.get("bet_type") == "contrarian"

            if is_contrarian:
                mark = "▼"
                label = "[逆張り]"
            else:
                mark = "◎" if ev >= 0.2 else "○" if ev >= 0.1 else "△" if ev >= 0.05 else "×"
                label = ""

            lines.append(
                f"  {mark} {label}{r['handicap_team']} <{r['handicap_display']}>"
                f"  {r['home_team']} vs {r['away_team']}"
                f"  EV:{ev:+.1%}  確率:{prob:.1%}"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ハンデ貼り付け → 予測")
    parser.add_argument("--file", "-f", help="入力ファイル（省略時はstdin）")
    parser.add_argument("--date", "-d", default=None, help="対象日 (YYYYMMDD)")
    parser.add_argument("--version", "-v", default="v2", help="モデルバージョン")
    parser.add_argument("--save", "-s", action="store_true", help="Tursoに保存")
    args = parser.parse_args()

    # 日付
    if args.date:
        from scraper.date_utils import parse_date
        target_date = parse_date(args.date)
    else:
        target_date = date.today()

    # 入力読み込み
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("ハンデデータを貼り付けてください（Ctrl+Dで確定）:", file=sys.stderr)
        text = sys.stdin.read()

    if not text.strip():
        print("入力が空です", file=sys.stderr)
        return

    # パース
    parsed = parse_paste(text, target_date)
    print(f"\n{len(parsed)} 試合を検出", file=sys.stderr)

    if not parsed:
        print("試合が見つかりませんでした", file=sys.stderr)
        return

    # 予測
    results = predict_matches(parsed, target_date, args.version)

    # 表示
    print(format_output(results))

    # Turso保存
    if args.save:
        saved = save_to_turso(results, target_date, args.version)
        print(f"\n→ {saved}件をTursoに保存しました", file=sys.stderr)


if __name__ == "__main__":
    main()
