# sports-bet — スポーツベット ハンデ予測システム

## 概要
「ハンデの森」「ハンデの嵐」「ハンディマン」の3サイトから日本独自のハンデ値を収集し、
LightGBMでハンデ込み勝敗を予測するシステム。keiba-aiアーキテクチャベース。

## 対象スポーツ
- **野球**: NPB, MLB (ハンデの森) — 投手データ含む
- **サッカー**: Jリーグ, プレミア, ラ・リーガ, セリエA, ブンデスリーガ (ハンデの嵐)
- **バスケ**: NBA, Bリーグ (ハンディマン)

## アーキテクチャ
```
scraper/ → database/ → features/ → models/ → betting/
                                       ↓
                              cli/  (Typer CLI)
                              web/  (Streamlit)
```

## コマンド
```bash
cd ~/sports-bet && source .venv/bin/activate

# DB初期化
python -m cli.main init-db

# スクレイプ (1日)
python -m cli.main scrape npb --date 20250801

# バックフィル (日付範囲)
python -m cli.main backfill npb --start 20240301 --end 20251031

# 学習
python -m cli.main train baseball
python -m cli.main train soccer --league jleague

# 時系列CV
python -m cli.main cv baseball --folds 5

# バックテスト
python -m cli.main backtest baseball --ev 0.05 --periods 4

# 推奨ベット
python -m cli.main recommend baseball --date 20260308

# DB状態 / モデル一覧
python -m cli.main status
python -m cli.main models

# テスト
python -m pytest tests/ -v
```

## 技術スタック
- Python 3.9+ / venv
- SQLAlchemy (SQLite)
- LightGBM (binary classifier, スポーツ別モデル)
- requests + BeautifulSoup (スクレイピング)
- Playwright + Chrome (OddsPortal)
- Typer (CLI) / Streamlit (Web)

## DB (9テーブル)
Sport → League → Team → Match → HandicapData, Prediction, BookmakerOdds
                   ↓
                 Player (投手マスタ等)

## 特徴量 (116個, 野球の場合)
**共通** (32個): ハンデ系(5), チーム成績(12), H2H(4), 順位(5), 日程(6)
**野球固有** (9個): 投手勝率, 投手ERA, ERA差, 登板数, 合計得点
**サッカー固有**: クリーンシート率, 引分率, 平均ゴール数
**バスケ固有**: ペース, 僅差率, 大量リード率
**オッズ** (10個): consensus確率, sharp確率, dispersion, sharp-soft gap, overround, best odds
**セイバーメトリクス** (16個, NPBのみ): チームOPS, RPG, 先発ローテERA/FIP/WHIP/K9, OPS差, ERA差

## ML設計
- ターゲット: ハンデ有利チームが5分勝ち以上 (payout >= 1.5)
- CV: 時系列CV (expanding window + 30日gap)
- EV: P(fav)×1.73 + P(push)×1.0 + P(unfav)×0.5 - 1.0
- Kelly: 1/4 Kelly (conservative)
- バックテスト: Walk-forward (4期間, 再学習+ギャップ)

## 実装状況
- [x] Phase 1: 基盤 (config, DB, handicap_resolver, CLI)
- [x] Phase 2: スクレイパー (base, handenomori, football_hande, b_handicap, manager)
- [x] Phase 3: 特徴量 (41個) + LightGBMモデル + 時系列CV + 学習パイプライン
- [x] Phase 4: EV計算, Kelly基準, フィルタ, Walk-forwardバックテスト
- [x] Phase 5.1: 外部データ統合 — OddsPortal, DELTA Inc (1point02.jp)
- [ ] Phase 5.2: 外部データ統合 — Flashscore, Sports Navi (後回し)
- [ ] Phase 6: Streamlitダッシュボード (7タブ)

## コマンド (追加)
```bash
# セイバーメトリクス表示
python -m cli.main sabermetrics 2025

# オッズ取得 (OddsPortal)
python -m cli.main odds nba --source oddsportal
python -m cli.main odds premier --source oddsportal

# オッズ取得 (The Odds API)
python -m cli.main odds nba --source odds_api
```

## スクレイパー構造
| サイト | クラス | 対象 | 方式 |
|--------|--------|------|------|
| ハンデの森 | HandenomoriScraper | NPB/MLB | requests + BS4 |
| ハンデの嵐 | FootballHandeScraper | サッカー | requests + BS4 |
| ハンディマン | BHandicapScraper | バスケ | requests + BS4 |
| OddsPortal | OddsPortalScraper | 全スポーツ | Playwright + Chrome |
| 1point02.jp | DeltaScraper | NPBセイバー | requests + BS4 |
| The Odds API | odds_api.py | 全スポーツ | REST API |

## keiba-aiからの流用元
- `~/Documents/06_プロダクト開発/keiba-ai/`
- base_scraper, lgbm_model, training, kelly, ev_calculator
