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
- Typer (CLI) / Streamlit (Web)

## DB (8テーブル)
Sport → League → Team → Match → HandicapData, Prediction
                   ↓
                 Player (投手マスタ等)

## 特徴量 (41個, 野球の場合)
**共通** (32個): ハンデ系(5), チーム成績(12), H2H(4), 順位(5), 日程(6)
**野球固有** (9個): 投手勝率, 投手ERA, ERA差, 登板数, 合計得点
**サッカー固有**: クリーンシート率, 引分率, 平均ゴール数
**バスケ固有**: ペース, 僅差率, 大量リード率

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
- [ ] Phase 5: Streamlitダッシュボード (7タブ)

## 次のステップ
1. バックフィル実行 (過去2年分のデータ収集)
2. モデル学習 + CV検証
3. バックテストでEV閾値最適化
4. Streamlitダッシュボード実装

## スクレイパー構造
| サイト | クラス | 対象 | HTML構造 |
|--------|--------|------|----------|
| ハンデの森 | HandenomoriScraper | NPB/MLB | section > div.game-detail2, table.single-handi |
| ハンデの嵐 | FootballHandeScraper | サッカー | table.index-handi-table, win-cell, "0半3"形式 |
| ハンディマン | BHandicapScraper | バスケ | table.single-gamedata, span.handhi |

## keiba-aiからの流用元
- `~/Documents/06_プロダクト開発/keiba-ai/`
- base_scraper, lgbm_model, training, kelly, ev_calculator
