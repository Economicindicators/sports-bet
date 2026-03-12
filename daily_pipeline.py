"""統合デイリーパイプライン: scrape → predict → save → settle

export_predictions.py と daily_predict.sh の処理を統合。
prediction_server.py の generate_predictions() / save_to_turso() を再利用。

Usage:
    python daily_pipeline.py                    # 全スポーツ実行
    python daily_pipeline.py --sport baseball   # 野球のみ
    python daily_pipeline.py --dry-run          # DB保存せずに確認
    python daily_pipeline.py --days-back 14     # 14日分の予測生成
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta

from config.constants import ALL_LEAGUES, LEAGUE_TO_SPORT, SPORT_TO_LEAGUES, SPORT_MODEL_VERSION
from database.models import init_db
from scraper.manager import ScrapeManager
from scraper.date_utils import format_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def scrape_leagues(leagues: list[str], days: list[date]) -> dict:
    """指定リーグ・日付のデータをスクレイプ"""
    manager = ScrapeManager(use_cache=False)
    results = {}
    for league in leagues:
        for d in days:
            try:
                count = manager.scrape_and_save(league, d)
                results[f"{league}/{d}"] = count
                if count > 0:
                    logger.info(f"Scraped {league}/{format_date(d)}: {count} matches")
            except Exception as e:
                results[f"{league}/{d}"] = f"error: {e}"
                logger.error(f"Scrape error {league}/{d}: {e}")
    return results


def fetch_odds(leagues: list[str]) -> int:
    """ブックメーカーオッズを取得 (ODDS_API_KEY設定時のみ)"""
    if not os.environ.get("ODDS_API_KEY"):
        logger.info("ODDS_API_KEY not set, skipping odds fetch")
        return 0

    try:
        from scraper.odds_manager import OddsManager
        mgr = OddsManager()
        count = 0
        for lg in leagues:
            try:
                count += mgr.scrape_odds_api(lg)
            except Exception as e:
                logger.warning(f"Odds API failed for {lg}: {e}")
        mgr.close()
        logger.info(f"Fetched {count} bookmaker odds")
        return count
    except Exception as e:
        logger.warning(f"Odds fetch skipped: {e}")
        return 0


def run_pipeline(
    sports: list[str] | None = None,
    days_back: int = 7,
    dry_run: bool = False,
) -> dict:
    """メインパイプライン: scrape → predict → save → settle"""
    from prediction_server import generate_predictions, save_to_turso

    if sports is None:
        sports = list(SPORT_TO_LEAGUES.keys())

    today = date.today()
    yesterday = today - timedelta(days=1)

    # 対象リーグを収集
    target_leagues = []
    for sport in sports:
        target_leagues.extend(SPORT_TO_LEAGUES.get(sport, []))

    # Step 1: Scrape (today + yesterday)
    logger.info(f"=== Step 1: Scrape {len(target_leagues)} leagues ===")
    scrape_results = scrape_leagues(target_leagues, [today, yesterday])

    # Step 2: Fetch odds
    logger.info("=== Step 2: Fetch bookmaker odds ===")
    odds_count = fetch_odds(target_leagues)

    # Step 3: Generate & save predictions per sport
    total_predictions = 0
    total_saved = 0

    for sport in sports:
        logger.info(f"=== Step 3: Predict {sport} (model {SPORT_MODEL_VERSION.get(sport, 'v1')}) ===")
        try:
            predictions = generate_predictions(sport, days_back=days_back)
            total_predictions += len(predictions)
            logger.info(f"  Generated {len(predictions)} predictions for {sport}")

            if dry_run:
                logger.info(f"  [DRY-RUN] Skipping Turso save")
                for p in predictions[:5]:
                    logger.info(
                        f"    {p['date']} {p['home_team']} vs {p['away_team']} "
                        f"EV={p['handicap_ev']:.4f} Kelly={p['kelly_fraction']:.4f}"
                    )
                if len(predictions) > 5:
                    logger.info(f"    ... and {len(predictions) - 5} more")
            else:
                saved = save_to_turso(predictions)
                total_saved += saved
                logger.info(f"  Saved {saved} predictions to Turso")

        except Exception as e:
            logger.exception(f"  Prediction error for {sport}: {e}")

    # Step 4: Settle
    if not dry_run:
        logger.info("=== Step 4: Settle predictions ===")
        try:
            from settle_predictions import settle_predictions
            settle_result = settle_predictions()
            logger.info(f"  Settled: {settle_result}")
        except Exception as e:
            logger.warning(f"  Settle error: {e}")

    result = {
        "sports": sports,
        "scrape": scrape_results,
        "odds_fetched": odds_count,
        "predictions": total_predictions,
        "saved": total_saved,
        "dry_run": dry_run,
    }

    logger.info(f"=== Pipeline complete: {total_predictions} predictions, {total_saved} saved ===")
    return result


def main():
    parser = argparse.ArgumentParser(description="Daily prediction pipeline")
    parser.add_argument("--sport", type=str, nargs="+", default=None,
                        help="Sports to process (default: all)")
    parser.add_argument("--days-back", type=int, default=7,
                        help="Days back for prediction window (default: 7)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate predictions without saving to Turso")
    args = parser.parse_args()

    init_db()
    result = run_pipeline(
        sports=args.sport,
        days_back=args.days_back,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        logger.info("Dry-run complete. No data was saved.")


if __name__ == "__main__":
    main()
