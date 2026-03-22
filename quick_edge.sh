#!/bin/bash
# 試合直前のオッズ取得 → 乖離検出 → Turso保存
# 使い方: ./quick_edge.sh [nba|premier|npb|all]

cd ~/sports-bet
source .venv/bin/activate
source .env 2>/dev/null

SPORT="${1:-all}"

echo "⚡ オッズ取得 + 乖離検出 ($SPORT)"
echo ""

python3 -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from scraper.odds_manager import OddsManager
from line_edge_detector import run_detection

sport = '$SPORT'

# Step 1: 最新オッズ取得
mgr = OddsManager()
if sport == 'all':
    leagues = ['nba', 'premier', 'laliga', 'seriea', 'bundesliga', 'ligue1', 'eredivisie', 'npb', 'mlb']
elif sport == 'nba':
    leagues = ['nba']
elif sport == 'premier':
    leagues = ['premier']
elif sport == 'npb':
    leagues = ['npb', 'mlb']
else:
    leagues = [sport]

total = 0
for lg in leagues:
    try:
        c = mgr.scrape_odds_api(lg)
        if c > 0:
            print(f'  ✓ {lg}: {c}件')
        total += c
    except Exception as e:
        print(f'  ✗ {lg}: {e}')
mgr.close()
print(f'  合計: {total}件のオッズ取得')
print()

# Step 2: 乖離検出
result = run_detection(min_diff=0.5)
print()
if result['detected'] == 0:
    print('乖離なし。')
"
