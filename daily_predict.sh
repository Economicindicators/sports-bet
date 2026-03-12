#!/bin/bash
# Daily auto-scrape + predict + upload to Turso
# Runs via LaunchAgent at 08:00 and 20:00
# All logic is in daily_pipeline.py

set -e

LOG_DIR="$HOME/sports-bet/logs"
LOG="$LOG_DIR/daily.log"
mkdir -p "$LOG_DIR"

cd "$HOME/sports-bet"
source .venv/bin/activate
source .env 2>/dev/null || true
export PATH="$HOME/.turso:$PATH"

echo "===== $(date) =====" >> "$LOG"

python daily_pipeline.py --days-back 7 >> "$LOG" 2>&1

echo "[done] $(date)" >> "$LOG"
echo "" >> "$LOG"
