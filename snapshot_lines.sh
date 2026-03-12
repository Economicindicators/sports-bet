#!/bin/bash
# Handicap line snapshot daemon
# Runs continuously, checks every 10 min for matches needing snapshots

LOG_DIR="$HOME/sports-bet/logs"
mkdir -p "$LOG_DIR"

cd "$HOME/sports-bet"
source .venv/bin/activate

echo "===== Snapshot daemon started $(date) =====" >> "$LOG_DIR/snapshot.log"
exec python snapshot_lines.py 2>&1 | tee -a "$LOG_DIR/snapshot.log"
