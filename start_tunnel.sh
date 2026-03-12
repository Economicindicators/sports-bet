#!/bin/bash
# Cloudflare Tunnel を起動し、公開URLをTursoに保存する
# LaunchAgent から呼ばれる

LOG_DIR="$HOME/sports-bet/logs"
mkdir -p "$LOG_DIR"

# Start cloudflared in background, capture output
cloudflared tunnel --url http://localhost:8888 2>"$LOG_DIR/tunnel.log" &
TUNNEL_PID=$!

# Wait for URL to appear in log
for i in $(seq 1 30); do
    TUNNEL_URL=$(grep -o 'https://[^ ]*trycloudflare.com' "$LOG_DIR/tunnel.log" 2>/dev/null | head -1)
    if [ -n "$TUNNEL_URL" ]; then
        echo "$(date): Tunnel URL: $TUNNEL_URL" >> "$LOG_DIR/tunnel_history.log"

        # Save to Turso DB
        echo "CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT DEFAULT (datetime('now')));
INSERT OR REPLACE INTO config (key, value) VALUES ('tunnel_url', '$TUNNEL_URL');" | turso db shell sports-bet 2>>"$LOG_DIR/tunnel.log"

        echo "Tunnel URL saved: $TUNNEL_URL"
        break
    fi
    sleep 1
done

# Keep running (wait for cloudflared to finish)
wait $TUNNEL_PID
