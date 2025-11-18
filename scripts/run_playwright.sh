#!/usr/bin/env bash
set -euo pipefail

# Find free port
PORT=$(python - <<'PY'
import socket
s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)

echo "Starting stubbed Streamlit on :$PORT"
STREAMLIT_SERVER_PORT="$PORT" nohup python -m streamlit run scripts/app_stubbed.py --server.headless true --server.port "$PORT" > .playwright.$PORT.log 2>&1 &
PID=$!
sleep 1
echo "PID=$PID" > .playwright.$PORT.pid

export PW_RUN=1
export PW_URL="http://localhost:$PORT"

echo "Running Playwright tests against $PW_URL"
python -m pytest -q tests_playwright || RC=$?

echo "Stopping $PID"
kill "$PID" || true
sleep 1
exit ${RC:-0}

