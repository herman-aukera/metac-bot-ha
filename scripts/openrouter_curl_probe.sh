#!/usr/bin/env bash
# Simple OpenRouter connectivity & rate-limit header probe
# Usage: bash scripts/openrouter_curl_probe.sh "test prompt"
set -euo pipefail
PROMPT=${1:-"ping"}
MODEL=${OPENROUTER_PROBE_MODEL:-"openrouter/auto"}
BASE_URL=${OPENROUTER_BASE_URL:-"https://openrouter.ai/api/v1"}
ENDPOINT="$BASE_URL/chat/completions"
START_TS_MS=$(python3 - <<'PY'
import time; print(int(time.time()*1000))
PY
)
HTTP_CODE=$(curl -s -o /tmp/or_probe_body.json -w '%{http_code}' \
  -H "Authorization: Bearer ${OPENROUTER_API_KEY:-}" \
  -H "Content-Type: application/json" \
  -H "HTTP-Referer: ${OPENROUTER_HTTP_REFERER:-https://github.com/metac-bot-ha}" \
  -H "X-Title: ${OPENROUTER_APP_TITLE:-Metaculus Forecasting Bot HA}" \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"'"$PROMPT"'"}],"temperature":0.0,"max_tokens":8}' \
  "$ENDPOINT" || true)
END_TS_MS=$(python3 - <<'PY'
import time; print(int(time.time()*1000))
PY
)
LATENCY_MS=$((END_TS_MS-START_TS_MS))
echo "status=$HTTP_CODE latency_ms=$LATENCY_MS" >&2
# Extract key headers (case-insensitive)
for H in x-ratelimit-remaining x-ratelimit-reset retry-after; do
  VAL=$(curl -s -I -H "Authorization: Bearer ${OPENROUTER_API_KEY:-}" "$ENDPOINT" | grep -i "^$H:" | head -n1 | cut -d':' -f2- | xargs || true)
  [ -n "$VAL" ] && echo "header:$H=$VAL" >&2
done
if [ -f /tmp/or_probe_body.json ]; then
  echo "body:" >&2
  head -c 400 /tmp/or_probe_body.json >&2 || true
fi
