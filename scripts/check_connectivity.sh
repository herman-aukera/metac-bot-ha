#!/usr/bin/env bash
set -euo pipefail

# Simple connectivity checks for OpenRouter and Metaculus using curl
# Requires: curl, jq (optional)

HTTP() {
  url="$1"; shift
  code=$(curl -sS -o /tmp/conn_body.$$ -w "%{http_code}" "$@" "$url" || true)
  echo "$code"
}

info() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
err() { echo "[ERR ] $*"; }

# OpenRouter check
OPENROUTER_BASE_URL=${OPENROUTER_BASE_URL:-"https://openrouter.ai/api/v1"}
info "Checking OpenRouter models endpoint: $OPENROUTER_BASE_URL/models"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  warn "OPENROUTER_API_KEY not set; skipping authorized check"
else
  code=$(HTTP "$OPENROUTER_BASE_URL/models" \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    -H "Accept: application/json" \
    ${OPENROUTER_HTTP_REFERER:+-H "HTTP-Referer: $OPENROUTER_HTTP_REFERER"} \
    ${OPENROUTER_APP_TITLE:+-H "X-Title: $OPENROUTER_APP_TITLE"})
  echo "OpenRouter /models HTTP $code"
  if command -v jq >/dev/null 2>&1 && [[ "$code" == "200" ]]; then
    count=$(jq -r '.data | length' </tmp/conn_body.$$ 2>/dev/null || echo "?")
    echo "Models available: $count"
  fi
fi

# Metaculus check (requires API Token)
if [[ -z "${METACULUS_TOKEN:-}" ]]; then
  warn "METACULUS_TOKEN not set; skipping Metaculus check"
else
  METACULUS_URL=${METACULUS_URL:-"https://www.metaculus.com/api2/questions/?limit=1"}
  info "Checking Metaculus API: $METACULUS_URL"
  code=$(HTTP "$METACULUS_URL" \
    -H "Authorization: Token $METACULUS_TOKEN" \
    -H "Accept: application/json")
  echo "Metaculus HTTP $code"
fi

rm -f /tmp/conn_body.$$ || true
info "Connectivity checks completed"
