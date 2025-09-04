#!/usr/bin/env bash
set -Eeuo pipefail

echo "[local-ci] Starting MiniBench local CI simulation..."

# Load .env without echoing values
if [[ -f ".env" ]]; then
  echo "[local-ci] Loading .env"
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
else
  echo "[local-ci] No .env found; continuing with current env"
fi

# Safe local overrides to avoid side effects
export TOURNAMENT_MODE=true
export DRY_RUN=true
export PUBLISH_REPORTS=false
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

python3 --version || true
python3 -m pip install --upgrade pip >/dev/null 2>&1 || true

echo "[local-ci] Installing minimal dependencies (pip fast path)"
python3 -m pip install -q python-dotenv pydantic requests openai anthropic httpx aiofiles pyyaml typer pytest || true
python3 -m pip install -q forecasting-tools || echo "[local-ci] forecasting-tools not available via pip"
python3 -m pip install -q -e . || echo "[local-ci] Local editable install failed (non-fatal)"

echo "[local-ci] Verifying critical imports"
python3 - <<'PY'
import sys
missing = []
for pkg in ['dotenv','pydantic','requests','openai']:
    try:
        __import__(pkg)
        print(f" - {pkg}: ok")
    except Exception:
        missing.append(pkg)
if missing:
    print(f"[local-ci] Missing critical: {missing}")
    sys.exit(1)
PY

echo "[local-ci] Preflight: checking secrets and target (values not printed)"
if [[ -z "${METACULUS_TOKEN:-}" ]]; then
  echo "[local-ci] ERROR: METACULUS_TOKEN missing. Add it to .env or your shell env."
  exit 1
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "[local-ci] WARN: OPENROUTER_API_KEY not set; will rely on proxy/fallbacks"
fi

# Resolve target from .env similar to the workflow (prefer slug then id)
TARGET="${AIB_TOURNAMENT_SLUG:-}"
if [[ -z "$TARGET" ]]; then
  TARGET="${TOURNAMENT_SLUG:-}"
fi
if [[ -z "$TARGET" ]]; then
  TARGET="${AIB_TOURNAMENT_ID:-32813}"
fi
echo "[local-ci] Tournament target resolved"

echo "[local-ci] Running bot (dry-run: $DRY_RUN, publish: $PUBLISH_REPORTS)"
set +e
if AIB_TOURNAMENT_SLUG="$TARGET" \
   AIB_TOURNAMENT_ID="$TARGET" \
   TOURNAMENT_MODE=true \
   PUBLISH_REPORTS=false \
   DRY_RUN=true \
   python3 main.py --mode tournament; then
  echo "[local-ci] Bot run completed"
else
  echo "[local-ci] Bot run failed"
fi
set -e

if [[ -f run_summary.json ]]; then
  echo "[local-ci] run_summary.json:"
  cat run_summary.json
else
  echo "[local-ci] No run_summary.json produced"
fi

echo "[local-ci] Done"
