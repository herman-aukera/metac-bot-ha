#!/usr/bin/env bash
set -Eeuo pipefail

echo "[local-ci] Starting MiniBench local CI simulation..."

# Safe local overrides to avoid side effects (can be overridden by env)
export TOURNAMENT_MODE=${TOURNAMENT_MODE:-true}
export DRY_RUN=${DRY_RUN:-false}
export PUBLISH_REPORTS=${PUBLISH_REPORTS:-false}
# Avoid setting PYTHONPATH=src to prevent module name collisions with forecasting_tools

# Offline by default to avoid accidental spend; set LOCAL_CI_NETWORK=1 to enable networked run
: "${LOCAL_CI_NETWORK:=0}"
if [[ "$LOCAL_CI_NETWORK" != "1" ]]; then
  export OPENROUTER_API_KEY="dummy_local"
  export ENABLE_PROXY_CREDITS=false
  export ENABLE_ASKNEWS_RESEARCH=false
  export DRY_RUN=true
  echo "[local-ci] Network disabled (LOCAL_CI_NETWORK=0): OpenRouter/Proxy/AskNews disabled"
else
  echo "[local-ci] Network enabled (LOCAL_CI_NETWORK=1)"
  # Do not source .env directly to avoid parsing issues; python-dotenv will load it where needed
  if [[ ! -f .env ]]; then
    echo "[local-ci] WARNING: .env not found; relying on current environment"
  else
    # Export only required keys from .env into current shell without printing values
    ALLOWLIST=(
      OPENROUTER_API_KEY OPENROUTER_BASE_URL OPENROUTER_HTTP_REFERER OPENROUTER_APP_TITLE
      METACULUS_TOKEN ASKNEWS_CLIENT_ID ASKNEWS_SECRET
      DEFAULT_MODEL MINI_MODEL NANO_MODEL
      ENABLE_PROXY_CREDITS ENABLE_ASKNEWS_RESEARCH
      PRIMARY_RESEARCH_MODEL PRIMARY_FORECAST_MODEL SIMPLE_TASK_MODEL
      AIB_TOURNAMENT_SLUG TOURNAMENT_SLUG AIB_TOURNAMENT_ID
    )
    if python3 - <<'PY'
import os, shlex, sys
try:
  from dotenv import dotenv_values
except Exception:
  sys.exit(3)
vals = dotenv_values('.env')
allowlist = [
  'OPENROUTER_API_KEY','OPENROUTER_BASE_URL','OPENROUTER_HTTP_REFERER','OPENROUTER_APP_TITLE',
  'METACULUS_TOKEN','ASKNEWS_CLIENT_ID','ASKNEWS_SECRET',
  'DEFAULT_MODEL','MINI_MODEL','NANO_MODEL',
  'ENABLE_PROXY_CREDITS','ENABLE_ASKNEWS_RESEARCH',
  'PRIMARY_RESEARCH_MODEL','PRIMARY_FORECAST_MODEL','SIMPLE_TASK_MODEL',
  'AIB_TOURNAMENT_SLUG','TOURNAMENT_SLUG','AIB_TOURNAMENT_ID'
]
with open('.local_env_export.sh','w') as f:
    f.write('set -a\n')
    for k in allowlist:
        v = vals.get(k)
        if v is not None:
            f.write(f"export {k}={shlex.quote(v)}\n")
    f.write('set +a\n')
PY
    then
      # shellcheck disable=SC1090
      . ./.local_env_export.sh
      rm -f .local_env_export.sh
    else
      echo "[local-ci] python-dotenv not available; using POSIX fallback to load .env keys"
      {
        echo 'set -a'
        while IFS='=' read -r key val; do
          # skip comments/blank lines
          [[ -z "$key" || "$key" =~ ^# ]] && continue
          for allow in "${ALLOWLIST[@]}"; do
            if [[ "$key" == "$allow" ]]; then
              # Trim surrounding quotes if any
              val="${val%\r}"
              echo "export $key=$val"
              break
            fi
          done
        done < .env
        echo 'set +a'
      } > .local_env_export.sh
      # shellcheck disable=SC1090
      . ./.local_env_export.sh
      rm -f .local_env_export.sh
    fi
  fi
fi

if [[ "${FAST:-0}" != "1" ]]; then
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
fi

echo "[local-ci] Preflight: loading .env via python-dotenv and checking requirements"
python3 - <<'PY'
import os, sys
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env', override=True)
DRY_RUN = os.getenv('DRY_RUN', 'false').lower() in ('1','true','yes')
if not DRY_RUN and not os.getenv('METACULUS_TOKEN'):
  print('[local-ci] ERROR: METACULUS_TOKEN missing (.env or env) and DRY_RUN is false')
  sys.exit(2)
if not os.getenv('OPENROUTER_API_KEY'):
  print('[local-ci] WARN: OPENROUTER_API_KEY not set; will rely on fallbacks')
target = (
  os.getenv('AIB_TOURNAMENT_SLUG')
  or os.getenv('TOURNAMENT_SLUG')
  or os.getenv('AIB_TOURNAMENT_ID')
  or 'minibench'
)
print('[local-ci] Tournament target:', '(hidden)')
with open('.local_target.tmp','w') as f:
  f.write(target)
PY
TARGET=$(cat .local_target.tmp); rm -f .local_target.tmp

# Optional: quick connectivity check (HTTP codes only)
if [[ "$LOCAL_CI_NETWORK" == "1" ]]; then
  echo "[local-ci] Connectivity check (OpenRouter/Metaculus)"
  python3 - <<'PY'
import os, subprocess
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path='.env', override=True)
except Exception:
    pass
subprocess.run(['bash', 'scripts/check_connectivity.sh'], check=False)
PY
fi

echo "[local-ci] Running bot (dry-run: $DRY_RUN, publish: $PUBLISH_REPORTS)"
set +e

# Decide whether TARGET is an ID (digits) or a slug, then export accordingly
if [[ "$TARGET" =~ ^[0-9]+$ ]]; then
  echo "[local-ci] Using numeric tournament ID: (hidden)"
  if AIB_TOURNAMENT_ID="$TARGET" \
     TOURNAMENT_MODE=true \
     PUBLISH_REPORTS=true \
     DRY_RUN="$DRY_RUN" \
     SKIP_PREVIOUSLY_FORECASTED=false \
     python3 main.py --mode tournament; then
    echo "[local-ci] Bot run completed"
  else
    echo "[local-ci] Bot run failed"
  fi
else
  echo "[local-ci] Using tournament slug: (hidden)"
  if AIB_TOURNAMENT_SLUG="$TARGET" \
     TOURNAMENT_MODE=true \
     PUBLISH_REPORTS=true \
     DRY_RUN="$DRY_RUN" \
     SKIP_PREVIOUSLY_FORECASTED=false \
     python3 main.py --mode tournament; then
    echo "[local-ci] Bot run completed"
  else
    echo "[local-ci] Bot run failed"
  fi
fi
set -e

if [[ -f run_summary.json ]]; then
  echo "[local-ci] run_summary.json:"
  cat run_summary.json
else
  echo "[local-ci] No run_summary.json produced"
fi

echo "[local-ci] Done"
