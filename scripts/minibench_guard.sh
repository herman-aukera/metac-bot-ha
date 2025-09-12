#!/usr/bin/env bash
# MiniBench guard: prevent unnecessary or costly runs
# - Skips weekends by default (Mon–Fri only)
# - Enforces a run window in a configurable timezone
# - Limits to N runs per day using a stamp file

set -euo pipefail

# Configurable envs
MINIBENCH_TZ=${MINIBENCH_TZ:-UTC}
MINIBENCH_WINDOW_START_HH=${MINIBENCH_WINDOW_START_HH:-15}   # 15:00
MINIBENCH_WINDOW_END_HH=${MINIBENCH_WINDOW_END_HH:-23}       # 23:59
MINIBENCH_MAX_RUNS_PER_DAY=${MINIBENCH_MAX_RUNS_PER_DAY:-1}
MINIBENCH_ALLOW_WEEKENDS=${MINIBENCH_ALLOW_WEEKENDS:-false}

STATE_DIR=${MINIBENCH_STATE_DIR:-"$(pwd)/data"}
STAMP_FILE="$STATE_DIR/minibench_run_$(TZ="$MINIBENCH_TZ" date +%Y-%m-%d).stamp"

mkdir -p "$STATE_DIR"

# Compute local time in timezone
current_hour=$(TZ="$MINIBENCH_TZ" date +%H)
current_dow=$(TZ="$MINIBENCH_TZ" date +%u)  # 1=Mon .. 7=Sun

# Weekend guard
if [[ "$MINIBENCH_ALLOW_WEEKENDS" != "true" ]]; then
  if [[ "$current_dow" -eq 6 || "$current_dow" -eq 7 ]]; then
    echo "[minibench-guard] Weekend detected in $MINIBENCH_TZ (dow=$current_dow); skipping run." >&2
    exit 20
  fi
fi

# Time window guard
if (( 10#$current_hour < 10#$MINIBENCH_WINDOW_START_HH || 10#$current_hour > 10#$MINIBENCH_WINDOW_END_HH )); then
  echo "[minibench-guard] Outside window ${MINIBENCH_WINDOW_START_HH}-${MINIBENCH_WINDOW_END_HH} in $MINIBENCH_TZ (hour=$current_hour); skipping run." >&2
  exit 21
fi

# Per-day run limiter
if [[ -f "$STAMP_FILE" ]]; then
  runs=$(cat "$STAMP_FILE" 2>/dev/null || echo 0)
  runs=${runs:-0}
  if (( runs >= MINIBENCH_MAX_RUNS_PER_DAY )); then
    echo "[minibench-guard] Daily run limit reached ($runs >= $MINIBENCH_MAX_RUNS_PER_DAY); skipping run." >&2
    exit 22
  fi
fi

# Record a run (optimistically)
if [[ -f "$STAMP_FILE" ]]; then
  runs=$(cat "$STAMP_FILE" 2>/dev/null || echo 0)
  runs=${runs:-0}
  echo $((runs+1)) > "$STAMP_FILE"
else
  echo 1 > "$STAMP_FILE"
fi

echo "[minibench-guard] Run allowed (tz=$MINIBENCH_TZ hour=$current_hour, window ${MINIBENCH_WINDOW_START_HH}-${MINIBENCH_WINDOW_END_HH}); count=$(cat "$STAMP_FILE")." >&2
exit 0
