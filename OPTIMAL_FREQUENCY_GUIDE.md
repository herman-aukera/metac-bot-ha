# Optimal Frequency Configuration for Donated API Key
_Based on analysis of the 85.5% success run that still exhausted quotas within ~20 minutes._

## Root Cause Analysis

- ✅ Not a daily spend issue—the run processed 71/83 questions successfully (85.5%).
- ❌ Hit per-minute request limits on both OpenRouter and AskNews simultaneously.
- 💡 Mitigation: slow the seasonal sweep to once per day and let MiniBench keep the faster 6-hour cadence.

## Tournament Schedule (coordinated with MiniBench)

### Main Tournament Bot (Seasonal)

- Frequency: once every 24 hours (daily sweep).
- Run window: pick a stable UTC hour, e.g. 02:00 UTC, to dodge peak traffic.
- Concurrent questions: 2 (enough to clear backlog without bursty spikes).
- Delay between questions: 60 seconds to respect per-minute limits.
- Daily budget: 10 questions, reserving 3 for MiniBench usage.

### MiniBench Cadence

- Frequency: every 6 hours (00:00, 06:00, 12:00, 18:00 UTC).
- Offset the seasonal sweep by ≥4 hours when possible.
- Reserved budget: 3 questions/day to keep the fast leaderboard updates.
- Fallbacks (DuckDuckGo + Wikipedia) stay enabled so runs succeed even if premium APIs throttle.

## Timing Strategy

```bash
export SCHEDULING_FREQUENCY_HOURS=24
export MAX_CONCURRENT_QUESTIONS=2
export API_DELAY_BETWEEN_QUESTIONS=60
export DAILY_QUESTION_BUDGET=10
export MINIBENCH_RESERVED_BUDGET=3
# Optional: helper for MiniBench automation
export MINIBENCH_SCHEDULING_HOURS=6
```

## MiniBench Timing Tips

- If the seasonal run is at 02:00 UTC, schedule MiniBench at 06:00/12:00/18:00/00:00 UTC.
- Maintain at least a 3–4 hour gap before/after the seasonal sweep to avoid overlapping peak API usage.

## Rate Limiting Protection

- Enforce ≤10 API calls per minute.
- Maintain 60-second pauses between questions and 3-minute spacing between batches.
- Continue logging quota hits so we can raise cadence again if provider limits improve.

## Why This Cadence Works

- ✅ Daily seasonal sweep dramatically lowers the chance of dual provider throttle events.
- ✅ MiniBench keeps competitive responsiveness with its 6-hour loop.
- ✅ Plenty of headroom to tighten cadence later if quotas expand.
- ✅ Aligns with conservative defaults baked into `tournament_config.py` and `.env.*` templates.
- ❌ Requires manual review if tournament volume spikes suddenly (may need temporary 12/8-hour runs).
- ❌ Both bots still share the same donated keys—monitor dashboards for sustained 403/429 streaks.
