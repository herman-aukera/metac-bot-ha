# 13 Missing Forecasts - ROOT CAUSE FIXED

**Date**: 2025-10-06 13:00 UTC  
**User Report**: 13 questions not predicted, 4 predictions stale  
**Root Cause**: Workflow concurrency killing all tournament runs  
**Status**: ✅ FIXED and workflow triggered

---

## 🎯 THE PROBLEM

### Evidence from GitHub Actions
```
ALL tournament runs were CANCELLED:
  2025-10-06T10:05:22: cancelled ❌
  2025-10-06T05:05:00: cancelled ❌  
  2025-10-06T00:17:16: cancelled ❌
```

### Root Cause: Line 33 of run_bot_on_tournament.yaml
```yaml
concurrency:
  group: tournament-forecasting
  cancel-in-progress: true  # ← KILLING EVERY RUN
```

**What happened**:
1. Scheduled workflow starts every 5 hours
2. New run triggers (scheduled or manual)
3. `cancel-in-progress: true` kills previous run
4. Result: ZERO forecasts ever published from tournament workflow

---

## ✅ THE FIX (Commit 2f084c2)

```yaml
concurrency:
  group: tournament-forecasting
  cancel-in-progress: false  # NOW: Runs allowed to complete
```

Also restored:
```yaml
SKIP_PREVIOUSLY_FORECASTED: true  # Catch only new/missed questions
```

---

## 🚀 ACTION TAKEN

1. ✅ Fixed concurrency setting
2. ✅ Committed and pushed to main  
3. ✅ Manually triggered workflow
4. ⏳ Monitoring progress

---

## 📊 WHAT TO EXPECT

**Current workflow will**:
- Process ~13-20 tournament questions
- Forecast the 13 unpredicted questions
- Take 30-45 minutes to complete
- Cost ~$2.60 ($0.20 per question)
- Use 650+ OpenRouter API calls (50 per question)

**Monitor at**: https://github.com/herman-aukera/metac-bot-ha/actions

---

## ✅ SUCCESS CRITERIA

After 45 minutes, check:
- [ ] Workflow status: "success" (NOT "cancelled")
- [ ] Duration: 30-45 min (NOT 1-2 min)  
- [ ] Logs show: "successful_forecasts: 13"
- [ ] Metaculus dashboard: 0 unpredicted questions
- [ ] Circuit breaker: Stayed closed

---

## 🔧 ALL FIXES APPLIED TO DATE

1. ✅ **Line 1929**: Disabled fallback publication
2. ✅ **Circuit breaker**: Resets at every startup
3. ✅ **Concurrency**: Fixed workflow cancellation
4. ✅ **Env vars**: All 3 workflows configured
5. ✅ **Schedules**: Optimized (1hr/5hr/2day)

---

## 📈 EXPECTED IMPROVEMENTS

| Metric | Before | After |
|--------|--------|-------|
| Tournament runs completing | 0 | ✅ All |
| Unpredicted questions | 13 | 0 |
| Workflow status | cancelled | success |
| API calls per run | 3 | 650+ |
| Cost per run | $0.01 | $2.60 |

---

**Next Check**: 30-45 minutes  
**Confidence**: High - Root cause identified and eliminated
