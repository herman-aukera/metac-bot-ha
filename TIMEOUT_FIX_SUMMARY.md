# ACTUAL ROOT CAUSE: Workflow Timeouts (Not Concurrency!)

## 🎯 THE REAL PROBLEM DISCOVERED

**What we thought**: `cancel-in-progress: true` was killing workflows  
**What it actually was**: **Workflows timing out at exactly their limit**

### Evidence
```
Tournament runs: ALL killed at 1820s (exactly 30 minutes)
MiniBench runs: ALL killed at 1520s (exactly 25 minutes)

NOT random cancellations - EXACT timeout matches!
```

---

## ✅ THE FIX (Commit: e9e80f6)

### 1. Increased Timeouts
- **Tournament**: 30min → 90min (needs 40-50min for 13+ questions)
- **MiniBench**: 25min → 60min (needs 30-40min for 10+ questions)

### 2. Enabled Re-forecasting  
- `SKIP_PREVIOUSLY_FORECASTED: false` (update existing forecasts)
- Scheduled once daily (12:00 UTC) to manage costs

### 3. Why Timeouts Were Too Short
- Each question: 3-4 minutes with proper research
- 13 questions: ~40-50 minutes minimum
- Rate limiting + retries add time
- Old timeout: 30min = impossible ❌
- New timeout: 90min = comfortable ✅

---

## 📊 CONFIGURATION NOW

### Tournament (Daily at noon UTC)
```yaml
schedule: '0 12 * * *'
timeout-minutes: 90
SKIP_PREVIOUSLY_FORECASTED: false  # Allow updates
```

### MiniBench (Hourly)
```yaml
schedule: '0 * * * *'
timeout-minutes: 60
SKIP_PREVIOUSLY_FORECASTED: true  # Only new
```

---

## 💰 Expected Cost (Reasonable)

- **Tournament**: $2.60-$4.00/day (13-20 questions)
- **MiniBench**: $5-$15/day (only when new questions)
- **Total**: ~$228-$570/month (within $1,650 budget)

---

## ✅ What to Check Tomorrow

**Tournament Run (12:00 UTC)**:
- [ ] Status: "success" (NOT "cancelled")
- [ ] Duration: 40-90 minutes
- [ ] All 13 questions forecasted
- [ ] Proper research (650+ API calls)

---

**Status**: Fixed and deployed  
**Next Run**: Tomorrow 12:00 UTC  
**Confidence**: Very High (timeout evidence is definitive)
