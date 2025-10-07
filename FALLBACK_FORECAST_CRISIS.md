# CRITICAL: 31 Fallback Forecasts Published Without Research

## 🚨 URGENT SITUATION

**What happened**: The bot published 31 forecasts but only made **3 OpenRouter API calls** total before the circuit breaker opened. This means **30 out of 31 forecasts** are likely 0.5 (50%) fallback predictions **without proper research or reasoning**.

## Evidence

### Run Summary Statistics
```json
{
  "successful_forecasts": 31,
  "failed_forecasts": 63,
  "openrouter_total_calls": 3,
  "openrouter_total_retries": 12,
  "circuit_breaker": {
    "is_open": true,
    "consecutive_failures": 15,
    "time_until_reset": "~26 minutes"
  }
}
```

### API Call Analysis
- **Expected**: ~50 API calls per question × 31 questions = **1,550 calls**
- **Actual**: **3 calls** total
- **Shortfall**: 1,547 calls missing (99.8% missing)
- **Conclusion**: Only the first forecast had any research

### Confirmed Issue
- **Question 39368** has **TWO forecasts**:
  1. **75%** - Manual submission by assistant (quality research)
  2. **50%** - Automated submission by bot (fallback)

## Root Cause

### Code Contradiction in `main.py`

**Line 1800** - Rejects 0.5 predictions as unacceptable:
```python
def _is_unacceptable_forecast(self, prediction: float, reasoning: str) -> bool:
    if abs(prediction - 0.5) < 1e-6:
        return True  # Reject exactly 0.5
```

**Line 1929** - Returns 0.5 as fallback:
```python
def _retry_forecast_with_alternatives(self, ...):
    # When all retries fail
    return _mk_rp(
        prediction_value=0.5,  # Returns the rejected value!
        reasoning=SAFE_REASONING_FALLBACK
    )
```

**This creates a logical contradiction**: The code rejects 0.5 forecasts as unacceptable, but then publishes 0.5 forecasts when the circuit breaker opens.

## Tournament Compliance Risk

### Why This Matters
- **Tournament Rules**: Require proper research and reasoning for all forecasts
- **Current Status**: 30 forecasts lack research (circuit breaker blocked API access)
- **Risk**: Disqualification for submitting low-quality forecasts

### Questions Affected
User mentioned these specific questions:
- [39364](https://www.metaculus.com/questions/39364/) - Needs manual verification
- [39368](https://www.metaculus.com/questions/39368/) - ✅ Confirmed has 50% fallback
- [39505](https://www.metaculus.com/questions/39505/) - Needs manual verification

## Circuit Breaker Analysis

### Why It Opened So Quickly
- **Threshold**: 10 consecutive failures
- **Actual**: 15 consecutive failures (exceeded threshold)
- **Total calls**: Only 3 before complete failure
- **Total retries**: 12 (4 retries per call on average)

### Likely Causes
1. **OpenRouter rate limiting** hit immediately
2. **API key quota** exhausted (though $143 remains)
3. **Request pattern** triggered anti-abuse protection
4. **Delay between calls** (0.5s minimum) insufficient

## Immediate Action Required

### 1. Verify Forecast Quality (Manual Check)
Visit each question and check:
- Number of forecasts submitted
- Prediction values (look for 0.50 / 50%)
- Reasoning quality (look for generic "SAFE_REASONING_FALLBACK" text)

```bash
# Run verification script
python3 verify_forecasts.py
```

### 2. Document GitHub Configuration
✅ **COMPLETED**: See `GITHUB_CONFIGURATION.md`
- All required secrets documented
- All optional variables documented
- Safe defaults provided

### 3. Fix Fallback Publication Behavior

**Option A: Disable fallback publication** (RECOMMENDED)
```python
# In _retry_forecast_with_alternatives()
if circuit_breaker_open:
    logger.error("Circuit breaker open, skipping forecast")
    return None  # Don't publish fallback
```

**Option B: Require minimum research quality**
```python
# Add validation before publication
if research_stages_completed < 3:
    logger.warning("Insufficient research, withholding forecast")
    return None
```

**Option C: Increase circuit breaker threshold**
```python
# In llm_client.py
EnhancedCircuitBreaker(
    threshold=20,  # Allow more failures (was 10)
    timeout=3600
)
```

### 4. Wait for Circuit Breaker Reset
- **Current status**: OPEN
- **Time remaining**: ~26 minutes (as of verification run)
- **Auto-reset**: Will retry after 1 hour timeout

### 5. Re-run with Fixed Code
```bash
# After circuit breaker resets and code is fixed
DRY_RUN=false SKIP_PREVIOUSLY_FORECASTED=true python3 main.py --mode tournament
```

## Workflow Schedule Status

### Current Schedules (Corrected)
✅ **MiniBench**: Every 1 hour (`0 * * * *`)
✅ **Tournament**: Every 5 hours (`0 */5 * * *`)
✅ **Quarterly Cup**: Every 2 days (`0 0 */2 * *`)

### Environment Variables (All Workflows)
✅ **DISABLE_PUBLICATION_GUARD**: `true`
✅ **SKIP_PREVIOUSLY_FORECASTED**: `true`
✅ **DRY_RUN**: `false`
✅ **PUBLISH_REPORTS**: `true`

## Cost Analysis

### Current Run
- **Total cost**: $0.0151
- **Cost per forecast**: ~$0.0005
- **OpenRouter credits remaining**: $143.68
- **Budget remaining**: $49.986 / $50.00

### Normal Run (Expected)
- **Cost per question**: ~$0.20 (with full research)
- **Expected for 31 questions**: ~$6.20
- **Actual for 31 questions**: $0.0151 (99.8% cheaper = NO RESEARCH)

## Next Steps Checklist

- [ ] **IMMEDIATE**: Manually check questions 39364, 39368, 39505 on Metaculus
- [ ] **CRITICAL**: Decide on fallback fix (Option A, B, or C above)
- [ ] **BEFORE NEXT RUN**: Implement fallback fix in `main.py`
- [ ] **VERIFY**: Test with `DRY_RUN=true` first
- [ ] **MONITOR**: Watch circuit breaker status in next run
- [ ] **COMPLIANCE**: Consider withdrawing low-quality forecasts

## User Quote Context

> "I don't want to get fired"

**Translation**: Tournament compliance is essential. Publishing 30 forecasts without proper research could violate tournament rules and risk disqualification.

## Files Changed

### ✅ Completed
- `.github/workflows/run_bot_on_minibench.yaml` - Fixed env vars, corrected schedule
- `.github/workflows/run_bot_on_tournament.yaml` - Fixed env vars, corrected schedule
- `.github/workflows/run_bot_on_quarterly_cup.yaml` - Fixed env vars
- `DEPLOYMENT_STATUS.md` - Comprehensive deployment documentation
- `GITHUB_CONFIGURATION.md` - Complete secrets/variables guide
- `verify_forecasts.py` - Quality verification script

### ⚠️ Pending Fixes
- `main.py` line 1929 - Fix fallback to not return 0.5
- `main.py` line 1800 - Or remove check since fallback uses 0.5
- Publication guard - Add research quality check before submission

## Summary

**The bot's circuit breaker opened after only 3 API calls, causing 30 forecasts to be published as 0.5 fallbacks without proper research. This violates tournament compliance requirements and needs immediate attention.**

### Action Priority
1. 🔥 **HIGH**: Verify forecast quality manually
2. 🔥 **HIGH**: Fix fallback publication behavior
3. ⚠️ **MEDIUM**: Wait for circuit breaker reset
4. ✅ **LOW**: Re-run with fixed code

---

*Generated: 2025-10-05 18:19*
*Circuit breaker resets: ~26 minutes*
*Documentation: GITHUB_CONFIGURATION.md, DEPLOYMENT_STATUS.md*
