# CRITICAL ISSUES IDENTIFIED - 4 October 2025

## Summary
Bot is generating forecasts but **ZERO are being published** due to overly aggressive publication guards and conditional question handling issues.

## Root Causes

### 1. Publication Guard Blocking ALL Forecasts (CRITICAL)
**Location**: `main.py` lines 575-593 in `_wrap_publication_func`
**Problem**: The `_low_information()` and `_is_uniform()` checks are blocking legitimate tournament forecasts
**Evidence**:
- 26 forecasts blocked with message "Publication guard: blocked post_multiple_choice_question_prediction due to uniform/low-info MC distribution"
- Summary shows 26 "successful_forecasts" but 0 actual publications
- ALL blocked questions are in `blocked_publication_qids` list

**Impact**: **100% of generated forecasts are blocked from publication**

**Fix Options**:
1. **IMMEDIATE**: Set environment variable `DISABLE_PUBLICATION_GUARD=true` to bypass
2. **SHORT-TERM**: Relax thresholds in `_is_uniform()` and `_low_information()`
3. **LONG-TERM**: Remove or redesign guard to only block truly invalid forecasts

### 2. Conditional/Group Questions Not Filtered (HIGH)
**Problem**: API returns parent questions (e.g., 39618) that contain conditional sub-questions (e.g., 38995) that are already closed
**Evidence**:
- Question 39618 (mifepristone) attempts to post to 38995 (already closed)
- Question 39544 (federal interest rates) attempts to post to 38921 (already closed)
- 38 out of 93 questions fail with "already closed" or "not open yet" errors

**Impact**: **41% of questions cannot be forecasted**

**Fix**: Add filtering in patch to exclude questions with `type: "group"` or `conditional: true`

### 3. Summary Metrics Inverted (MEDIUM)
**Location**: `main.py` lines 4665-4776 (run_summary generation)
**Problem**: Code counts forecasts generated, not forecasts actually published
**Evidence**:
- Summary shows `"successful_forecasts": 26`
- But `"publish_attempts": 0` and `"published_success": 0`
- The 26 "successful" are actually the 26 blocked publications

**Impact**: **Misleading metrics prevent proper monitoring**

**Fix**: Count publications that reach Metaculus API, not forecast generations

### 4. OpenRouter Circuit Breaker Open (MEDIUM)
**Evidence**:
```json
"openrouter_circuit_breaker": {
  "is_open": true,
  "consecutive_failures": 15,
  "failure_threshold": 10
}
```

**Impact**: All forecasts falling back to emergency/mock modes
**Fix**: Reset circuit breaker and ensure valid API key

## Verification Steps

To verify these issues are real:

```bash
# Check publication guard blocks
grep "Publication guard: blocked" /tmp/forecast_test.log | wc -l
# Should show 26 blocks

# Check closed question errors
grep "already closed to forecasting" /tmp/forecast_test.log | wc -l
# Should show ~20+ errors

# Check actual publications
grep -i "successfully published\|forecast submitted" /tmp/forecast_test.log | wc -l
# Should show 0

# Verify questions retrieved correctly
grep "Retrieved.*questions" /tmp/forecast_test.log
# Should show "Retrieved 93 questions" with correct status=open filter
```

## Immediate Action Plan

### Priority 1: Enable Publications (30 minutes)
1. **Set environment variable**: `DISABLE_PUBLICATION_GUARD=true`
2. **Re-run bot**: `DRY_RUN=false SKIP_PREVIOUSLY_FORECASTED=true DISABLE_PUBLICATION_GUARD=true python3 main.py --mode tournament`
3. **Verify publications**: Check Metaculus website for actual forecasts

### Priority 2: Fix Question Filtering (1 hour)
1. **Update patch**: `src/infrastructure/patches/forecasting_tools_fix.py`
2. **Add filters**: Exclude questions where `type == "group"` or `conditional == true`
3. **Test**: Verify reduced error rate for closed questions

### Priority 3: Fix Summary Metrics (30 minutes)
1. **Track publication attempts**: Increment counter on actual API calls
2. **Track publication success**: Increment on 200/201 responses
3. **Separate blocked from failed**: Don't count guards as "successful"

### Priority 4: Reset Circuit Breaker (15 minutes)
1. **Verify API key**: Check `OPENROUTER_API_KEY` is valid
2. **Reset state**: Delete circuit breaker state file if exists
3. **Test with single question**: Verify API connectivity

## Expected Outcomes

After fixes:
- **Publications**: 50-70 forecasts published (down from 0)
- **Blocked**: 0-5 legitimately low-quality forecasts blocked
- **Closed errors**: 5-10 (down from 38) after filtering
- **Success rate**: 70-90% (up from 0%)

## Testing Protocol

Before considering this fixed:
1. Run bot with fixes applied
2. Check Metaculus tournament page: https://www.metaculus.com/tournament/fall-aib-2025/
3. Verify YOUR username appears on leaderboard
4. Verify specific question pages show your forecast (e.g., https://www.metaculus.com/questions/39409/)
5. Confirm `run_summary.json` shows `published_success > 20`

## Notes
- These issues explain why no forecasts visible on website
- These issues explain why user not appearing on tournament leaderboard
- All API keys are valid; filtering is working correctly
- The forecasting LOGIC is fine; only publication pipeline is broken
