# OpenRouter Rate Limit Investigation Summary

**Date**: October 7, 2025
**Status**: INVESTIGATION IN PROGRESS

## Facts Established

### 1. API Calls Per Question
- **Research stage**: 3 OpenRouter calls
  - Synthesis: 1 call (GPT-5-mini)
  - Quality validation: 1 call
  - Gap detection: 1 call
  - AskNews: 0 calls (uses AskNews API, not OpenRouter)

- **Forecasting stage**: 5 OpenRouter calls
  - 5 predictions × 1 call each = 5 calls

- **TOTAL**: **8 OpenRouter API calls per question**

### 2. Actual Run Data (Oct 7, 06:52 UTC)
- Questions attempted: 66
- OpenRouter calls made: 3
- Circuit breaker opened: After 15 consecutive failures
- Error message: "Key limit exceeded (total limit)"

**Analysis**: The FIRST call already failed. Key was already at limit when run started.

### 3. Schedule Configuration
- **Tournament**: Every 4 hours + daily noon re-forecast
- **MiniBench**: Every hour (but ~1 question/day max)
- **Quarterly**: Every 2 days

Expected daily API calls (rough estimate):
- Tournament: 6 runs/day × ~10 questions/run × 8 calls = ~480 calls/day
- MiniBench: 1 question/day × 8 calls = ~8 calls/day
- Total: ~488 calls/day

### 4. OpenRouter Limits (from documentation)
- Free tier: 50 requests/day (if < $10 credits purchased)
- Paid tier (>$10): 1000 requests/day for `:free` models
- Regular paid models: Varies by model, governed by credits

**Key finding**: Error says "total limit" not "rate limit per minute"
- This suggests a DAILY or CREDIT limit, not per-minute rate limit

## Unknowns (Need to Investigate)

1. **What is the actual daily request limit for tournament keys?**
   - Metaculus provides keys - may have custom limits
   - Need to check with tournament organizers or forums

2. **When does the limit reset?**
   - Daily at midnight UTC?
   - Rolling 24-hour window?
   - Need to test or find documentation

3. **Are we using free models?**
   - Code shows: openai/gpt-5, openai/gpt-5-mini, openai/gpt-5-nano
   - These are NOT `:free` models
   - But error message unclear if it's request limit or credit limit

4. **Current delay between calls**:
   - `OPENROUTER_MIN_DELAY_SECONDS = 0.5`
   - Is this sufficient? OpenRouter docs don't specify minimum delay

## Proposed Solutions

### Immediate Actions (Can Do Now)

1. **Increase delay between API calls**
   ```python
   # In llm_client.py line 35
   OPENROUTER_MIN_DELAY_SECONDS = 2.0  # Was 0.5
   ```
   - Spreads 488 calls/day over more time
   - Reduces burst patterns that might trigger limits

2. **Add startup key status check**
   ```python
   # Query https://openrouter.ai/api/v1/auth/key
   # Check limit_remaining before starting run
   # Skip run if < $5 remaining
   ```

3. **Reduce redundant calls** (if any exist)
   - Review if quality_validation and gap_detection can be combined
   - Check if we can reduce predictions from 5 to 3

### Medium-term Actions (Need More Info)

4. **Adjust schedule based on actual limits**
   - If limit is 1000 requests/day: Current schedule OK
   - If limit is 200 requests/day: Need to reduce frequency
   - Tournament every 6 hours instead of 4?

5. **Implement smart queueing**
   - Don't process all questions if key is low on credits
   - Process top N priority questions only

6. **Monitor and alert**
   - Track daily usage
   - Alert when >80% of daily limit used
   - Pause runs until reset

### Long-term (Architecture Changes)

7. **Request pooling across runs**
   - Share API call budget across all scheduled runs
   - Coordinate MiniBench + Tournament to not exceed daily limit

8. **Model optimization**
   - Use GPT-5-nano for quality_validation (cheaper)
   - Use GPT-5-mini for everything except final predictions

## Next Steps

1. **Run check script to see actual key status**:
   ```bash
   python3 scripts/check_openrouter_limits.py
   ```
   - This will show: limit, limit_remaining, usage_daily
   - Will confirm if it's credit limit or request limit

2. **Search Metaculus Discord/Forums**
   - Look for "OpenRouter rate limit" or "API key limit"
   - Check if other bots hit same issue
   - Find official guidance on tournament key limits

3. **Test with reduced load**
   - Set MAX_PREDICTIONS_PER_REPORT=3 (instead of 5)
   - This reduces calls from 8 to 6 per question (-25%)

4. **Implement fixes based on findings**
   - Increase MIN_DELAY to 2.0
   - Add startup key status check
   - Adjust schedule if needed

## Files to Modify

1. `src/infrastructure/external_apis/llm_client.py` - Line 35 (MIN_DELAY)
2. `main.py` - Add startup key status check
3. `.github/workflows/run_bot_on_tournament.yaml` - Schedule (if needed)
4. `main.py` or env - MAX_PREDICTIONS_PER_REPORT (if needed)

## Timeline

- **Today**: Run check script, search forums, increase MIN_DELAY
- **Tomorrow**: Test with next scheduled run, monitor results
- **This week**: Implement smart queueing if limits are tight

---

**Key Insight**: We need ACTUAL data from the API key endpoint before making major changes. The error message alone isn't enough to determine the right fix.
