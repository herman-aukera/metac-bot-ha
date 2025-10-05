# Manual Workflow Trigger Guide - Re-forecast Fallback Questions

## 🎯 OBJECTIVE
Re-run forecasting on the 30 questions that received 0.5 fallback predictions without proper research.

## ✅ FIXES IMPLEMENTED

1. **Line 1929 Fix**: Disabled fallback publication
   - OLD: Returns 0.5 with generic reasoning
   - NEW: Returns None to skip publication
   - Result: No more low-quality forecasts published

2. **Circuit Breaker Reset**: Added at startup (line 4120)
   - Resets on every run
   - Prevents cascading failures from previous runs
   - Ensures fresh API attempts

3. **GitHub Workflows**: Already configured correctly
   - Environment variables set
   - Schedules optimized (1hr MiniBench, 5hr Tournament)
   - Ready for production use

## 🚀 HOW TO MANUALLY TRIGGER RE-FORECAST

### Option 1: GitHub Actions UI (Recommended)

1. **Go to Actions tab**:
   ```
   https://github.com/herman-aukera/metac-bot-ha/actions
   ```

2. **Select workflow**:
   - Click "Run Bot on Tournament" (left sidebar)

3. **Click "Run workflow"** (right side)

4. **Configure run**:
   ```
   Branch: main
   tournament_slug: [leave empty - uses AIB_TOURNAMENT_ID=32813]
   scheduling_frequency_hours: 5
   tournament_mode: normal
   ```

5. **Click green "Run workflow" button**

### Option 2: GitHub CLI (If installed)

```bash
# Trigger tournament workflow manually
gh workflow run run_bot_on_tournament.yaml

# Check run status
gh run list --limit 5

# Watch logs in real-time
gh run watch
```

### Option 3: API Trigger (Advanced)

```bash
# Using curl with GitHub token
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://api.github.com/repos/herman-aukera/metac-bot-ha/actions/workflows/run_bot_on_tournament.yaml/dispatches \
  -d '{"ref":"main"}'
```

## ⚙️ ENVIRONMENT CONFIGURATION

### For Re-forecasting ALL Questions (Including Fallbacks)

The workflow is already configured with:
```yaml
SKIP_PREVIOUSLY_FORECASTED: true  # Normal operation
```

But to **re-forecast the 30 fallback questions**, you need to temporarily override this:

1. **Edit workflow file** (one-time change):
   ```bash
   # Edit .github/workflows/run_bot_on_tournament.yaml
   # Change line 161:
   SKIP_PREVIOUSLY_FORECASTED: false  # Allow re-forecasting
   ```

2. **Commit and push**:
   ```bash
   git add .github/workflows/run_bot_on_tournament.yaml
   git commit -m "temp: allow re-forecasting for fallback updates"
   git push
   ```

3. **Trigger workflow** (see Option 1 above)

4. **Revert after successful run**:
   ```bash
   # Change back to:
   SKIP_PREVIOUSLY_FORECASTED: true  # Prevent duplicates
   git commit -am "revert: restore skip_previously_forecasted"
   git push
   ```

### Alternative: Use workflow inputs (no code changes)

The workflow supports input parameters. Add this to the workflow YAML if not present:

```yaml
on:
  workflow_dispatch:
    inputs:
      skip_previously_forecasted:
        description: 'Skip previously forecasted questions'
        required: false
        default: 'true'
        type: choice
        options:
          - 'true'
          - 'false'
```

Then use `false` when manually triggering to allow re-forecasting.

## 📊 MONITORING THE RUN

### During Execution

1. **Watch GitHub Actions tab**:
   - Live logs appear in real-time
   - Check for "✅ OpenRouter circuit breaker reset at startup"
   - Monitor API call counts

2. **Look for key log messages**:
   ```
   ✅ OpenRouter circuit breaker reset at startup
   ✅ OpenRouter configuration validated successfully
   Processing tournament 32813...
   Forecast retry alternatives exhausted - skipping publication  # Should NOT appear
   ```

3. **API Usage Expectations**:
   - Expected: ~50 OpenRouter calls per question
   - For 31 questions: ~1,550 calls total
   - Cost: ~$0.20 per question = $6.20 total

### After Completion

1. **Check run_summary.json artifact**:
   ```json
   {
     "successful_forecasts": 31,
     "openrouter_total_calls": 1550,  // Should be ~1500+
     "circuit_breaker": {
       "is_open": false  // Should be closed
     }
   }
   ```

2. **Verify on Metaculus**:
   - Question 39368: https://www.metaculus.com/questions/39368/
   - Should have NEW forecast with proper reasoning
   - Check timestamp to confirm it's recent

3. **Check logs artifact**:
   - Download logs from Actions → Artifacts
   - Search for "skipping publication" (should be absent)
   - Verify reasoning quality in logs

## ✅ SUCCESS CRITERIA

### Run is successful when:

- [ ] **API Calls**: 1,000+ OpenRouter calls (not just 3)
- [ ] **Circuit Breaker**: Remains closed (not opened)
- [ ] **Cost**: $5-10 range (not $0.01)
- [ ] **Forecasts**: 31 successful with proper research
- [ ] **Reasoning**: Check 3-5 questions on Metaculus for quality comments
- [ ] **No Fallbacks**: Zero "skipping publication" messages in logs

### Warning Signs:

- ⚠️ Circuit breaker opens again (only 3-10 API calls)
- ⚠️ Cost remains < $0.10 (indicates no real research)
- ⚠️ "Forecast retry alternatives exhausted" appears
- ⚠️ 403 rate limit errors within first few questions

## 🔧 TROUBLESHOOTING

### Problem: Circuit breaker opens immediately again

**Cause**: OpenRouter rate limiting still active

**Solution**:
1. Wait 1 hour between runs
2. Check OpenRouter dashboard for rate limit status
3. Consider adding longer delays between API calls:
   ```bash
   export OPENROUTER_MIN_DELAY_SECONDS=1.0  # Increase from 0.5s
   ```

### Problem: Still getting 0.5 forecasts

**Cause**: Fix not applied or code cache

**Solution**:
1. Verify commit `ede1bbf` is on main branch
2. Check workflow is pulling latest main
3. Clear any GitHub Actions cache

### Problem: Workflow fails to start

**Cause**: Secrets or variables missing

**Solution**:
1. Verify secrets: `METACULUS_TOKEN`, `OPENROUTER_API_KEY`
2. Verify variable: `AIB_TOURNAMENT_ID=32813`
3. Check Settings → Secrets and variables → Actions

## 📋 POST-RUN CHECKLIST

After successful re-forecast:

- [ ] Verify all 31 questions have NEW forecasts with proper reasoning
- [ ] Check 3-5 questions manually on Metaculus website
- [ ] Confirm run_summary.json shows expected API usage
- [ ] Restore `SKIP_PREVIOUSLY_FORECASTED: true` in workflow
- [ ] Document any issues encountered
- [ ] Update DEPLOYMENT_STATUS.md with results

## 🎯 NEXT SCHEDULED RUNS

After this manual re-forecast, the bot will run automatically:

- **MiniBench**: Every hour (0 * * * *)
- **Tournament**: Every 5 hours (0 */5 * * *)
- **Quarterly Cup**: Every 2 days (0 0 */2 * *)

All future runs will:
- ✅ Reset circuit breaker at startup
- ✅ Skip fallback publication (return None instead)
- ✅ Only publish forecasts with proper research
- ✅ Skip previously forecasted questions (avoid duplicates)

---

## 🚨 CRITICAL REMINDERS

1. **First Run After Fix**: Set `SKIP_PREVIOUSLY_FORECASTED=false` to update fallback forecasts
2. **After First Run**: Restore `SKIP_PREVIOUSLY_FORECASTED=true` to avoid duplicates
3. **Monitor API Usage**: Should see 1,000+ calls, not 3
4. **Tournament Compliance**: Verify reasoning quality on Metaculus
5. **Cost Budget**: $6-10 expected for 31 questions with proper research

---

*Created: 2025-10-05*
*Fixes committed: ede1bbf*
*Status: Ready for manual trigger*
