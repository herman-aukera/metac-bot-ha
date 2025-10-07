# GitHub Actions Configuration Guide

## Required Secrets (Settings → Secrets and variables → Actions → Secrets)

### Essential Secrets

```
METACULUS_TOKEN
  Value: cb6cc028c85dba9fef4b3ea874679cc964c350d0
  Purpose: Authentication for Metaculus API (forecast submission)
  Required: YES

OPENROUTER_API_KEY
  Value: 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
  Purpose: Primary LLM API access (paid tier, $143 remaining)
  Required: YES
```

### Optional Secrets (Recommended)

```
ASKNEWS_CLIENT_ID
  Value: 8d71be6d-49bd-4730-80b9-30461c65537e
  Purpose: AskNews API for research (free via METACULUSQ4 promo)
  Required: NO (but recommended - free research)
  Default: Bot will skip AskNews if missing

ASKNEWS_SECRET
  Value: GhdYAxf1THLQPPOhVL4DernAtz
  Purpose: AskNews API authentication
  Required: NO (but recommended)
  Default: Bot will skip AskNews if missing
```

### Optional Secrets (Not Currently Used)

```
OPENROUTER_API_KEY_2
  Value: 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
  Purpose: Secondary OpenRouter key (free tier, for load balancing)
  Required: NO
  Default: Uses primary key only
  Note: Currently disabled in code

PERPLEXITY_API_KEY
  Value: [not set - expensive]
  Purpose: Perplexity research API (disabled to save costs)
  Required: NO
  Default: Disabled (ENABLE_PERPLEXITY_RESEARCH=false)

EXA_API_KEY
  Value: [not set - not needed]
  Purpose: Exa search API
  Required: NO
  Default: Bot uses DuckDuckGo + Wikipedia instead

OPENAI_API_KEY
  Value: [not set - using OpenRouter]
  Purpose: Direct OpenAI access
  Required: NO
  Default: All LLM calls go through OpenRouter

ANTHROPIC_API_KEY
  Value: [not set - using OpenRouter]
  Purpose: Direct Anthropic access
  Required: NO
  Default: All LLM calls go through OpenRouter
```

---

## Repository Variables (Settings → Secrets and variables → Actions → Variables)

### Essential Variables

```
AIB_TOURNAMENT_ID
  Value: 32813
  Purpose: Primary tournament ID (Fall 2025 AI Benchmark)
  Required: YES (for tournament mode)
  Default: minibench (if not set)

AIB_TOURNAMENT_SLUG
  Value: fall-aib-2025
  Purpose: Alternative tournament identifier
  Required: NO (ID takes precedence)
  Default: Uses AIB_TOURNAMENT_ID if not set
```

### Optional Variables (MiniBench Support)

```
AIB_MINIBENCH_TOURNAMENT_ID
  Value: [not set - using primary tournament]
  Purpose: MiniBench-specific tournament ID
  Required: NO
  Default: Uses main tournament if not set

AIB_MINIBENCH_TOURNAMENT_SLUG
  Value: minibench
  Purpose: MiniBench tournament slug
  Required: NO
  Default: Uses main tournament if not set
  Note: Only set if you want separate MiniBench runs
```

---

## Workflow Environment Variables (Already Configured in YAML)

### All Workflows Have These Defaults

```yaml
env:
  # Required for API access
  METACULUS_TOKEN: ${{ secrets.METACULUS_TOKEN }}
  OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}

  # Optional but recommended
  ASKNEWS_CLIENT_ID: ${{ secrets.ASKNEWS_CLIENT_ID }}
  ASKNEWS_SECRET: ${{ secrets.ASKNEWS_SECRET }}

  # Production settings (set in workflow YAML)
  DRY_RUN: false
  DISABLE_PUBLICATION_GUARD: true
  SKIP_PREVIOUSLY_FORECASTED: true
  PUBLISH_REPORTS: true
  TOURNAMENT_MODE: true

  # Budget and limits
  BUDGET_LIMIT: 50.0
  MAX_RESEARCH_REPORTS_PER_QUESTION: 1
  MAX_PREDICTIONS_PER_REPORT: 5
```

---

## Verification Checklist

### Before First Run

- [ ] `METACULUS_TOKEN` secret set
- [ ] `OPENROUTER_API_KEY` secret set
- [ ] `AIB_TOURNAMENT_ID` variable set to `32813`
- [ ] `ASKNEWS_CLIENT_ID` secret set (optional)
- [ ] `ASKNEWS_SECRET` secret set (optional)

### After Setting Up

1. **Test with workflow_dispatch**:
   ```
   Actions → run_bot_on_tournament → Run workflow → Run
   ```

2. **Check run logs** for:
   ```
   ✅ "METACULUS_TOKEN is set"
   ✅ "OPENROUTER_API_KEY is set"
   ✅ "Tournament target resolved: 32813"
   ```

3. **Verify forecast submission**:
   - Check run_summary.json artifact
   - Look for `successful_forecasts > 0`
   - Verify `published_success > 0` (if comments published)

---

## Common Issues

### Issue: "METACULUS_TOKEN is missing"
**Fix**: Add `METACULUS_TOKEN` to repository secrets

### Issue: "AIB_TOURNAMENT_ID not configured"
**Fix**: Add `AIB_TOURNAMENT_ID=32813` to repository variables

### Issue: "Key limit exceeded" (OpenRouter)
**Cause**: Hitting rate limits due to too many API calls
**Fix**:
- Circuit breaker will auto-reset after 1 hour
- Schedules already adjusted (1hr/5hr)
- Consider adding delays between questions

### Issue: Bot publishes 50% forecasts
**Cause**: Circuit breaker opened, fallback forecasts submitted
**Fix**:
- Wait for circuit breaker to reset
- Check OpenRouter credits
- Reduce research complexity if rate limiting persists

---

## Cost Monitoring

### Current Usage
- **Budget**: $50.00
- **Spent**: $0.014 (0.03%)
- **Remaining**: $49.986
- **Questions processed**: 31
- **Cost per question**: ~$0.0005

### OpenRouter Credits
- **Remaining**: $143.68 / $150.00
- **Spent**: $6.32
- **Rate**: ~$0.20 per question with full research

### AskNews Quota
- **Total**: 9,000 requests (free via METACULUSQ4)
- **Used**: 3 requests
- **Remaining**: 8,997

---

## Schedule Summary

```yaml
# MiniBench (run_bot_on_minibench.yaml)
schedule:
  - cron: '0 * * * *'  # Every hour

# Main Tournament (run_bot_on_tournament.yaml)
schedule:
  - cron: '0 */5 * * *'  # Every 5 hours

# Quarterly Cup (run_bot_on_quarterly_cup.yaml)
schedule:
  - cron: '0 0 */2 * *'  # Every 2 days
```

---

## Security Best Practices

1. **Never commit secrets** to repository
2. **Rotate tokens** if accidentally exposed
3. **Use repository secrets** for all sensitive values
4. **Limit secret access** to necessary workflows only
5. **Monitor API usage** regularly

---

## Quick Setup Commands

### 1. Set Required Secrets
```bash
# Via GitHub CLI (if installed)
gh secret set METACULUS_TOKEN
gh secret set OPENROUTER_API_KEY

# Via GitHub UI
# Settings → Secrets and variables → Actions → New repository secret
```

### 2. Set Tournament Variable
```bash
# Via GitHub CLI
gh variable set AIB_TOURNAMENT_ID --body "32813"

# Via GitHub UI
# Settings → Secrets and variables → Actions → Variables → New repository variable
```

### 3. Test Workflow
```bash
# Via GitHub CLI
gh workflow run run_bot_on_tournament.yaml

# Via GitHub UI
# Actions → run_bot_on_tournament → Run workflow
```

---

## Summary

**Minimum Required**:
- `METACULUS_TOKEN` (secret)
- `OPENROUTER_API_KEY` (secret)
- `AIB_TOURNAMENT_ID=32813` (variable)

**Recommended**:
- `ASKNEWS_CLIENT_ID` (secret) - Free research
- `ASKNEWS_SECRET` (secret) - Free research

**Optional** (not currently needed):
- All other API keys (disabled or using OpenRouter)
- MiniBench variables (if using separate MiniBench runs)

The workflows have **safe defaults** for all other variables - they will work with just the minimum required secrets and variables.
