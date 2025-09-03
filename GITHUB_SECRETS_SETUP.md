# GitHub Secrets and Variables Setup

This document lists all the required **Secrets** and **Variables** that need to be configured in your GitHub repository for the workflows to function properly.

## 🔐 Required Secrets

### Core API Keys (Required)
- `METACULUS_TOKEN` - Your Metaculus API token for forecasting
- `OPENROUTER_API_KEY` - OpenRouter API key for LLM access

### Optional API Keys (Used if available)
- `ASKNEWS_CLIENT_ID` - AskNews client ID for news research
- `ASKNEWS_SECRET` - AskNews secret key
- `PERPLEXITY_API_KEY` - Perplexity API for additional research
- `EXA_API_KEY` - Exa API for web search
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic Claude API key

### Model Configuration (Optional)
- `METACULUS_DEFAULT_MODEL` - Default model (default: 'metaculus/claude-3-5-sonnet')
- `METACULUS_SUMMARIZER_MODEL` - Summarizer model (default: 'metaculus/gpt-4o-mini')
- `METACULUS_RESEARCH_MODEL` - Research model (default: 'metaculus/gpt-4o')

## 📊 Required Variables

### Tournament Configuration
- `AIB_TOURNAMENT_ID` - Main tournament ID (numeric, e.g., '32813')
- `AIB_TOURNAMENT_SLUG` - Main tournament slug (e.g., 'fall-aib-2025')
- `AIB_MINIBENCH_TOURNAMENT_SLUG` - MiniBench tournament slug (should be 'minibench')
- `AIB_MINIBENCH_TOURNAMENT_ID` - MiniBench tournament ID (numeric, alternative to slug)

### Budget Management
- `BUDGET_LIMIT` - Maximum budget in dollars (default: '100')
- `CURRENT_SPEND` - Current spending amount (default: '0.0')

### Scheduling Configuration (Optional)
- `SCHEDULING_FREQUENCY_HOURS` - Main scheduling frequency (default: '4')
- `DEADLINE_AWARE_SCHEDULING` - Enable deadline-aware scheduling (default: 'true')
- `CRITICAL_PERIOD_FREQUENCY_HOURS` - Frequency during critical periods (default: '2')
- `FINAL_24H_FREQUENCY_HOURS` - Frequency in final 24 hours (default: '1')
- `TOURNAMENT_SCOPE` - Tournament scope (default: 'seasonal')

### Deployment Configuration (Optional)
- `ENABLE_DEPLOY` - Enable deployment workflows (default: 'false')

## 🚀 How to Set Up

### 1. Go to Repository Settings
Navigate to your repository → Settings → Secrets and variables → Actions

### 2. Add Required Secrets
In the "Secrets" tab, add:
```
METACULUS_TOKEN=your_metaculus_token_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

### 3. Add Required Variables
In the "Variables" tab, add:
```
AIB_MINIBENCH_TOURNAMENT_SLUG=minibench
BUDGET_LIMIT=100
```

### 4. Optional: Add Tournament Configuration
If you know the numeric tournament ID, you can use either:
```
# Option A: Use tournament slug (recommended)
AIB_TOURNAMENT_SLUG=fall-aib-2025

# Option B: Use numeric tournament ID
AIB_TOURNAMENT_ID=32813
```

⚠️ **Important**: Make sure tournament IDs are in **Variables**, not **Secrets**!

## ⚠️ Troubleshooting

If workflows are skipping with messages like:
- "MiniBench slug/ID is not configured" → Set `AIB_MINIBENCH_TOURNAMENT_SLUG=minibench` in **Variables** (not Secrets)
- "Missing required API keys" → Check that `METACULUS_TOKEN` and `OPENROUTER_API_KEY` are set in **Secrets**
- "Budget exhausted" → Check `BUDGET_LIMIT` and `CURRENT_SPEND` variables

### Common Configuration Mistakes:
1. **Tournament IDs in Secrets instead of Variables** - Workflows read `vars.AIB_*`, not `secrets.AIB_*`
2. **Wrong tournament ID format** - Use numeric IDs (like '32813') or proper slugs (like 'fall-aib-2025')
3. **MiniBench ID as slug** - Don't set `AIB_MINIBENCH_TOURNAMENT_ID=minibench`, use `AIB_MINIBENCH_TOURNAMENT_SLUG=minibench`

## 🔧 Testing Configuration

### Option 1: Debug Workflow
Run the debug workflow to see what GitHub Actions is reading:
```bash
gh workflow run debug_config.yaml
```

### Option 2: Manual Test
```bash
# Test MiniBench
gh workflow run run_bot_on_minibench.yaml --field tournament_slug=minibench

# Test main tournament
gh workflow run run_bot_on_tournament.yaml --field tournament_slug=fall-aib-2025
```

---

**Note**: Variables are public within your repository, while Secrets are encrypted. Never put sensitive API keys in Variables.
