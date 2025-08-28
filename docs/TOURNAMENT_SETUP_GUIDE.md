# Tournament Setup Guide - Fall 2025 AI Benchmark Tournament

## Overview

This guide provides step-by-step instructions for setting up the Metaculus AI Forecasting Bot for optimal performance in the Fall 2025 AI Benchmark Tournament with the provided $100 OpenRouter API credit.

## Quick Start

### 1. Environment Configuration

Copy the environment template and configure your settings:

```bash
cp .env.template .env
```

Edit `.env` with your specific configuration:

```bash
# Required - Replace with your actual Metaculus token
METACULUS_TOKEN=your_actual_metaculus_token

# Primary API Key - Set your OpenRouter key here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Budget Management - Pre-configured for $100 budget
BUDGET_LIMIT=100.0
COST_TRACKING_ENABLED=true
```

### 2. Verify Configuration

Run the configuration check script:

```bash
python scripts/check_env.py
```

This will verify:
- All required environment variables are set
- API keys are valid
- Budget tracking is properly configured
- Tournament settings are optimized

### 3. Test Budget Management

Run a budget management test:

```bash
python scripts/test_budget_management.py
```

This will:
- Test cost calculation accuracy
- Verify budget threshold triggers
- Confirm alert system functionality

## Detailed Configuration

### Budget Management Settings

The bot is pre-configured with optimal budget management settings:

| Setting                       | Value | Purpose                             |
| ----------------------------- | ----- | ----------------------------------- |
| `BUDGET_LIMIT`                | 100.0 | Total tournament budget in USD      |
| `CONSERVATIVE_MODE_THRESHOLD` | 0.80  | Switch to cost-saving at 80% budget |
| `EMERGENCY_MODE_THRESHOLD`    | 0.95  | Emergency mode at 95% budget        |
| `MAX_COST_PER_QUESTION`       | 2.00  | Safety limit per question           |

### Model Selection Strategy

The bot uses intelligent model selection to optimize cost-performance:

| Task Type    | Model       | Cost per 1K tokens         | Use Case              |
| ------------ | ----------- | -------------------------- | --------------------- |
| Research     | GPT-4o-mini | $0.15 input, $0.60 output  | Information gathering |
| Forecasting  | GPT-4o      | $2.50 input, $10.00 output | Critical predictions  |
| Simple Tasks | GPT-4o-mini | $0.15 input, $0.60 output  | Basic processing      |

### Tournament Scheduling

Optimized scheduling reduces costs while maintaining competitiveness:

- **Standard Frequency**: Every 4 hours (reduced from 30 minutes)
- **Critical Periods**: Every 2 hours near deadlines
- **Final 24 Hours**: Every hour for last-minute opportunities
- **Expected Questions**: 50-100 total (seasonal scope, not daily)

## Budget Monitoring

### Real-time Monitoring

The bot provides comprehensive budget monitoring:

1. **Budget Dashboard**: Real-time utilization tracking
2. **Cost per Question**: Detailed breakdown of expenses
3. **Alert System**: Notifications at 50%, 75%, 90%, 95% budget usage
4. **Performance Metrics**: Accuracy vs. cost analysis

### Monitoring Commands

Check current budget status:
```bash
python -c "from src.infrastructure.config.budget_manager import BudgetManager; print(BudgetManager().get_budget_status())"
```

View cost tracking logs:
```bash
tail -f logs/budget_alerts.json
```

## Operation Modes

The bot automatically switches between operation modes based on budget utilization:

### Normal Mode (0-80% budget used)
- Uses GPT-4o for forecasting, GPT-4o-mini for research
- Full feature set enabled
- Standard scheduling frequency

### Conservative Mode (80-95% budget used)
- Switches to GPT-4o-mini for most tasks
- Reduces research depth
- Focuses on highest-value questions

### Emergency Mode (95-100% budget used)
- GPT-4o-mini only
- Minimal research
- Critical questions only
- Prepares for graceful shutdown

## Tournament Optimization Features

### Smart Question Selection
- Prioritizes questions with longer time horizons
- Focuses on questions where the bot has competitive advantage
- Avoids over-forecasting on low-value questions

### Calibration Optimization
- Implements overconfidence reduction techniques
- Uses community prediction anchoring
- Optimizes for log scoring performance

### Research Efficiency
- 48-hour news windows for recent developments
- Structured prompts for concise, factual summaries
- Adaptive research depth based on question complexity

## Troubleshooting

### Common Issues

**Budget Alerts Not Working**
```bash
# Check alert configuration
grep BUDGET_ALERT .env

# Test alert system
python scripts/test_budget_core.py
```

**High Cost per Question**
```bash
# Check model selection
python scripts/validate_integration.py

# Review token usage
cat logs/token_usage.json | jq '.[] | select(.cost > 1.0)'
```

**API Key Issues**
```bash
# Verify OpenRouter key
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/models

# Test fallback configuration
python scripts/check_env.py --test-fallbacks
```

### Performance Optimization

**Reduce Costs Further**
1. Increase `CONSERVATIVE_MODE_THRESHOLD` to 0.70
2. Set `MAX_COST_PER_QUESTION` to 1.50
3. Enable `ENABLE_ADAPTIVE_MODEL_SELECTION=true`

**Improve Accuracy**
1. Increase research depth for complex questions
2. Enable community prediction anchoring
3. Use GPT-4o for more tasks (if budget allows)

## Monitoring and Alerts

### GitHub Actions Integration

The bot includes automated monitoring in GitHub Actions:

```yaml
# Budget monitoring workflow runs every 6 hours
- name: Check Budget Status
  run: python scripts/deployment_cost_monitor.py
```

### Alert Channels

Configure alerts through:
1. **GitHub Actions**: Workflow notifications
2. **Log Files**: `logs/budget_alerts.json`
3. **Console Output**: Real-time status updates

## Success Metrics

Track these key metrics for tournament success:

### Budget Efficiency
- **Target**: Stay under $100 for entire tournament
- **Monitor**: Cost per question, daily burn rate
- **Alert**: 95% budget utilization

### Forecasting Performance
- **Target**: Competitive log scores
- **Monitor**: Calibration metrics, accuracy trends
- **Alert**: Performance degradation

### System Reliability
- **Target**: 99% uptime during tournament
- **Monitor**: API success rates, error rates
- **Alert**: System failures or degradation

## Support and Resources

### Documentation
- [Budget Management Implementation](BUDGET_MANAGEMENT_IMPLEMENTATION.md)
- [Tournament Integration Guide](TOURNAMENT_INTEGRATION_GUIDE.md)
- [API Documentation](API_DOCUMENTATION.md)

### Scripts and Tools
- `scripts/test_budget_management.py` - Budget system testing
- `scripts/validate_tournament_integration.py` - Tournament validation
- `scripts/deployment_cost_monitor.py` - Cost monitoring

### Monitoring Dashboards
- Budget utilization: `logs/budget_alerts.json`
- Performance metrics: `logs/performance/`
- Token usage: `logs/token_usage.json`

## Getting Help

If you encounter issues:

1. **Check the logs**: `logs/` directory contains detailed information
2. **Run diagnostics**: Use the provided test scripts
3. **Review configuration**: Ensure all environment variables are correct
4. **Monitor budget**: Keep track of spending to avoid overruns

The bot is designed to be self-managing and will automatically optimize for the tournament constraints while maintaining competitive performance.
