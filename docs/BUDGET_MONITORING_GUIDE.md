# Budget Management and Monitoring Guide

## Overview

This guide covers the comprehensive budget management and monitoring system for the Fall 2025 AI Benchmark Tournament. The system ensures you stay within the $100 budget while maintaining competitive forecasting performance.

## Cost Tracking Features

### Real-time Budget Monitoring

The budget manager provides continuous tracking of:
- Total budget utilization (current spend vs. $100 limit)
- Cost per question breakdown
- Daily spending patterns
- Model-specific cost analysis
- Token usage statistics

### Budget Status Levels

**Normal Mode (0-80% budget used)**
- Full functionality enabled
- Optimal model selection for accuracy
- Standard research depth
- All features available

**Conservative Mode (80-95% budget used)**
- Switches to cost-effective models
- Reduced research depth
- Focus on high-value questions
- Increased monitoring frequency

**Emergency Mode (95-100% budget used)**
- GPT-4o-mini only for all tasks
- Minimal research operations
- Critical questions prioritized
- Automatic cost controls activated

### Cost Calculation Accuracy

The system uses precise cost calculation based on:
- Exact token counting with tiktoken library
- Real-time model pricing from OpenRouter
- Input and output token differentiation
- Context length optimization

## Monitoring Commands

### Check Current Budget Status
```bash
python -c "from src.infrastructure.config.budget_manager import budget_manager; print(budget_manager.get_budget_status())"
```

### View Detailed Cost Breakdown
```bash
python scripts/budget_analysis.py --detailed
```

### Monitor Real-time Spending
```bash
tail -f logs/budget_tracking.json
```
### Alert System Configuration

The budget alert system triggers at these thresholds:
- **50% budget used**: First warning notification
- **75% budget used**: Increased monitoring recommended
- **90% budget used**: Conservative mode suggested
- **95% budget used**: Emergency mode activation

### Budget Dashboard Access

View the budget dashboard:
```bash
python -m src.infrastructure.monitoring.budget_dashboard
```

This provides:
- Visual budget utilization charts
- Cost trend analysis
- Model usage statistics
- Performance vs. cost metrics

## Troubleshooting Budget Issues

### High Cost per Question

**Symptoms:**
- Questions costing more than $2.00
- Rapid budget depletion
- Frequent emergency mode activation
**Diagnosis Steps:**
1. Check model selection logic:
   ```bash
   python scripts/validate_model_selection.py
   ```

2. Review token usage patterns:
   ```bash
   grep "high_token_usage" logs/budget_tracking.json
   ```

3. Analyze prompt efficiency:
   ```bash
   python scripts/analyze_prompt_costs.py
   ```

**Solutions:**
- Reduce `MAX_COST_PER_QUESTION` to $1.50
- Enable `ENABLE_ADAPTIVE_MODEL_SELECTION=true`
- Increase `CONSERVATIVE_MODE_THRESHOLD` to 0.70
- Review and optimize research prompts

### Budget Alerts Not Triggering

**Symptoms:**
- No alert notifications despite high budget usage
- Missing budget threshold warnings
- Silent budget overruns
**Diagnosis Steps:**
1. Verify alert configuration:
   ```bash
   grep BUDGET_ALERT .env
   ```

2. Test alert system:
   ```bash
   python scripts/test_budget_alerts.py
   ```

3. Check alert cooldown settings:
   ```bash
   python -c "from src.infrastructure.config.budget_alerts import BudgetAlerts; print(BudgetAlerts().get_alert_status())"
   ```

**Solutions:**
- Ensure `BUDGET_ALERT_ENABLED=true` in .env
- Verify `BUDGET_ALERT_THRESHOLDS` are properly set
- Check log file permissions for `logs/budget_alerts.json`
- Restart the monitoring system

### Inaccurate Cost Tracking

**Symptoms:**
- Costs don't match actual API usage
- Budget calculations seem incorrect
- Unexpected budget overruns
**Diagnosis Steps:**
1. Validate token counting:
   ```bash
   python scripts/validate_token_counting.py
   ```

2. Compare with OpenRouter usage:
   ```bash
   curl -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/usage
   ```

3. Check model pricing updates:
   ```bash
   python scripts/update_model_pricing.py --check
   ```

**Solutions:**
- Update tiktoken library: `pip install --upgrade tiktoken`
- Refresh model pricing data
- Recalibrate token estimation algorithms
- Enable detailed logging for cost tracking

## Performance Optimization Recommendations

### Cost-Performance Balance

**For Maximum Accuracy (if budget allows):**
- Use GPT-4o for all forecasting tasks
- Enable deep research mode
- Increase context window utilization
- Use community prediction anchoring
**For Maximum Efficiency (recommended):**
- Use GPT-4o-mini for research tasks
- Reserve GPT-4o for complex forecasting only
- Implement smart question prioritization
- Enable adaptive model selection

**For Emergency Budget Conservation:**
- GPT-4o-mini for all tasks
- Minimal research depth
- Focus on highest-value questions only
- Skip low-confidence predictions

### Model Selection Optimization

**Research Tasks:**
- Primary: `openai/gpt-4o-mini` ($0.15/$0.60 per 1K tokens)
- Fallback: `anthropic/claude-3-haiku` (if available)
- Emergency: Local processing (if implemented)

**Forecasting Tasks:**
- Simple questions: `openai/gpt-4o-mini`
- Complex questions: `openai/gpt-4o` ($2.50/$10.00 per 1K tokens)
- Emergency: `openai/gpt-4o-mini` only

### Token Usage Optimization

**Prompt Engineering:**
- Use structured, concise prompts
- Implement prompt templates
- Remove unnecessary context
- Optimize for token efficiency
**Context Management:**
- Limit research context to 48-hour windows
- Use summarization for long documents
- Implement smart context pruning
- Cache frequently used information

### Scheduling Optimization

**Standard Operation:**
- Forecast every 4 hours (reduced from 30 minutes)
- Focus on questions with longer time horizons
- Batch similar questions together

**Critical Periods:**
- Increase frequency to every 2 hours near deadlines
- Final 24 hours: hourly checks
- Emergency mode: manual intervention only

## Advanced Monitoring Features

### Cost Prediction

The system provides forward-looking cost estimates:
- Projected tournament total based on current usage
- Daily burn rate analysis
- Question completion rate vs. budget utilization
- Early warning for budget overrun risk

### Performance Metrics Integration

Track cost-effectiveness with these metrics:
- Cost per log score point
- Accuracy vs. spending correlation
- ROI analysis for different question types
- Calibration improvement per dollar spent
### Automated Reporting

The system generates automated reports:
- Daily budget utilization summaries
- Weekly performance vs. cost analysis
- Model efficiency comparisons
- Alert history and trend analysis

Access reports:
```bash
# Daily summary
python scripts/generate_daily_report.py

# Weekly analysis
python scripts/generate_weekly_analysis.py

# Custom date range
python scripts/generate_custom_report.py --start 2025-09-01 --end 2025-09-07
```

## Integration with Tournament Features

### Compliance Monitoring

Budget management integrates with tournament compliance:
- Ensures spending stays within tournament rules
- Tracks API usage for transparency requirements
- Monitors model selection compliance
- Validates cost reporting accuracy

### Calibration Optimization

Cost-aware calibration features:
- Balances accuracy improvement with budget constraints
- Optimizes research depth based on available budget
- Implements smart question prioritization
- Maintains competitive performance within budget limits
