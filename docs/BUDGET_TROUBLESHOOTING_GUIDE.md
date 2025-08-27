# Budget Troubleshooting Guide

## Quick Diagnostic Commands

### Check System Status
```bash
# Overall system health
python scripts/system_health_check.py

# Budget system status
python scripts/budget_system_check.py

# API connectivity test
python scripts/test_api_connections.py
```

### Common Issues and Solutions

## Issue: Budget Exceeded Unexpectedly

**Symptoms:**
- Spending above $100 limit
- Emergency mode not triggered
- Costs higher than estimates

**Immediate Actions:**
1. Stop all automated processes:
   ```bash
   pkill -f "python main.py"
   ```

2. Check actual spending:
   ```bash
   python scripts/verify_actual_costs.py
   ```

3. Review recent transactions:
   ```bash
   tail -50 logs/budget_tracking.json | jq '.[] | select(.timestamp > "2025-08-25")'
   ```
**Root Cause Analysis:**
- Check for token counting errors
- Verify model pricing accuracy
- Review emergency mode thresholds
- Analyze unexpected high-cost operations

**Prevention:**
- Lower `EMERGENCY_MODE_THRESHOLD` to 0.90
- Set `MAX_COST_PER_QUESTION` to $1.00
- Enable `STRICT_BUDGET_ENFORCEMENT=true`

## Issue: Performance Degradation in Conservative Mode

**Symptoms:**
- Lower forecasting accuracy
- Reduced research quality
- Competitive disadvantage

**Diagnosis:**
1. Compare performance metrics:
   ```bash
   python scripts/compare_performance_modes.py
   ```

2. Analyze model selection impact:
   ```bash
   python scripts/model_performance_analysis.py
   ```

**Optimization:**
- Use GPT-4o for complex questions only
- Implement smart question complexity detection
- Optimize prompt efficiency for GPT-4o-mini
- Enable community prediction anchoring
## Issue: API Key Problems

**Symptoms:**
- Authentication failures
- Unexpected API costs
- Service unavailable errors

**OpenRouter API Key Issues:**
1. Verify key validity:
   ```bash
   curl -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/models
   ```

2. Check remaining credits:
   ```bash
   curl -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/usage
   ```

3. Test model access:
   ```bash
   python scripts/test_model_access.py --model openai/gpt-4o-mini
   ```

**Fallback Configuration:**
- Ensure fallback keys are properly configured
- Test fallback chain functionality
- Verify graceful degradation works

## Emergency Procedures

### Budget Overrun Emergency

1. **Immediate Stop:**
   ```bash
   touch .emergency_stop
   pkill -f "main.py"
   ```

2. **Damage Assessment:**
   ```bash
   python scripts/emergency_budget_assessment.py
   ```

3. **Recovery Planning:**
   ```bash
   python scripts/generate_recovery_plan.py
   ```
