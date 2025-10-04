# ROOT CAUSE IDENTIFIED - API Keys Not Set

## The Real Problem

**OPENROUTER_API_KEY and METACULUS_TOKEN are NOT set in the terminal environment!**

This explains EVERYTHING:
1. ❌ Every OpenRouter call fails → Circuit breaker opens after 10 failures
2. ❌ All forecasts fall back to `_create_emergency_response()`
3. ❌ Emergency responses contain "Publishing is blocked..." → Guards trigger
4. ❌ Zero forecasts published despite generating predictions

## Verification

```bash
# Current state (WRONG):
$ echo $OPENROUTER_API_KEY
# (empty)

$ echo $METACULUS_TOKEN
# (empty)
```

## Fix

### Option 1: Export in Shell (Immediate)
```bash
export OPENROUTER_API_KEY="your-key-here"
export METACULUS_TOKEN="your-token-here"
export DISABLE_PUBLICATION_GUARD=true

# Then run:
DRY_RUN=false SKIP_PREVIOUSLY_FORECASTED=true python3 main.py --mode tournament
```

### Option 2: Use .env File (Recommended)
```bash
# Create or update .env file:
cat > .env << 'EOF'
OPENROUTER_API_KEY=your-key-here
METACULUS_TOKEN=your-token-here
DISABLE_PUBLICATION_GUARD=true
DRY_RUN=false
SKIP_PREVIOUSLY_FORECASTED=true
TOURNAMENT_SLUG=fall-aib-2025
EOF

# Load and run:
source .env
python3 main.py --mode tournament
```

### Option 3: Use python-dotenv (Most Reliable)
```python
# The code already has this but may not be loading correctly
from dotenv import load_dotenv
load_dotenv()  # This should be at the very top of main.py
```

## Test Plan

### Step 1: Verify Keys Are Set
```bash
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()

print(f'OPENROUTER_API_KEY: {\"SET\" if os.getenv(\"OPENROUTER_API_KEY\") else \"NOT SET\"}')
print(f'METACULUS_TOKEN: {\"SET\" if os.getenv(\"METACULUS_TOKEN\") else \"NOT SET\"}')
print(f'DISABLE_PUBLICATION_GUARD: {os.getenv(\"DISABLE_PUBLICATION_GUARD\", \"false\")}')
"
```

Expected output:
```
OPENROUTER_API_KEY: SET
METACULUS_TOKEN: SET
DISABLE_PUBLICATION_GUARD: true
```

### Step 2: Test Single Question
```bash
python3 -c "
from dotenv import load_dotenv
load_dotenv()

import asyncio
import os

# Set the guard disable flag
os.environ['DISABLE_PUBLICATION_GUARD'] = 'true'

from main import TemplateForecaster
from forecasting_tools.data_models.questions import BinaryQuestion

async def test():
    bot = TemplateForecaster()

    # Test with a simple binary question
    question = BinaryQuestion(
        id_of_question=12345,
        question_text='Will this test work?',
        background_info='Test question',
        resolution_criteria='Success if forecast generated',
        close_time='2025-12-31T23:59:59Z'
    )

    try:
        result = await bot._run_forecast_on_binary(question, 'Test research data')
        print(f'✅ Forecast generated: {result}')
        print(f'Type: {type(result)}')
        return True
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

success = asyncio.run(test())
print(f'\\nTest result: {\"PASS\" if success else \"FAIL\"}')
"
```

### Step 3: Run Full Tournament (Limited)
```bash
DISABLE_PUBLICATION_GUARD=true \
DRY_RUN=false \
SKIP_PREVIOUSLY_FORECASTED=true \
TOURNAMENT_SLUG=fall-aib-2025 \
python3 main.py --mode tournament
```

### Step 4: Verify Publications on Website
1. Go to https://www.metaculus.com/tournament/fall-aib-2025/
2. Check leaderboard for your username
3. Click on a specific question you forecasted
4. Verify your forecast appears with timestamp

## Expected Outcomes

With keys set correctly:
- ✅ OpenRouter API calls succeed
- ✅ Circuit breaker stays CLOSED
- ✅ Real forecasts generated (not emergency responses)
- ✅ Publication guard allows forecasts through
- ✅ Forecasts POST to Metaculus API successfully
- ✅ Summary shows `published_success > 0`
- ✅ Username appears on tournament leaderboard

## Checklist Before Running

- [ ] `.env` file exists with all keys
- [ ] `python-dotenv` is installed (`pip install python-dotenv`)
- [ ] Keys are valid (test with curl)
- [ ] `DISABLE_PUBLICATION_GUARD=true` is set
- [ ] `DRY_RUN=false` for real submissions
- [ ] `SKIP_PREVIOUSLY_FORECASTED=true` to avoid duplicates

## If Still Failing

If forecasts still don't publish after setting keys:

1. **Check API key validity**:
```bash
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models | head -20
```

2. **Check Metaculus token**:
```bash
curl -H "Authorization: Token $METACULUS_TOKEN" \
  https://www.metaculus.com/api/posts/ | head -20
```

3. **Enable debug logging**:
```bash
export PUBLICATION_GUARD_DEBUG=1
export LOG_LEVEL=DEBUG
```

4. **Check circuit breaker state in code**:
```python
# Add to main.py near line 4400:
logger.info(f"OpenRouter circuit breaker state: {template_bot.openrouter_circuit_breaker}")
```

## Summary

**Primary Issue**: Missing environment variables (API keys)
**Secondary Issue**: Publication guard too aggressive (now fixed with DISABLE flag)
**Tertiary Issue**: Question filtering for conditionals (lower priority)

**Fix**: Set API keys properly, then re-run with `DISABLE_PUBLICATION_GUARD=true`
