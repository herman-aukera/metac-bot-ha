# Fast Testing Configuration
# =============================================================================
# Use this configuration for rapid testing of closed question filter
# =============================================================================

# Copy these to your .env file for fast testing:

# 1. DISABLE ASKNEWS (use free fallback immediately)
ASKNEWS_QUOTA_LIMIT=0              # 0=Fast testing (skip AskNews), 9000=Production

# 2. REDUCE RETRY DELAYS
ASKNEWS_MAX_RETRIES=1              # Fast fail to fallback (1 try = ~12s max delay)

# 3. LIMIT QUESTIONS FOR TESTING
MAX_QUESTIONS_PER_RUN=3            # Small batch for quick validation

# =============================================================================
# Expected Behavior with This Configuration
# =============================================================================

# ✅ AskNews calls: Skipped immediately (quota=0)
# ✅ Fallback: DuckDuckGo + Wikipedia (free, fast)
# ✅ Closed questions: Filtered with warning message
# ✅ Open questions: Fully processed with forecast
# ✅ Run time: ~2-5 minutes (vs 15 hours with rate limits)

# =============================================================================
# Production Configuration
# =============================================================================

# For real tournament runs, use:
ASKNEWS_QUOTA_LIMIT=9000           # Full quota
ASKNEWS_MAX_RETRIES=3              # Full retry logic
MAX_QUESTIONS_PER_RUN=10           # Standard batch size

# =============================================================================
# Test Commands
# =============================================================================

# Test 1: Dry run (no API costs, validates filter)
# DRY_RUN=true SKIP_PREVIOUSLY_FORECASTED=true MAX_QUESTIONS_PER_RUN=3 python3 main.py --mode tournament

# Test 2: Small production run (validates publication)
# DRY_RUN=false SKIP_PREVIOUSLY_FORECASTED=false MAX_QUESTIONS_PER_RUN=5 python3 main.py --mode tournament

# Test 3: Full production run
# DRY_RUN=false SKIP_PREVIOUSLY_FORECASTED=true MAX_QUESTIONS_PER_RUN=15 python3 main.py --mode tournament
