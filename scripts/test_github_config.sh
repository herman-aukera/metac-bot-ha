#!/bin/bash
# Test script to validate GitHub Actions secrets and variables loading

echo "🔍 Testing GitHub Actions Secrets and Variables Configuration"
echo "=============================================================="

# Test required secrets (these should be defined but not readable)
echo ""
echo "📋 Required Secrets Status:"
echo "METACULUS_TOKEN: ${METACULUS_TOKEN:+✅ Set}${METACULUS_TOKEN:-❌ Missing}"
echo "OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:+✅ Set}${OPENROUTER_API_KEY:-❌ Missing}"

# Test optional secrets
echo ""
echo "📋 Optional Secrets Status:"
echo "ASKNEWS_CLIENT_ID: ${ASKNEWS_CLIENT_ID:+✅ Set}${ASKNEWS_CLIENT_ID:-⚠️ Not set (optional)}"
echo "ASKNEWS_SECRET: ${ASKNEWS_SECRET:+✅ Set}${ASKNEWS_SECRET:-⚠️ Not set (optional)}"
echo "PERPLEXITY_API_KEY: ${PERPLEXITY_API_KEY:+✅ Set}${PERPLEXITY_API_KEY:-⚠️ Not set (optional)}"
echo "EXA_API_KEY: ${EXA_API_KEY:+✅ Set}${EXA_API_KEY:-⚠️ Not set (optional)}"
echo "OPENAI_API_KEY: ${OPENAI_API_KEY:+✅ Set}${OPENAI_API_KEY:-⚠️ Not set (optional)}"
echo "ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:+✅ Set}${ANTHROPIC_API_KEY:-⚠️ Not set (optional)}"

# Test variables (these can be shown)
echo ""
echo "📊 Variables Status:"
echo "BUDGET_LIMIT: ${BUDGET_LIMIT:-100 (default)}"
echo "CURRENT_SPEND: ${CURRENT_SPEND:-0.0 (default)}"
echo "AIB_TOURNAMENT_ID: ${AIB_TOURNAMENT_ID:-32813 (default)}"
echo "AIB_TOURNAMENT_SLUG: ${AIB_TOURNAMENT_SLUG:-not set}"
echo "AIB_MINIBENCH_TOURNAMENT_SLUG: ${AIB_MINIBENCH_TOURNAMENT_SLUG:-not set}"
echo "AIB_MINIBENCH_TOURNAMENT_ID: ${AIB_MINIBENCH_TOURNAMENT_ID:-not set}"

# Check critical missing configuration
echo ""
echo "🚨 Critical Issues:"
ISSUES=0

if [ -z "${METACULUS_TOKEN}" ]; then
    echo "❌ METACULUS_TOKEN is required but not set"
    ((ISSUES++))
fi

if [ -z "${OPENROUTER_API_KEY}" ]; then
    echo "❌ OPENROUTER_API_KEY is required but not set"
    ((ISSUES++))
fi

if [ -z "${AIB_MINIBENCH_TOURNAMENT_SLUG}" ] && [ -z "${AIB_MINIBENCH_TOURNAMENT_ID}" ]; then
    echo "❌ MiniBench tournament not configured - set AIB_MINIBENCH_TOURNAMENT_SLUG='minibench'"
    ((ISSUES++))
fi

if [ "$ISSUES" -eq 0 ]; then
    echo "✅ No critical configuration issues found"
else
    echo "❌ Found $ISSUES critical configuration issue(s)"
fi

echo ""
echo "💡 Setup Instructions:"
echo "1. Go to: Repository → Settings → Secrets and variables → Actions"
echo "2. Add required secrets: METACULUS_TOKEN, OPENROUTER_API_KEY"
echo "3. Add required variable: AIB_MINIBENCH_TOURNAMENT_SLUG=minibench"
echo "4. See GITHUB_SECRETS_SETUP.md for complete setup guide"

exit $ISSUES
