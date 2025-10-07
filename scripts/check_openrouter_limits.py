#!/usr/bin/env python3
"""
Check OpenRouter API key limits and usage.
Uses OpenRouter's /api/v1/auth/key endpoint to get detailed information.
"""
import os
import sys
import requests
import json
from datetime import datetime

def check_openrouter_key():
    """Check OpenRouter API key status and limits."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        print("   Set it first: export OPENROUTER_API_KEY='your-key'")
        return 1

    try:
        print("🔍 Checking OpenRouter API key status...")
        print(f"   Key prefix: {api_key[:15]}...")
        print()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/herman-aukera/metac-bot-ha",
            "X-Title": "Metaculus Tournament Bot"
        }

        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            print(f"❌ API returned status {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return 1

        data = response.json()["data"]

        print("=" * 80)
        print("OPENROUTER API KEY STATUS")
        print("=" * 80)
        print()

        # Basic info
        print(f"📛 Label: {data.get('label', 'N/A')}")
        print(f"🆔 Is free tier: {data.get('is_free_tier', 'unknown')}")
        print()

        # Credit limits
        limit = data.get('limit')
        limit_remaining = data.get('limit_remaining')
        limit_reset = data.get('limit_reset')

        print("💰 CREDIT LIMITS:")
        if limit is None:
            print("   Limit: ♾️  Unlimited")
        else:
            print(f"   Total limit: ${limit:.2f}")

        if limit_remaining is None:
            print("   Remaining: ♾️  Unlimited")
        else:
            print(f"   Remaining: ${limit_remaining:.2f}")
            if limit:
                used = limit - limit_remaining
                pct = (used / limit * 100) if limit > 0 else 0
                print(f"   Used: ${used:.2f} ({pct:.1f}%)")

        if limit_reset:
            print(f"   Reset type: {limit_reset}")
        print()

        # Usage stats
        usage = data.get('usage', 0)
        usage_daily = data.get('usage_daily', 0)
        usage_weekly = data.get('usage_weekly', 0)
        usage_monthly = data.get('usage_monthly', 0)

        print("📊 USAGE STATISTICS:")
        print(f"   All time: ${usage:.4f}")
        print(f"   Today (UTC): ${usage_daily:.4f}")
        print(f"   This week: ${usage_weekly:.4f}")
        print(f"   This month: ${usage_monthly:.4f}")
        print()

        # Rate limit info (if available)
        rate_limit = data.get('rate_limit', {})
        if rate_limit and isinstance(rate_limit, dict):
            print("⏱️  RATE LIMITS:")
            for key, value in rate_limit.items():
                print(f"   {key}: {value}")
            print()

        # Analysis
        print("=" * 80)
        print("ANALYSIS")
        print("=" * 80)
        print()

        # Check if key is exhausted
        if limit and limit_remaining is not None:
            if limit_remaining <= 0:
                print("🚨 CRITICAL: Credit limit exhausted!")
                print("   Action required: Add credits or increase limit")
                print()
            elif limit_remaining < 5:
                print("⚠️  WARNING: Less than $5 remaining")
                print(f"   Can process ~{int(limit_remaining / 0.02)} more questions")
                print()

        # Estimate remaining capacity
        if limit_remaining and limit_remaining > 0:
            # Assuming $0.02 per question (8 calls × $0.0025 avg)
            est_questions = int(limit_remaining / 0.02)
            print(f"📈 Estimated capacity:")
            print(f"   Questions remaining: ~{est_questions}")
            print(f"   (assuming $0.02 per question)")
            print()

        # Daily usage analysis
        if usage_daily > 0:
            print(f"📅 Today's activity:")
            questions_today = int(usage_daily / 0.02)
            print(f"   Estimated questions processed: ~{questions_today}")
            print(f"   Cost per question: ~${usage_daily / max(questions_today, 1):.4f}")
            print()

        # Recommendations
        print("💡 RECOMMENDATIONS:")
        if limit and limit_remaining is not None:
            if limit_remaining <= 0:
                print("   1. Increase credit limit at https://openrouter.ai/settings/keys")
                print("   2. Or add more credits to your account")
                print("   3. Current limit appears insufficient for tournament")
            elif limit_remaining < 20:
                print(f"   1. Current remaining (${limit_remaining:.2f}) is low")
                print("   2. Recommended: Increase limit to $100-150 for tournament")
                print("   3. This allows ~5000-7500 questions before exhaustion")

        if data.get('is_free_tier'):
            print("   ⚠️  Free tier detected - consider upgrading for better limits")

        print()
        print("🔗 Manage key: https://openrouter.ai/settings/keys")
        print()

        return 0

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(check_openrouter_key())
