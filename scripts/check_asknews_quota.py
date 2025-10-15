#!/usr/bin/env python3
"""
Check AskNews API quota and rate limit status.
This helps diagnose 429 errors and determine if we need to adjust usage patterns.
"""

import os
import sys
from datetime import datetime
import httpx

def check_asknews_quota():
    """Check AskNews API quota status."""

    print("=" * 80)
    print("ASKNEWS API QUOTA CHECK")
    print("=" * 80)
    print()

    # Get credentials from environment
    client_id = os.getenv('ASKNEWS_CLIENT_ID')
    secret = os.getenv('ASKNEWS_SECRET')

    if not client_id or not secret:
        print("❌ AskNews credentials not found in environment")
        print("   Required: ASKNEWS_CLIENT_ID, ASKNEWS_SECRET")
        return

    print(f"✅ Client ID: {client_id[:10]}...{client_id[-4:]}")
    print()

    # Get configured limits
    quota_limit = int(os.getenv('ASKNEWS_QUOTA_LIMIT', '9000'))
    daily_limit = int(os.getenv('ASKNEWS_DAILY_LIMIT', '500'))

    print(f"📊 Configured Limits:")
    print(f"   Monthly Quota: {quota_limit:,}")
    print(f"   Daily Limit: {daily_limit:,}")
    print()

    # Try to make a minimal test call to check rate limit response
    print("🔍 Testing API access...")

    try:
        # Make a minimal search request
        headers = {
            'X-Client-ID': client_id,
            'X-Api-Key': secret,
        }

        params = {
            'query': 'test',
            'n_articles': 1,
            'time_filter': 'crawl_date',
            'hours_back': 1,
            'return_type': 'string',
            'method': 'nl',
        }

        response = httpx.get(
            'https://api.asknews.app/v1/news/search',
            headers=headers,
            params=params,
            timeout=10.0
        )

        print(f"   Status Code: {response.status_code}")

        if response.status_code == 200:
            print("   ✅ API accessible!")

            # Check response headers for rate limit info
            headers_to_check = [
                'X-RateLimit-Remaining',
                'X-RateLimit-Limit',
                'X-RateLimit-Reset',
                'X-Daily-Limit-Remaining',
                'X-Monthly-Quota-Remaining',
            ]

            print()
            print("📈 Rate Limit Headers:")
            for header in headers_to_check:
                value = response.headers.get(header)
                if value:
                    print(f"   {header}: {value}")

            # Parse response to check for quota info
            try:
                data = response.json()
                if isinstance(data, dict) and 'quota' in data:
                    print()
                    print("💰 Quota Information:")
                    print(f"   {data.get('quota')}")
            except:
                pass

        elif response.status_code == 429:
            print("   ⚠️  RATE LIMITED (429)")
            print()
            print("Response:")
            print(f"   {response.text}")
            print()

            # Check Retry-After header
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                print(f"   Retry-After: {retry_after} seconds")

            print()
            print("🔧 Recommended Actions:")
            print("   1. Wait before making more requests")
            print("   2. Reduce ASKNEWS_QUOTA_LIMIT in .env if monthly quota exceeded")
            print("   3. Enable fallback providers (already configured)")
            print("   4. Consider upgrading AskNews plan if needed")

        else:
            print(f"   ⚠️  Unexpected status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")

    except Exception as e:
        print(f"   ❌ Error: {type(e).__name__}: {e}")

    print()
    print("=" * 80)
    print()
    print("💡 Current Configuration:")
    print(f"   - Bot configured with {quota_limit:,} monthly quota")
    print(f"   - Exponential backoff: 12s → 24s → 48s")
    print(f"   - Fallback enabled: DuckDuckGo + Wikipedia")
    print(f"   - Max retries: {os.getenv('ASKNEWS_MAX_RETRIES', '3')}")
    print()
    print("📝 Notes:")
    print("   - Free tier with METACULUSQ4 promo: 9,000 articles/month")
    print("   - Rate limit typically: 10 requests/minute (estimated)")
    print("   - Current strategy: Fail to fallback after 3 attempts")
    print()

if __name__ == "__main__":
    check_asknews_quota()
