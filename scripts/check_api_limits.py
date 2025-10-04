#!/usr/bin/env python3
"""
Check OpenRouter API key limits and usage using the token directly.
"""

import os
import requests
import json
from datetime import datetime


def check_api_limits() -> dict | None:
    """Check the actual API key limits and current usage."""

    # Try to get API key from environment or ask user to set it
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ No OPENROUTER_API_KEY found in environment")
        print("Please set it with: export OPENROUTER_API_KEY='your_key_here'")
        return None

    print("=== OpenRouter API Key Analysis ===")

    try:
        # Check key info and limits
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        print("\n🔍 Checking API key details...")
        response = requests.get("https://openrouter.ai/api/v1/auth/key", headers=headers)

        if response.status_code == 200:
            data = response.json()
            print(f"\n📊 API Key Information:")
            print(f"   • Label: {data.get('label', 'N/A')}")
            print(f"   • Usage: ${data.get('usage', 0):.6f}")
            print(f"   • Credit Limit: ${data.get('limit', 'N/A')}")

            # Rate limiting info
            rate_limit = data.get('rate_limit', {})
            if rate_limit:
                print(f"\n⏱️  Rate Limits:")
                print(f"   • Requests per minute: {rate_limit.get('requests', 'N/A')}")
                print(f"   • Tokens per minute: {rate_limit.get('tokens', 'N/A')}")

            # Check if there are per-day limits
            if 'daily_limit' in data:
                print(f"   • Daily limit: {data.get('daily_limit', 'N/A')}")

            # Calculate remaining credits
            usage = float(data.get('usage', 0))
            limit = float(data.get('limit', 0))
            if limit > 0:
                remaining = limit - usage
                percent_used = (usage / limit) * 100
                print(f"\n💰 Credit Status:")
                print(f"   • Remaining: ${remaining:.6f}")
                print(f"   • Percent used: {percent_used:.2f}%")

                if percent_used > 95:
                    print("   ❌ CRITICAL: Nearly out of credits!")
                elif percent_used > 80:
                    print("   ⚠️  WARNING: Credits running low")
                else:
                    print("   ✅ Credit balance healthy")

            return data

        elif response.status_code == 401:
            print("❌ API key is invalid or expired")
            return None
        elif response.status_code == 403:
            print("❌ API key lacks permission or quota exceeded")
            print(f"Response: {response.text}")
            return None
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.RequestException as e:
        print(f"❌ Network error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def analyze_quota_pattern() -> None:
    """Analyze the quota exceeded pattern from logs."""

    print("\n🔍 Analyzing Previous Quota Issues...")

    try:
        with open("run_summary.json", "r") as f:
            summary = json.load(f)

        print(f"\n📈 Last Run Pattern:")
        start_time = summary.get('started_at')
        finish_time = summary.get('finished_at')

        if start_time and finish_time:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            finish_dt = datetime.fromisoformat(finish_time.replace('Z', '+00:00'))
            duration = finish_dt - start_dt

            print(f"   • Start: {start_dt.strftime('%H:%M:%S')}")
            print(f"   • Finish: {finish_dt.strftime('%H:%M:%S')}")
            print(f"   • Duration: {duration.total_seconds()/60:.1f} minutes")

        total_questions = summary.get('total_processed', 0)
        successful = summary.get('successful_forecasts', 0)

        if total_questions > 0:
            success_rate = successful / total_questions
            print(f"   • Questions processed: {successful}/{total_questions} ({success_rate:.1%})")

            if success_rate > 0.8:
                print("   ✅ High success rate - likely hit request rate limit, not daily quota")
                print("   💡 Suggestion: Keep 6-hour frequency, but reduce concurrent questions")
            else:
                print("   ⚠️  Lower success rate - might be daily quota limit")
                print("   💡 Suggestion: Space out requests more (8-12 hour frequency)")

    except Exception as e:
        print(f"⚠️  Could not analyze run pattern: {e}")


def recommend_frequency() -> None:
    """Recommend optimal frequency based on analysis."""

    print("\n🎯 Frequency Recommendations:")

    # Get API key data
    api_data = check_api_limits()

    if api_data:
        rate_limit = api_data.get('rate_limit', {})
        requests_per_min = rate_limit.get('requests')

        if requests_per_min:
            print(f"\n⏱️  Based on rate limits ({requests_per_min} req/min):")

            # Estimate requests per question (research + forecast + validation)
            requests_per_question = 5  # Conservative estimate

            max_questions_per_run = int(requests_per_min * 0.8 / requests_per_question)  # 80% safety margin
            print(f"   • Safe questions per run: ~{max_questions_per_run}")

            if max_questions_per_run >= 10:
                print("   💡 6 hours frequency should work fine")
            elif max_questions_per_run >= 5:
                print("   💡 8-12 hours frequency recommended")
            else:
                print("   💡 24 hours frequency needed")

    # Analyze previous pattern
    analyze_quota_pattern()

    print(f"\n🏆 Tournament Considerations:")
    print(f"   • Active tournament needs regular updates (6-12 hours)")
    print(f"   • Minibench runs separately (coordinate timing)")
    print(f"   • Weekend/off-peak hours may have higher quotas")


if __name__ == "__main__":
    recommend_frequency()
