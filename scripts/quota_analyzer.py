#!/usr/bin/env python3
"""
Quota monitoring script for donated API key management.
Analyzes usage patterns and provides recommendations.
"""

import json
import os
from datetime import datetime
from pathlib import Path


def analyze_quota_usage() -> None:
    """Analyze recent quota usage and provide recommendations."""

    print("=== Donated API Key Quota Analysis ===")

    # Check run summary
    try:
        with open("run_summary.json", "r") as f:
            summary = json.load(f)

        print(f"\n📊 Last Run Analysis:")
        print(f"   • Total questions: {summary.get('total_processed', 0)}")
        print(f"   • Successful forecasts: {summary.get('successful_forecasts', 0)}")
        print(f"   • Failed forecasts: {summary.get('failed_forecasts', 0)}")
        print(f"   • Quota exceeded: {'❌ YES' if summary.get('openrouter_quota_exceeded') else '✅ NO'}")
        print(f"   • Total API calls: {summary.get('openrouter_total_calls', 0)}")
        print(f"   • Cost estimate: ${summary.get('total_estimated_cost', 0):.4f}")

        if summary.get('openrouter_quota_exceeded'):
            print(f"\n⚠️  QUOTA EXCEEDED MESSAGE:")
            print(f"   {summary.get('openrouter_quota_message', 'No message available')}")

    except FileNotFoundError:
        print("❌ No run_summary.json found - run the bot first")
        return
    except json.JSONDecodeError:
        print("❌ Invalid run_summary.json format")
        return

    # Check budget tracking
    try:
        budget_file = Path("logs/budget_tracking.json")
        if budget_file.exists():
            with open(budget_file, "r") as f:
                budget_data = json.load(f)

            print(f"\n💰 Budget Status:")
            print(f"   • Budget limit: ${budget_data.get('budget_limit', 0):.2f}")
            print(f"   • Current spend: ${budget_data.get('current_spend', 0):.4f}")
            remaining = budget_data.get('budget_limit', 0) - budget_data.get('current_spend', 0)
            print(f"   • Remaining: ${remaining:.2f}")

    except Exception as e:
        print(f"⚠️  Could not read budget data: {e}")

    # Provide recommendations
    print(f"\n🎯 Recommendations for Donated API Key:")

    success_rate = summary.get('successful_forecasts', 0) / max(summary.get('total_processed', 1), 1)

    if summary.get('openrouter_quota_exceeded'):
        print("   📅 IMMEDIATE: Switch to 24-hour scheduling interval")
        print("   🔢 REDUCE: Max concurrent questions to 3")
        print("   📉 REDUCE: Predictions per report to 3")
        print("   ⏰ TIMING: Run once per day at off-peak hours")
        print("   📋 SPLIT: Reserve 5 questions/day for minibench")

    elif success_rate < 0.8:
        print("   ⚠️  Consider increasing scheduling interval")
        print("   🔍 Monitor for quota warnings in logs")

    else:
        print("   ✅ Current settings appear sustainable")
        print("   📊 Monitor quota usage trends")

    # Calculate suggested daily limits
    total_questions = summary.get('total_processed', 0)
    if total_questions > 0:
        # Estimate API calls per question (research + forecast + validation)
        api_calls_per_question = summary.get('openrouter_total_calls', 0) / total_questions
        print(f"\n📈 Usage Pattern:")
        print(f"   • API calls per question: ~{api_calls_per_question:.1f}")
        print(f"   • Success rate: {success_rate:.1%}")

        # If quota exceeded, calculate safe daily limits
        if summary.get('openrouter_quota_exceeded'):
            successful_questions = summary.get('successful_forecasts', 0)
            suggested_daily_limit = max(10, int(successful_questions * 0.8))  # 80% of what worked
            print(f"\n💡 Suggested Daily Limits:")
            print(f"   • Tournament questions: {suggested_daily_limit}")
            print(f"   • Minibench questions: 5")
            print(f"   • Total daily budget: {suggested_daily_limit + 5}")


if __name__ == "__main__":
    analyze_quota_usage()
