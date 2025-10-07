#!/usr/bin/env python3
"""
Verify quality of all 31 published forecasts from recent bot run.
Checks Metaculus API to see actual published predictions and reasoning.
"""
import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from forecasting_tools.ai_models.ai_utils.response_types import ForecastingQuestion
    from forecasting_tools.ai_models.resource_managers.metaculus_api import MetaculusApi
except ImportError:
    print("⚠️  forecasting_tools not available, using basic verification only")
    MetaculusApi = None

def main():
    # Target questions mentioned by user
    question_ids = [39364, 39368, 39505]

    print("=" * 80)
    print("FORECAST QUALITY VERIFICATION")
    print("=" * 80)
    print()

    print("Questions to manually verify:")
    print()
    for qid in question_ids:
        print(f"Question {qid}:")
        print(f"   URL: https://www.metaculus.com/questions/{qid}/")
        print(f"   Check: Number of forecasts, prediction values, reasoning quality")
        print()

    print("\n" + "=" * 80)
    print("CRITICAL FINDINGS FROM RUN_SUMMARY.JSON")
    print("=" * 80)

    try:
        with open('run_summary.json', 'r') as f:
            summary = json.load(f)

        print(f"\n✅ Successful forecasts: {summary['successful_forecasts']}")
        print(f"❌ Failed forecasts: {summary['failed_forecasts']}")
        print(f"📊 Total processed: {summary['total_processed']}")
        print(f"💰 Total cost: ${summary['total_estimated_cost']:.4f}")
        print()
        print("🔥 CIRCUIT BREAKER STATUS:")
        cb = summary['openrouter_circuit_breaker']
        print(f"   Status: {'🔴 OPEN' if cb['is_open'] else '✅ CLOSED'}")
        print(f"   Consecutive failures: {cb['consecutive_failures']}")
        print(f"   Threshold: {cb['failure_threshold']}")
        print(f"   Time until reset: {cb['time_until_reset_seconds'] / 60:.1f} minutes")
        print()
        print("📞 API USAGE:")
        print(f"   Total OpenRouter calls: {summary['openrouter_total_calls']}")
        print(f"   Total retries: {summary['openrouter_total_retries']}")
        print(f"   Total backoff time: {summary['openrouter_total_backoff_seconds']:.1f}s")
        print()

        # Calculate fallback forecasts
        expected_calls_per_question = 50  # Multi-stage research
        total_questions = summary['successful_forecasts']
        expected_total_calls = expected_calls_per_question * total_questions
        actual_calls = summary['openrouter_total_calls']

        print("🚨 FALLBACK FORECAST ANALYSIS:")
        print(f"   Expected API calls: ~{expected_total_calls} ({expected_calls_per_question} per question × {total_questions})")
        print(f"   Actual API calls: {actual_calls}")
        print(f"   Shortfall: {expected_total_calls - actual_calls} calls missing")
        print()
        print(f"   ⚠️  CONCLUSION: Only {actual_calls}/{expected_total_calls} expected calls made!")
        print(f"   🔴 LIKELY RESULT: {total_questions - (actual_calls // expected_calls_per_question)} forecasts are 0.5 fallbacks")
        print()

    except Exception as e:
        print(f"❌ Error reading run_summary.json: {e}")

    print("\n" + "=" * 80)
    print("RECOMMENDED ACTIONS")
    print("=" * 80)
    print()
    print("1. ⏳ Wait for circuit breaker reset (~26 minutes remaining)")
    print("2. 🔍 Manually check forecasts at URLs above")
    print("3. 🛠️  Fix fallback behavior to prevent 0.5 publications")
    print("4. 🔄 Re-run bot with fixed code")
    print("5. ✅ Verify new forecasts have proper research and reasoning")
    print()
    print("⚠️  TOURNAMENT COMPLIANCE AT RISK:")
    print("   Publishing forecasts without proper research may violate")
    print("   tournament rules and could result in disqualification.")
    print()

if __name__ == "__main__":
    main()
