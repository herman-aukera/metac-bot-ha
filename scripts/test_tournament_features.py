#!/usr/bin/env python3
"""
Comprehensive testing script for tournament features.
Tests all tournament optimizations and fallback mechanisms.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.main import MetaculusForecastingBot

from src.infrastructure.config.settings import Config
from src.infrastructure.external_apis.metaculus_proxy_client import MetaculusProxyClient
from src.infrastructure.external_apis.tournament_asknews_client import (
    TournamentAskNewsClient,
)


async def test_tournament_features():
    """Test all tournament features comprehensively."""
    print("🧪 TOURNAMENT FEATURES TESTING")
    print("=" * 50)

    config = Config()
    bot = MetaculusForecastingBot(config)

    # Test 1: Single Question Forecast
    print("\n1. Testing Single Question Forecast...")
    try:
        result = await bot.forecast_question(12345, "chain_of_thought")
        print(f"   ✅ Single forecast successful")
        print(f"   📊 Question: {result['question']['title'][:50]}...")
        print(f"   📊 Prediction: {result['forecast']['prediction']:.3f}")
        print(f"   📊 Confidence: {result['forecast']['confidence']:.3f}")

        # Check for tournament optimizations in metadata
        if "error" in result["metadata"]:
            print(
                f"   ⚠️  Research error handled: {result['metadata']['error'][:50]}..."
            )
        else:
            print(f"   ✅ Research completed successfully")

    except Exception as e:
        print(f"   ❌ Single forecast failed: {e}")
        return False

    # Test 2: Ensemble Forecast
    print("\n2. Testing Ensemble Forecast...")
    try:
        agents = ["chain_of_thought", "tree_of_thought", "react"]
        ensemble_result = await bot.forecast_question_ensemble(12346, agents)

        print(f"   ✅ Ensemble forecast successful")
        print(f"   📊 Agents used: {len(ensemble_result['individual_forecasts'])}")
        print(
            f"   📊 Ensemble prediction: {ensemble_result['ensemble_forecast']['prediction']:.3f}"
        )
        print(
            f"   📊 Consensus strength: {ensemble_result['metadata']['consensus_strength']:.3f}"
        )

        # Check individual agent results
        for i, forecast in enumerate(ensemble_result["individual_forecasts"]):
            print(
                f"      Agent {i+1} ({forecast['agent']}): {forecast['prediction']:.3f}"
            )

    except Exception as e:
        print(f"   ❌ Ensemble forecast failed: {e}")
        return False

    # Test 3: Batch Processing
    print("\n3. Testing Batch Processing...")
    try:
        question_ids = [12347, 12348, 12349]
        batch_results = await bot.forecast_questions_batch(
            question_ids, "chain_of_thought"
        )

        successful = len([r for r in batch_results if "error" not in r])
        print(f"   ✅ Batch processing completed")
        print(f"   📊 Questions processed: {len(batch_results)}")
        print(f"   📊 Successful forecasts: {successful}")
        print(f"   📊 Success rate: {successful/len(batch_results)*100:.1f}%")

    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")
        return False

    # Test 4: Tournament Mode
    print("\n4. Testing Tournament Mode...")
    try:
        # Test with small number of questions
        tournament_results = await bot.run_tournament(32813, max_questions=3)

        print(f"   ✅ Tournament mode completed")
        print(f"   📊 Tournament ID: {tournament_results['tournament_id']}")
        print(f"   📊 Total questions: {tournament_results['total_questions']}")
        print(
            f"   📊 Successful forecasts: {tournament_results['successful_forecasts']}"
        )
        print(f"   📊 Success rate: {tournament_results['success_rate']:.1f}%")

    except Exception as e:
        print(f"   ❌ Tournament mode failed: {e}")
        return False

    # Test 5: Resource Management
    print("\n5. Testing Resource Management...")
    try:
        asknews_stats = bot.tournament_asknews.get_usage_stats()
        proxy_stats = bot.metaculus_proxy.get_usage_stats()

        print(f"   ✅ Resource management working")
        print(
            f"   📊 AskNews quota usage: {asknews_stats['quota_usage_percentage']:.1f}%"
        )
        print(f"   📊 AskNews success rate: {asknews_stats['success_rate']:.1f}%")
        print(f"   📊 Proxy requests: {proxy_stats['total_requests']}")
        print(f"   📊 Proxy fallback rate: {proxy_stats['fallback_rate']:.1f}%")

        # Check quota limits
        if asknews_stats["quota_usage_percentage"] > 80:
            print(
                f"   ⚠️  Warning: AskNews quota usage high ({asknews_stats['quota_usage_percentage']:.1f}%)"
            )

    except Exception as e:
        print(f"   ❌ Resource management failed: {e}")
        return False

    # Test 6: Error Handling and Fallbacks
    print("\n6. Testing Error Handling and Fallbacks...")
    try:
        # Test with invalid question ID to trigger fallbacks
        fallback_result = await bot.forecast_question(999999, "chain_of_thought")

        print(f"   ✅ Fallback mechanisms working")
        print(
            f"   📊 Fallback prediction: {fallback_result['forecast']['prediction']:.3f}"
        )

        # Check if error was handled gracefully
        if "error" in fallback_result["metadata"]:
            print(
                f"   ✅ Error handled gracefully: {fallback_result['metadata']['error'][:50]}..."
            )

    except Exception as e:
        print(f"   ❌ Error handling failed: {e}")
        return False

    print("\n" + "=" * 50)
    print("🧪 TOURNAMENT FEATURES TESTING COMPLETE")
    print("✅ All tournament features working correctly!")
    print("🏆 Bot ready for competitive forecasting!")

    return True


def main():
    """Main testing function."""
    print("Starting comprehensive tournament features testing...")

    # Check environment
    env_warnings = []
    if not os.path.exists(".env"):
        env_warnings.append("⚠️  .env file not found")

    if not os.getenv("ASKNEWS_CLIENT_ID"):
        env_warnings.append("⚠️  ASKNEWS_CLIENT_ID not set")

    if not os.getenv("ASKNEWS_SECRET"):
        env_warnings.append("⚠️  ASKNEWS_SECRET not set")

    if not os.getenv("OPENROUTER_API_KEY"):
        env_warnings.append("⚠️  OPENROUTER_API_KEY not set")

    if env_warnings:
        print("Environment warnings:")
        for warning in env_warnings:
            print(f"  {warning}")
        print("Some features may use fallback mechanisms.\n")

    # Run comprehensive testing
    success = asyncio.run(test_tournament_features())

    if success:
        print("\n🎉 ALL TESTS PASSED - Tournament bot is fully operational!")
        print("🚀 Ready to dominate the Fall 2025 tournament!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Please check the errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
