#!/usr/bin/env python3
"""
Validation script for tournament integration.
Tests all critical components and tournament optimizations.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.config.settings import Config
from src.infrastructure.external_apis.tournament_asknews_client import TournamentAskNewsClient
from src.infrastructure.external_apis.metaculus_proxy_client import MetaculusProxyClient
from src.main import MetaculusForecastingBot


async def validate_tournament_integration():
    """Validate all tournament integration components."""
    print("🏆 TOURNAMENT INTEGRATION VALIDATION")
    print("=" * 50)

    # Test 1: Configuration Loading
    print("\n1. Testing Configuration Loading...")
    try:
        config = Config()
        print(f"   ✅ Config loaded successfully")
        print(f"   📊 LLM Provider: {config.llm.provider}")
        print(f"   📊 Tournament ID: {config.metaculus.tournament_id}")
    except Exception as e:
        print(f"   ❌ Config loading failed: {e}")
        return False

    # Test 2: Tournament AskNews Client
    print("\n2. Testing Tournament AskNews Client...")
    try:
        asknews_client = TournamentAskNewsClient()
        stats = asknews_client.get_usage_stats()
        print(f"   ✅ AskNews client initialized")
        print(f"   📊 Quota usage: {stats['quota_usage_percentage']:.1f}%")
        print(f"   📊 Success rate: {stats['success_rate']:.1f}%")
    except Exception as e:
        print(f"   ❌ AskNews client failed: {e}")
        return False

    # Test 3: Metaculus Proxy Client
    print("\n3. Testing Metaculus Proxy Client...")
    try:
        proxy_client = MetaculusProxyClient(config)
        proxy_stats = proxy_client.get_usage_stats()
        print(f"   ✅ Proxy client initialized")
        print(f"   📊 Total requests: {proxy_stats['total_requests']}")
        print(f"   📊 Credits available: {proxy_client.proxy_credits_enabled}")
    except Exception as e:
        print(f"   ❌ Proxy client failed: {e}")
        return False

    # Test 4: Tournament Bot Integration
    print("\n4. Testing Tournament Bot Integration...")
    try:
        bot = MetaculusForecastingBot(config)
        print(f"   ✅ Tournament bot initialized")
        print(f"   📊 Pipeline ready: {bot.pipeline is not None}")
        print(f"   📊 LLM client ready: {bot.llm_client is not None}")
        print(f"   📊 Search client ready: {bot.search_client is not None}")
    except Exception as e:
        print(f"   ❌ Tournament bot failed: {e}")
        return False

    # Test 5: Sample Forecast
    print("\n5. Testing Sample Forecast...")
    try:
        result = await bot.forecast_question(12345, "chain_of_thought")
        print(f"   ✅ Sample forecast completed")
        print(f"   📊 Prediction: {result['forecast']['prediction']:.3f}")
        print(f"   📊 Confidence: {result['forecast']['confidence']:.3f}")
        print(f"   📊 Method: {result['forecast']['method']}")
    except Exception as e:
        print(f"   ❌ Sample forecast failed: {e}")
        return False

    # Test 6: Ensemble Forecast
    print("\n6. Testing Ensemble Forecast...")
    try:
        ensemble_result = await bot.forecast_question_ensemble(
            12346,
            ["chain_of_thought", "tree_of_thought"]
        )
        print(f"   ✅ Ensemble forecast completed")
        print(f"   📊 Ensemble prediction: {ensemble_result['ensemble_forecast']['prediction']:.3f}")
        print(f"   📊 Agents used: {len(ensemble_result['individual_forecasts'])}")
        print(f"   📊 Consensus strength: {ensemble_result['metadata']['consensus_strength']:.3f}")
    except Exception as e:
        print(f"   ❌ Ensemble forecast failed: {e}")
        return False

    # Test 7: Resource Usage Summary
    print("\n7. Resource Usage Summary...")
    try:
        final_asknews_stats = asknews_client.get_usage_stats()
        final_proxy_stats = proxy_client.get_usage_stats()

        print(f"   📊 AskNews Final Stats:")
        print(f"      - Total requests: {final_asknews_stats['total_requests']}")
        print(f"      - Success rate: {final_asknews_stats['success_rate']:.1f}%")
        print(f"      - Quota usage: {final_asknews_stats['quota_usage_percentage']:.1f}%")

        print(f"   📊 Proxy Final Stats:")
        print(f"      - Total requests: {final_proxy_stats['total_requests']}")
        print(f"      - Fallback rate: {final_proxy_stats['fallback_rate']:.1f}%")
        print(f"      - Credits used: {final_proxy_stats['estimated_credits_used']:.2f}")

    except Exception as e:
        print(f"   ⚠️  Resource summary warning: {e}")

    print("\n" + "=" * 50)
    print("🏆 TOURNAMENT INTEGRATION VALIDATION COMPLETE")
    print("✅ All critical components working correctly!")
    print("🚀 Bot ready for tournament domination!")

    return True


def main():
    """Main validation function."""
    print("Starting tournament integration validation...")

    # Check environment
    if not os.path.exists(".env"):
        print("⚠️  Warning: .env file not found. Some features may not work.")

    # Run validation
    success = asyncio.run(validate_tournament_integration())

    if success:
        print("\n🎉 VALIDATION SUCCESSFUL - Tournament bot is ready!")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED - Please check the errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
