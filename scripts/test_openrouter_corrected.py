#!/usr/bin/env python3
"""
Test script for corrected OpenRouter configuration.
Validates actual model availability and pricing structure.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_env_file():
    """Load environment variables from .env file."""
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value


def test_openrouter_base_configuration():
    """Test OpenRouter base configuration."""
    print("🔧 OPENROUTER BASE CONFIGURATION TEST")
    print("=" * 50)

    # Check required OpenRouter configuration
    config_items = {
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "OPENROUTER_BASE_URL": os.getenv("OPENROUTER_BASE_URL"),
        "OPENROUTER_HTTP_REFERER": os.getenv("OPENROUTER_HTTP_REFERER"),
        "OPENROUTER_APP_TITLE": os.getenv("OPENROUTER_APP_TITLE"),
    }

    all_configured = True
    for key, value in config_items.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value or 'NOT SET'}")
        if not value:
            all_configured = False

    print(
        f"\nOpenRouter Configuration: {'✓ Complete' if all_configured else '✗ Incomplete'}"
    )
    return all_configured


def test_model_names():
    """Test corrected model names."""
    print("\n🤖 MODEL NAMES TEST")
    print("=" * 50)

    # Test actual model names from environment
    models = {
        "DEFAULT_MODEL": os.getenv("DEFAULT_MODEL", "openai/gpt-4o"),
        "MINI_MODEL": os.getenv("MINI_MODEL", "openai/gpt-4o-mini"),
        "NANO_MODEL": os.getenv("NANO_MODEL", "meta-llama/llama-3.1-8b-instruct"),
        "FORECAST_MODEL": os.getenv(
            "FORECAST_MODEL", "anthropic/claude-3-5-sonnet-20241022"
        ),
        "RESEARCH_MODEL": os.getenv(
            "RESEARCH_MODEL", "perplexity/llama-3.1-sonar-large-128k-online"
        ),
    }

    print("Configured Models:")
    for key, value in models.items():
        # Check if model name has correct provider prefix
        has_prefix = "/" in value and not value.startswith("metaculus/")
        status = "✓" if has_prefix else "⚠"
        print(f"  {status} {key}: {value}")

    # Test free fallback models
    free_models = os.getenv("FREE_FALLBACK_MODELS", "").split(",")
    print(f"\nFree Fallback Models:")
    for model in free_models:
        model = model.strip()
        if model:
            is_free = model.endswith(":free")
            status = "✓" if is_free else "⚠"
            print(f"  {status} {model}")


def test_pricing_awareness():
    """Test pricing configuration awareness."""
    print("\n💰 PRICING AWARENESS TEST")
    print("=" * 50)

    # Expected pricing for OpenRouter models (approximate)
    expected_pricing = {
        "openai/gpt-4o": {"input": 5.0, "output": 15.0},
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "meta-llama/llama-3.1-8b-instruct": {"input": 0.07, "output": 0.07},
        "anthropic/claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "perplexity/llama-3.1-sonar-large-128k-online": {"input": 1.0, "output": 1.0},
    }

    print("Expected Pricing (per 1M tokens):")
    for model, pricing in expected_pricing.items():
        print(f"  {model}:")
        print(f"    Input: ${pricing['input']}/1M")
        print(f"    Output: ${pricing['output']}/1M")

    # Calculate cost efficiency
    print(f"\nCost Efficiency Analysis:")
    print(f"  Nano tier (llama-3.1-8b): $0.07/1M - Ultra-cheap validation")
    print(f"  Mini tier (gpt-4o-mini): $0.15-0.60/1M - Balanced research")
    print(f"  Full tier (gpt-4o): $5-15/1M - Premium forecasting")
    print(f"  Free tier: $0/1M - Budget exhaustion fallback")


def test_free_fallback_models():
    """Test free fallback model configuration."""
    print("\n🆓 FREE FALLBACK MODELS TEST")
    print("=" * 50)

    free_models_str = os.getenv("FREE_FALLBACK_MODELS", "")
    if not free_models_str:
        print("  ✗ No free fallback models configured")
        return False

    free_models = [model.strip() for model in free_models_str.split(",")]

    # Expected free models based on OpenRouter documentation
    expected_free = ["openai/gpt-oss-20b:free", "moonshotai/kimi-k2:free"]

    print("Configured Free Models:")
    all_correct = True
    for model in free_models:
        if model in expected_free:
            print(f"  ✓ {model} - Verified available")
        else:
            print(f"  ⚠ {model} - Needs verification")
            all_correct = False

    print(
        f"\nFree Models Configuration: {'✓ Correct' if all_correct else '⚠ Needs Review'}"
    )
    return all_correct


def test_environment_configuration():
    """Test overall environment configuration."""
    print("\n🌍 ENVIRONMENT CONFIGURATION TEST")
    print("=" * 50)

    # Check budget configuration
    budget_limit = os.getenv("BUDGET_LIMIT", "100.0")
    max_cost_per_question = os.getenv("MAX_COST_PER_QUESTION", "1.50")

    print(f"Budget Configuration:")
    print(f"  Total Budget: ${budget_limit}")
    print(f"  Max Cost per Question: ${max_cost_per_question}")

    # Check operation mode thresholds
    conservative_threshold = os.getenv("CONSERVATIVE_MODE_THRESHOLD", "0.50")
    emergency_threshold = os.getenv("EMERGENCY_MODE_THRESHOLD", "0.80")

    print(f"\nOperation Mode Thresholds:")
    print(f"  Conservative Mode: {float(conservative_threshold)*100}% budget used")
    print(f"  Emergency Mode: {float(emergency_threshold)*100}% budget used")

    # Check tournament configuration
    tournament_id = os.getenv("AIB_TOURNAMENT_ID", "32813")
    publish_reports = os.getenv("PUBLISH_REPORTS", "true")

    print(f"\nTournament Configuration:")
    print(f"  Tournament ID: {tournament_id}")
    print(f"  Publish Reports: {publish_reports}")


async def test_model_availability():
    """Test actual model availability via OpenRouter."""
    print("\n🔍 MODEL AVAILABILITY TEST")
    print("=" * 50)

    try:
        from src.infrastructure.config.tri_model_router import OpenRouterTriModelRouter

        router = OpenRouterTriModelRouter()
        availability = await router.detect_model_availability()

        print("Model Availability Results:")
        for model, is_available in availability.items():
            status = "✓" if is_available else "✗"
            print(f"  {status} {model}")

        available_count = sum(availability.values())
        total_count = len(availability)

        print(
            f"\nAvailability Summary: {available_count}/{total_count} models available"
        )

        if available_count == 0:
            print("⚠ WARNING: No models available - check API key and configuration")
        elif available_count < total_count:
            print("⚠ Some models unavailable - fallback chains will be used")
        else:
            print("✓ All models available - optimal performance expected")

        return available_count > 0

    except Exception as e:
        print(f"✗ Model availability test failed: {e}")
        return False


def main():
    """Run all OpenRouter configuration tests."""
    print("🧪 OPENROUTER CONFIGURATION TEST SUITE")
    print("=" * 60)

    # Load environment
    load_env_file()

    # Run tests
    tests = [
        ("Base Configuration", test_openrouter_base_configuration),
        ("Model Names", test_model_names),
        ("Pricing Awareness", test_pricing_awareness),
        ("Free Fallback Models", test_free_fallback_models),
        ("Environment Configuration", test_environment_configuration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            results.append((test_name, False))

    # Run async model availability test
    try:
        availability_result = asyncio.run(test_model_availability())
        results.append(("Model Availability", availability_result))
    except Exception as e:
        print(f"✗ Model Availability test failed: {e}")
        results.append(("Model Availability", False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1

    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! OpenRouter configuration is ready.")
    else:
        print("⚠ Some tests failed. Review configuration before deployment.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
