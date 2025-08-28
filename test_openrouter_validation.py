#!/usr/bin/env python3
"""
Test script for OpenRouter model availability detection and auto-configuration.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path and handle imports carefully
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

try:
    from infrastructure.config.openrouter_startup_validator import OpenRouterStartupValidator
    from infrastructure.config.tri_model_router import OpenRouterTriModelRouter
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("This test requires the full project environment to be set up.")
    IMPORTS_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def test_openrouter_validation():
    """Test OpenRouter validation and auto-configuration."""
    print("="*60)
    print("OpenRouter Model Availability Detection and Auto-Configuration Test")
    print("="*60)

    # Test 1: Configuration Validation
    print("\n1. Testing Configuration Validation...")
    validator = OpenRouterStartupValidator()

    try:
        validation_result = await validator.validate_configuration()

        print(f"   Configuration Valid: {'✅ YES' if validation_result.is_valid else '❌ NO'}")
        print(f"   Errors: {len(validation_result.errors)}")
        print(f"   Warnings: {len(validation_result.warnings)}")

        if validation_result.errors:
            print("   Errors found:")
            for error in validation_result.errors:
                print(f"     • {error}")

        if validation_result.warnings:
            print("   Warnings:")
            for warning in validation_result.warnings[:3]:  # Show first 3
                print(f"     • {warning}")

    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        return False

    # Test 2: Model Availability Detection
    print("\n2. Testing Model Availability Detection...")

    try:
        router = OpenRouterTriModelRouter()
        availability = await router.detect_model_availability()

        print(f"   Models tested: {len(availability)}")
        available_count = sum(1 for available in availability.values() if available)
        print(f"   Available models: {available_count}/{len(availability)}")

        for model, available in availability.items():
            status = "✅" if available else "❌"
            print(f"     {status} {model}")

    except Exception as e:
        print(f"   ❌ Model availability detection failed: {e}")
        return False

    # Test 3: Auto-Configuration
    print("\n3. Testing Auto-Configuration...")

    try:
        router = await OpenRouterTriModelRouter.create_with_auto_configuration()

        # Get configuration status
        status_report = router.get_configuration_status_report()

        print(f"   Router initialized: ✅")
        print(f"   API key configured: {'✅' if status_report['router_info']['api_key_configured'] else '❌'}")

        # Check model status
        healthy_tiers = sum(1 for status in status_report['model_status'].values()
                          if status['is_available'])
        print(f"   Healthy model tiers: {healthy_tiers}/3")

        for tier, status in status_report['model_status'].items():
            tier_status = "✅" if status['is_available'] else "❌"
            print(f"     {tier_status} {tier.upper()}: {status['model_name']}")

    except Exception as e:
        print(f"   ❌ Auto-configuration failed: {e}")
        return False

    # Test 4: Fallback Chain Configuration
    print("\n4. Testing Fallback Chain Configuration...")

    try:
        optimized_chains = await router.auto_configure_fallback_chains()

        print(f"   Fallback chains configured: ✅")
        for tier, chain in optimized_chains.items():
            print(f"     {tier.upper()}: {' → '.join(chain[:3])}{'...' if len(chain) > 3 else ''}")

    except Exception as e:
        print(f"   ❌ Fallback chain configuration failed: {e}")
        return False

    # Test 5: Health Monitoring
    print("\n5. Testing Health Monitoring...")

    try:
        health_success = await router.health_monitor_startup()

        print(f"   Health monitoring: {'✅' if health_success else '⚠️'}")

        # Test individual tier health
        for tier in ["nano", "mini", "full"]:
            health_status = await router.check_model_health(tier)
            tier_health = "✅" if health_status.is_available else "❌"
            response_time = f" ({health_status.response_time:.2f}s)" if health_status.response_time else ""
            print(f"     {tier_health} {tier.upper()}{response_time}")

    except Exception as e:
        print(f"   ❌ Health monitoring failed: {e}")
        return False

    print("\n" + "="*60)
    print("✅ All tests completed successfully!")
    print("="*60)

    return True


async def test_environment_variables():
    """Test environment variable configuration."""
    print("\n📋 Environment Variable Status:")

    required_vars = [
        "OPENROUTER_API_KEY"
    ]

    recommended_vars = [
        "OPENROUTER_BASE_URL",
        "OPENROUTER_HTTP_REFERER",
        "OPENROUTER_APP_TITLE",
        "DEFAULT_MODEL",
        "MINI_MODEL",
        "NANO_MODEL"
    ]

    print("\n   Required:")
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            print(f"     ❌ {var}: Not set")
        elif value.startswith("dummy_"):
            print(f"     ⚠️  {var}: Dummy value")
        else:
            masked = value[:8] + "*" * (len(value) - 8) if len(value) > 8 else "*****"
            print(f"     ✅ {var}: {masked}")

    print("\n   Recommended:")
    for var in recommended_vars:
        value = os.getenv(var)
        if not value:
            print(f"     ⚠️  {var}: Not set")
        else:
            print(f"     ✅ {var}: {value}")


async def main():
    """Main test function."""
    print("Starting OpenRouter validation tests...\n")

    if not IMPORTS_AVAILABLE:
        print("❌ Cannot run tests - imports not available")
        print("This test requires the full project environment.")
        return 1

    # Test environment variables first
    await test_environment_variables()

    # Run main validation tests
    success = await test_openrouter_validation()

    if success:
        print("\n🎉 All OpenRouter validation tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
