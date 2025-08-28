#!/usr/bin/env python3
"""
Test OpenRouter configuration and model availability.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_openrouter_configuration():
    """Test OpenRouter configuration and model availability."""
    print("="*60)
    print("OpenRouter Configuration and Model Availability Test")
    print("="*60)

    try:
        from infrastructure.config.openrouter_startup_validator import OpenRouterStartupValidator
        from infrastructure.config.tri_model_router import OpenRouterTriModelRouter

        # Test 1: Configuration Validation
        print("\n1. Testing Configuration Validation...")
        validator = OpenRouterStartupValidator()
        validation_result = await validator.validate_configuration()

        print(f"   Configuration Valid: {'✅' if validation_result.is_valid else '❌'}")
        print(f"   Errors: {len(validation_result.errors)}")
        print(f"   Warnings: {len(validation_result.warnings)}")

        if validation_result.errors:
            print("   Critical Errors:")
            for error in validation_result.errors:
                print(f"     • {error}")

        if validation_result.warnings:
            print("   Warnings:")
            for warning in validation_result.warnings[:3]:
                print(f"     • {warning}")

        # Test 2: Model Availability Detection
        print("\n2. Testing Model Availability Detection...")
        router = OpenRouterTriModelRouter()
        availability = await router.detect_model_availability()

        available_count = sum(1 for available in availability.values() if available)
        print(f"   Models tested: {len(availability)}")
        print(f"   Available models: {available_count}/{len(availability)}")

        for model, available in availability.items():
            status = "✅" if available else "❌"
            print(f"     {status} {model}")

        # Test 3: Auto-Configuration
        print("\n3. Testing Auto-Configuration...")
        router = await OpenRouterTriModelRouter.create_with_auto_configuration()

        # Get status report
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

        # Test 4: Health Monitoring
        print("\n4. Testing Health Monitoring...")
        health_success = await router.health_monitor_startup()
        print(f"   Health monitoring: {'✅' if health_success else '⚠️'}")

        print("\n" + "="*60)

        if validation_result.is_valid and available_count > 0 and health_success:
            print("✅ All OpenRouter configuration tests passed!")
            print("🚀 OpenRouter is properly configured and ready for use.")
            return True
        else:
            print("⚠️ Some configuration issues detected.")
            print("📝 Check the output above for details.")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This indicates the OpenRouter modules are not properly set up.")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

async def main():
    """Main test function."""
    success = await test_openrouter_configuration()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
