#!/usr/bin/env python3
"""
Test script for OpenRouter tri-model configuration.
Verifies that the OpenRouter integration is properly configured.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set up environment
os.chdir(project_root)

try:
    from src.infrastructure.config.tri_model_router import OpenRouterTriModelRouter
except ImportError:
    print("Import error - trying alternative import path")
    # Alternative import without relative imports
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tri_model_router",
        project_root / "src" / "infrastructure" / "config" / "tri_model_router.py",
    )
    tri_model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tri_model_module)
    OpenRouterTriModelRouter = tri_model_module.OpenRouterTriModelRouter


def test_openrouter_configuration():
    """Test OpenRouter configuration and model setup."""
    print("🔧 Testing OpenRouter Tri-Model Configuration")
    print("=" * 50)

    # Initialize router
    try:
        router = OpenRouterTriModelRouter()
        print("✓ OpenRouter tri-model router initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize router: {e}")
        return False

    # Test configuration
    print(f"✓ OpenRouter base URL: {router.openrouter_base_url}")
    print(f"✓ Attribution headers: {router.openrouter_headers}")

    # Test model configurations
    print("\n📊 Model Tier Configuration:")
    costs = router.get_model_costs()
    for tier, cost_info in costs.items():
        print(f"  {tier.upper()}: {cost_info['model_name']}")
        print(f"    Input: ${cost_info['input_cost_per_million']}/1M tokens")
        print(f"    Output: ${cost_info['output_cost_per_million']}/1M tokens")

    # Test provider routing info
    print("\n🔀 Provider Routing Configuration:")
    routing_info = router.get_openrouter_provider_routing_info()
    for mode, description in routing_info["operation_modes"].items():
        print(f"  {mode}: {description}")

    # Test fallback chains
    print("\n🔄 Fallback Chains:")
    for tier, chain in router.fallback_chains.items():
        print(f"  {tier.upper()}: {' → '.join(chain[:3])}...")

    # Test model status
    print("\n📈 Model Status:")
    status = router.get_model_status()
    for tier, status_info in status.items():
        print(f"  {tier.upper()}: {status_info}")

    # Test cost estimation
    print("\n💰 Cost Estimation Examples:")
    test_cases = [
        ("validation", 100, "minimal"),
        ("research", 500, "medium"),
        ("forecast", 1000, "high"),
    ]

    for task_type, content_length, complexity in test_cases:
        cost = router.get_cost_estimate(task_type, content_length, complexity, 75.0)
        print(f"  {task_type} ({complexity}, {content_length} chars): ${cost:.6f}")

    return True


async def test_model_availability():
    """Test model availability detection."""
    print("\n🔍 Testing Model Availability Detection")
    print("=" * 50)

    router = OpenRouterTriModelRouter()

    try:
        availability = await router.detect_model_availability()

        print("Model Availability Results:")
        for model, available in availability.items():
            status = "✓ Available" if available else "✗ Unavailable"
            print(f"  {model}: {status}")

        available_count = sum(availability.values())
        total_count = len(availability)
        print(f"\nSummary: {available_count}/{total_count} models available")

        return available_count > 0

    except Exception as e:
        print(f"✗ Model availability test failed: {e}")
        return False


def test_operation_modes():
    """Test operation mode switching."""
    print("\n⚙️ Testing Operation Mode Switching")
    print("=" * 50)

    router = OpenRouterTriModelRouter()

    test_budgets = [25.0, 60.0, 85.0, 98.0]  # Different budget levels

    for budget in test_budgets:
        mode = router.get_operation_mode(budget)
        print(f"  Budget {budget}% remaining → {mode} mode")

    return True


def main():
    """Run all tests."""
    print("🚀 OpenRouter Configuration Test Suite")
    print("=" * 60)

    # Check environment
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key or openrouter_key.startswith("dummy_"):
        print("⚠️  Warning: OPENROUTER_API_KEY not set or is dummy")
    else:
        print("✓ OPENROUTER_API_KEY is configured")

    # Run tests
    tests = [
        ("Configuration Test", test_openrouter_configuration),
        ("Operation Modes Test", test_operation_modes),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
            print(f"✓ {test_name} passed" if result else f"✗ {test_name} failed")
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            results.append(False)

    # Run async test
    print(f"\nModel Availability Test:")
    try:
        availability_result = asyncio.run(test_model_availability())
        results.append(availability_result)
        print(
            f"✓ Model Availability Test passed"
            if availability_result
            else f"✗ Model Availability Test failed"
        )
    except Exception as e:
        print(f"✗ Model Availability Test failed with error: {e}")
        results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! OpenRouter configuration is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
