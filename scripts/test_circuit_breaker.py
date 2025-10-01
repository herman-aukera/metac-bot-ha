#!/usr/bin/env python3
"""Test script for OpenRouter circuit breaker functionality."""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from infrastructure.external_apis.llm_client import (
    LLMClient,
    is_openrouter_circuit_breaker_open,
    get_openrouter_circuit_breaker_status,
    reset_openrouter_circuit_breaker,
)
from infrastructure.config.settings import LLMConfig


async def test_circuit_breaker():
    """Test OpenRouter circuit breaker functionality."""
    print("🧪 Testing OpenRouter Circuit Breaker")
    print("=" * 50)

    # Create LLM client with invalid API key to trigger 403 errors
    config = LLMConfig(
        provider="openrouter",
        model="openai/gpt-5-nano",
        api_key="invalid-key-to-trigger-403",
        temperature=0.1,
    )

    client = LLMClient(config)

    # Reset circuit breaker to start clean
    reset_openrouter_circuit_breaker()
    print("✅ Circuit breaker reset")

    # Check initial state
    status = get_openrouter_circuit_breaker_status()
    print(f"Initial status: {status}")

    print("\n🔄 Testing quota failures...")

    # Generate enough failures to trigger circuit breaker
    for i in range(12):  # More than threshold (10)
        try:
            print(f"Attempt {i+1}/12: Making request that should fail...")
            await client.generate("test prompt", max_tokens=10)
            print("  ❌ Unexpected success!")
        except Exception as e:
            error_msg = str(e)
            print(f"  ✅ Expected failure: {error_msg[:100]}...")

            # Check if circuit breaker opened
            if is_openrouter_circuit_breaker_open():
                print(f"  🔴 Circuit breaker OPENED after {i+1} failures!")
                status = get_openrouter_circuit_breaker_status()
                print(f"  Status: {status}")
                break

        # Small delay between attempts
        await asyncio.sleep(0.5)

    print("\n🔍 Final circuit breaker status:")
    final_status = get_openrouter_circuit_breaker_status()
    print(f"  Is open: {final_status['is_open']}")
    print(f"  Consecutive failures: {final_status['consecutive_failures']}")
    print(f"  Failure threshold: {final_status['failure_threshold']}")
    print(f"  Time until reset: {final_status['time_until_reset_seconds']:.1f}s")

    if final_status['is_open']:
        print("✅ Circuit breaker working correctly - opened after quota failures")
    else:
        print("❌ Circuit breaker did not open as expected")

    print("\n🔧 Testing manual reset...")
    reset_openrouter_circuit_breaker()
    reset_status = get_openrouter_circuit_breaker_status()
    if not reset_status['is_open']:
        print("✅ Manual reset working correctly")
    else:
        print("❌ Manual reset failed")

    await client.client.aclose()


def test_circuit_breaker_functions():
    """Test circuit breaker utility functions."""
    print("\n🔧 Testing circuit breaker utility functions:")

    # Test status function
    status = get_openrouter_circuit_breaker_status()
    print(f"  Status function: {status}")

    # Test is_open function
    is_open = is_openrouter_circuit_breaker_open()
    print(f"  Is open function: {is_open}")

    print("✅ Utility functions working")


if __name__ == "__main__":
    # Test utility functions first
    test_circuit_breaker_functions()

    # Only test async functionality if we have a network connection
    if len(sys.argv) > 1 and sys.argv[1] == "--with-network":
        print("\n🌐 Running network tests...")
        asyncio.run(test_circuit_breaker())
    else:
        print("\n⚠️  Skipping network tests (use --with-network to enable)")
        print("   This avoids making actual API calls during testing")

    print("\n🎉 Circuit breaker tests completed!")
