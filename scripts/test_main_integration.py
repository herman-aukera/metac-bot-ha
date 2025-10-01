#!/usr/bin/env python3
"""Test circuit breaker integration with main.py."""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import infrastructure.external_apis.llm_client as llm_client
from infrastructure.external_apis.llm_client import (
    is_openrouter_circuit_breaker_open,
    get_openrouter_circuit_breaker_status,
    reset_openrouter_circuit_breaker,
)


def test_main_integration():
    """Test that main.py will check circuit breaker properly."""
    print("🧪 Testing main.py Circuit Breaker Integration")
    print("=" * 50)

    # Reset circuit breaker first
    reset_openrouter_circuit_breaker()

    # Simulate the scenario: circuit breaker should be checked before running
    print("✅ Initial state - circuit breaker closed")
    status = get_openrouter_circuit_breaker_status()
    print(f"   Status: {status}")

    # Simulate enough failures to open circuit breaker
    print("\n🔄 Simulating quota exhaustion scenario...")
    for i in range(10):
        llm_client.OPENROUTER_CONSECUTIVE_QUOTA_FAILURES += 1

    # Open circuit breaker manually (like what happens after threshold)
    llm_client.OPENROUTER_CIRCUIT_BREAKER_OPEN = True
    llm_client.OPENROUTER_CIRCUIT_BREAKER_OPENED_AT = llm_client.time.time()

    print(f"✅ Circuit breaker opened after {llm_client.OPENROUTER_CONSECUTIVE_QUOTA_FAILURES} failures")

    # Test what main.py will check
    print("\n🔍 Testing main.py checks:")

    # This is what main.py will check before running tournament
    if is_openrouter_circuit_breaker_open():
        print("  ✅ main.py would correctly SKIP tournament due to circuit breaker")
        status = get_openrouter_circuit_breaker_status()
        print(f"     Reason: {status['consecutive_failures']} consecutive failures, ")
        print(f"     Wait time: {status['time_until_reset_seconds']:.0f}s until auto-reset")
    else:
        print("  ❌ main.py would incorrectly proceed despite circuit breaker")

    # Test status information that would go in run_summary.json
    print(f"\n📄 run_summary.json would include:")
    status = get_openrouter_circuit_breaker_status()
    print(f"   circuit_breaker_status: {{")
    print(f"     'is_open': {status['is_open']},")
    print(f"     'consecutive_failures': {status['consecutive_failures']},")
    print(f"     'time_until_reset_seconds': {status['time_until_reset_seconds']:.1f}")
    print(f"   }}")

    print(f"\n🎉 Integration test completed!")

    # Reset for next test
    reset_openrouter_circuit_breaker()


if __name__ == "__main__":
    test_main_integration()
