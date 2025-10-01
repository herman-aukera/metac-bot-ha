#!/usr/bin/env python3
"""Simulate circuit breaker behavior without making API calls."""

import sys
import time
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


def simulate_quota_failure():
    """Simulate a quota failure by incrementing the failure counter."""
    # Increment the failure counter directly on the module
    llm_client.OPENROUTER_CONSECUTIVE_QUOTA_FAILURES += 1
    print(f"  Simulated quota failure #{llm_client.OPENROUTER_CONSECUTIVE_QUOTA_FAILURES}")

    # Check if we should open the circuit breaker
    if llm_client.OPENROUTER_CONSECUTIVE_QUOTA_FAILURES >= llm_client.OPENROUTER_CIRCUIT_BREAKER_THRESHOLD:
        if not llm_client.OPENROUTER_CIRCUIT_BREAKER_OPEN:
            llm_client.OPENROUTER_CIRCUIT_BREAKER_OPEN = True
            llm_client.OPENROUTER_CIRCUIT_BREAKER_OPENED_AT = time.time()
            print(f"  🔴 Circuit breaker OPENED after {llm_client.OPENROUTER_CONSECUTIVE_QUOTA_FAILURES} failures!")
            return True
    return False
def main():
    """Test circuit breaker simulation."""
    print("🧪 Circuit Breaker Simulation Test")
    print("=" * 40)

    # Reset circuit breaker
    reset_openrouter_circuit_breaker()
    print("✅ Circuit breaker reset")

    # Show initial status
    status = get_openrouter_circuit_breaker_status()
    print(f"Initial status: {status}")

    print("\n🔄 Simulating quota failures...")

    # Simulate 12 quota failures
    for i in range(12):
        print(f"\nAttempt {i+1}/12:")

        # Check if circuit breaker is open before "making request"
        if is_openrouter_circuit_breaker_open():
            print("  🚫 Circuit breaker is OPEN - request blocked!")
            status = get_openrouter_circuit_breaker_status()
            print(f"     Time until reset: {status['time_until_reset_seconds']:.1f}s")
            continue

        # Simulate the failure
        simulate_quota_failure()

    print("\n🔍 Final circuit breaker status:")
    final_status = get_openrouter_circuit_breaker_status()
    print(f"  Is open: {final_status['is_open']}")
    print(f"  Consecutive failures: {final_status['consecutive_failures']}")
    print(f"  Failure threshold: {final_status['failure_threshold']}")
    print(f"  Timeout: {final_status['timeout_seconds']}s")
    print(f"  Time until reset: {final_status['time_until_reset_seconds']:.1f}s")

    # Test what happens if we try more requests
    print("\n🚫 Testing blocked requests after circuit breaker opens:")
    for i in range(3):
        if is_openrouter_circuit_breaker_open():
            print(f"  Request {i+1}: BLOCKED by circuit breaker ✅")
        else:
            print(f"  Request {i+1}: Would be allowed ❌")

    print("\n🔧 Testing manual reset:")
    reset_openrouter_circuit_breaker()
    reset_status = get_openrouter_circuit_breaker_status()
    if not reset_status['is_open']:
        print("✅ Manual reset successful - circuit breaker closed")
        print(f"   Consecutive failures reset to: {reset_status['consecutive_failures']}")
    else:
        print("❌ Manual reset failed")

    print("\n🎉 Circuit breaker simulation completed successfully!")


if __name__ == "__main__":
    main()
