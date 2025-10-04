#!/usr/bin/env python3
"""
Reset OpenRouter circuit breaker to allow testing.
Run this before test_single_forecast.py if circuit breaker is open.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from infrastructure.external_apis.llm_client import (
    reset_openrouter_circuit_breaker,
    get_openrouter_circuit_breaker_status,
)

print("=" * 80)
print("OpenRouter Circuit Breaker Status Check")
print("=" * 80)
print()

status = get_openrouter_circuit_breaker_status()
print("Current Status:")
print(f"  is_open: {status['is_open']}")
print(f"  consecutive_failures: {status['consecutive_failures']}")
print(f"  failure_threshold: {status['failure_threshold']}")
print(f"  timeout_seconds: {status['timeout_seconds']}")
if status['is_open']:
    print(f"  ⚠️  time_until_reset: {status['time_until_reset_seconds']:.1f}s")
print()

if status['is_open']:
    print("Circuit breaker is OPEN. Resetting...")
    reset_openrouter_circuit_breaker()
    print("✅ Circuit breaker manually reset")
    print()

    # Verify
    new_status = get_openrouter_circuit_breaker_status()
    print("New Status:")
    print(f"  is_open: {new_status['is_open']}")
    print(f"  consecutive_failures: {new_status['consecutive_failures']}")
else:
    print("✅ Circuit breaker is already closed (not blocking)")

print()
print("=" * 80)
