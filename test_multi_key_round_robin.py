#!/usr/bin/env python3
"""Test script to verify multi-key round-robin strategy works correctly."""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from infrastructure.external_apis.llm_client import (
    LLMClient,
    OPENROUTER_KEYS,
    OPENROUTER_KEY_LIMITS,
    _initialize_openrouter_keys,
)
from infrastructure.config.settings import LLMConfig


async def test_multi_key_strategy():
    """Test that multiple OpenRouter keys work with round-robin."""

    print("=" * 70)
    print("MULTI-KEY ROUND-ROBIN STRATEGY TEST")
    print("=" * 70)
    print()

    # Initialize keys
    _initialize_openrouter_keys()

    print(f"📊 Initialized {len(OPENROUTER_KEYS)} keys:")
    for i, key in enumerate(OPENROUTER_KEYS):
        key_display = f"{key[:12]}...{key[-4:]}"
        limits = OPENROUTER_KEY_LIMITS[key]
        print(f"  KEY_{i+1} ({key_display}): {limits['limit']} req/day")

    total_capacity = sum(v['limit'] for v in OPENROUTER_KEY_LIMITS.values())
    print(f"\n✅ Total daily capacity: {total_capacity} requests")
    print("=" * 70)
    print()

    # Create LLM client
    config = LLMConfig()
    client = LLMClient(config)

    # Test multiple calls to see round-robin in action
    print("Testing round-robin with 3 calls:")
    print("-" * 70)

    for i in range(1, 4):
        try:
            print(f"\n🔄 Call #{i}:")

            response = await client._call_openrouter(
                prompt="Say OK",
                model="meta-llama/llama-3.2-3b-instruct:free",
                temperature=0.1,
                max_tokens=5
            )

            print(f"  ✅ Response: \"{response}\"")

            # Show current usage
            print(f"  📈 Current usage:")
            for j, key in enumerate(OPENROUTER_KEYS):
                limits = OPENROUTER_KEY_LIMITS[key]
                key_display = f"{key[:12]}...{key[-4:]}"
                pct = (limits['used'] / limits['limit'] * 100) if limits['limit'] > 0 else 0
                print(f"     KEY_{j+1} ({key_display}): {limits['used']}/{limits['limit']} ({pct:.0f}%)")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            break

    print()
    print("=" * 70)
    print("✅ Multi-key round-robin test complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_multi_key_strategy())
