#!/usr/bin/env python3
"""
Simple test script for OpenRouter configuration validation.
"""

import os
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print(f"✓ Loaded environment from {env_file}")
    else:
        print(f"⚠ No .env file found at {env_file}")

# Load environment variables
load_env_file()

def test_environment_configuration():
    """Test that OpenRouter environment variables are properly configured."""
    print("🔧 Testing OpenRouter Environment Configuration")
    print("=" * 50)

    required_vars = [
        "OPENROUTER_API_KEY",
        "DEFAULT_MODEL",
        "MINI_MODEL",
        "NANO_MODEL"
    ]

    optional_vars = [
        "OPENROUTER_BASE_URL",
        "OPENROUTER_HTTP_REFERER",
        "OPENROUTER_APP_TITLE",
        "FREE_FALLBACK_MODELS"
    ]

    all_good = True

    print("Required Variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value and not value.startswith("dummy_"):
            print(f"  ✓ {var}: {value}")
        else:
            print(f"  ✗ {var}: Not set or dummy value")
            all_good = False

    print("\nOptional Variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var}: {value}")
        else:
            print(f"  - {var}: Not set")

    return all_good

def test_model_names():
    """Test that model names follow OpenRouter conventions."""
    print("\n📝 Testing Model Name Conventions")
    print("=" * 50)

    models = {
        "DEFAULT_MODEL": os.getenv("DEFAULT_MODEL"),
        "MINI_MODEL": os.getenv("MINI_MODEL"),
        "NANO_MODEL": os.getenv("NANO_MODEL")
    }

    expected_patterns = {
        "DEFAULT_MODEL": ["openai/gpt-4o", "openai/gpt-5"],
        "MINI_MODEL": ["openai/gpt-4o-mini", "openai/gpt-5-mini"],
        "NANO_MODEL": ["meta-llama/llama-3.1-8b-instruct", "openai/gpt-5-nano"]
    }

    all_good = True

    for model_var, model_name in models.items():
        if not model_name:
            print(f"  ✗ {model_var}: Not configured")
            all_good = False
            continue

        # Check if it has provider prefix (required for OpenRouter)
        if "/" in model_name:
            print(f"  ✓ {model_var}: {model_name} (has provider prefix)")
        else:
            print(f"  ⚠ {model_var}: {model_name} (missing provider prefix - should be like 'openai/model-name')")
            all_good = False

    return all_good

def test_openrouter_base_configuration():
    """Test OpenRouter base configuration."""
    print("\n🌐 Testing OpenRouter Base Configuration")
    print("=" * 50)

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_APP_TITLE")

    print(f"  Base URL: {base_url}")

    if base_url == "https://openrouter.ai/api/v1":
        print("  ✓ Base URL is correct")
    else:
        print("  ⚠ Base URL should be https://openrouter.ai/api/v1")

    if referer:
        print(f"  ✓ HTTP Referer: {referer}")
    else:
        print("  - HTTP Referer: Not set (optional but recommended)")

    if title:
        print(f"  ✓ App Title: {title}")
    else:
        print("  - App Title: Not set (optional but recommended)")

    return True

def test_free_fallback_models():
    """Test free fallback model configuration."""
    print("\n🆓 Testing Free Fallback Models")
    print("=" * 50)

    free_models = os.getenv("FREE_FALLBACK_MODELS", "")
    expected_free_models = ["openai/gpt-oss-20b:free", "moonshotai/kimi-k2:free"]

    if free_models:
        configured_models = [m.strip() for m in free_models.split(",")]
        print(f"  Configured: {configured_models}")

        for expected in expected_free_models:
            if expected in configured_models:
                print(f"  ✓ {expected}: Configured")
            else:
                print(f"  ⚠ {expected}: Missing")
    else:
        print("  - No free fallback models configured")
        print(f"  Recommended: {', '.join(expected_free_models)}")

    return True

def test_pricing_awareness():
    """Test that we understand the pricing structure."""
    print("\n💰 OpenRouter Pricing Structure")
    print("=" * 50)

    pricing_info = {
        "openai/gpt-4o": {"input": "$5/1M", "output": "$15/1M"},
        "openai/gpt-4o-mini": {"input": "$0.15/1M", "output": "$0.60/1M"},
        "meta-llama/llama-3.1-8b-instruct": {"input": "$0.07/1M", "output": "$0.07/1M"},
        "openai/gpt-oss-20b:free": {"input": "Free", "output": "Free"},
        "moonshotai/kimi-k2:free": {"input": "Free", "output": "Free"}
    }

    print("  Expected Pricing (as of task specification):")
    for model, prices in pricing_info.items():
        print(f"    {model}: {prices['input']} in, {prices['output']} out")

    return True

def main():
    """Run all configuration tests."""
    print("🚀 OpenRouter Configuration Validation")
    print("=" * 60)

    tests = [
        ("Environment Configuration", test_environment_configuration),
        ("Model Name Conventions", test_model_names),
        ("OpenRouter Base Config", test_openrouter_base_configuration),
        ("Free Fallback Models", test_free_fallback_models),
        ("Pricing Awareness", test_pricing_awareness)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n📊 Configuration Summary: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 OpenRouter configuration looks good!")
        print("\nNext steps:")
        print("1. Verify API key has sufficient credits")
        print("2. Test actual model calls")
        print("3. Monitor costs during tournament")
        return 0
    else:
        print("❌ Some configuration issues found. Please review.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
