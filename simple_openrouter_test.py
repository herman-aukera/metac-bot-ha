#!/usr/bin/env python3
"""
Simple test for OpenRouter environment variable configuration.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment_variables():
    """Test OpenRouter environment variable configuration."""
    print("="*60)
    print("OpenRouter Environment Variable Configuration Test")
    print("="*60)

    required_vars = {
        "OPENROUTER_API_KEY": "Required for OpenRouter API access"
    }

    recommended_vars = {
        "OPENROUTER_BASE_URL": "OpenRouter API base URL (default: https://openrouter.ai/api/v1)",
        "OPENROUTER_HTTP_REFERER": "HTTP referer for attribution and ranking",
        "OPENROUTER_APP_TITLE": "Application title for attribution",
        "DEFAULT_MODEL": "Primary model for forecasting (default: openai/gpt-5)",
        "MINI_MODEL": "Mini model for research (default: openai/gpt-5-mini)",
        "NANO_MODEL": "Nano model for validation (default: openai/gpt-5-nano)"
    }

    optional_vars = {
        "FREE_FALLBACK_MODELS": "Free fallback models (default: openai/gpt-oss-20b:free,moonshotai/kimi-k2:free)",
        "ENABLE_PROXY_CREDITS": "Enable Metaculus proxy credits (default: true)"
    }

    print("\n📋 Required Environment Variables:")
    all_good = True

    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            print(f"   ❌ {var}: Not set")
            print(f"      {description}")
            all_good = False
        elif value.startswith("dummy_"):
            print(f"   ⚠️  {var}: Dummy value detected")
            print(f"      {description}")
            print(f"      Current: {value}")
            all_good = False
        else:
            masked = value[:8] + "*" * (len(value) - 8) if len(value) > 8 else "*****"
            print(f"   ✅ {var}: {masked}")
            print(f"      {description}")

    print("\n📋 Recommended Environment Variables:")

    for var, description in recommended_vars.items():
        value = os.getenv(var)
        if not value:
            print(f"   ⚠️  {var}: Not set (will use default)")
            print(f"      {description}")
        else:
            print(f"   ✅ {var}: {value}")
            print(f"      {description}")

    print("\n📋 Optional Environment Variables:")

    for var, description in optional_vars.items():
        value = os.getenv(var)
        if not value:
            print(f"   ℹ️  {var}: Not set (will use default)")
            print(f"      {description}")
        else:
            print(f"   ✅ {var}: {value}")
            print(f"      {description}")

    # Check for common configuration issues
    print("\n🔍 Configuration Analysis:")

    # Check base URL
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if base_url != "https://openrouter.ai/api/v1":
        print(f"   ⚠️  Non-standard base URL: {base_url}")
        print(f"      Recommended: https://openrouter.ai/api/v1")
    else:
        print(f"   ✅ Base URL is correct: {base_url}")

    # Check model configurations
    models = {
        "DEFAULT_MODEL": os.getenv("DEFAULT_MODEL", "openai/gpt-5"),
        "MINI_MODEL": os.getenv("MINI_MODEL", "openai/gpt-5-mini"),
        "NANO_MODEL": os.getenv("NANO_MODEL", "openai/gpt-5-nano")
    }

    expected_models = {
        "DEFAULT_MODEL": "openai/gpt-5",
        "MINI_MODEL": "openai/gpt-5-mini",
        "NANO_MODEL": "openai/gpt-5-nano"
    }

    for var, current in models.items():
        expected = expected_models[var]
        if current == expected:
            print(f"   ✅ {var}: {current} (standard)")
        else:
            print(f"   ℹ️  {var}: {current} (custom, expected: {expected})")

    # Check attribution headers
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_APP_TITLE")

    if not referer and not title:
        print(f"   ⚠️  No attribution headers set")
        print(f"      Consider setting OPENROUTER_HTTP_REFERER and OPENROUTER_APP_TITLE")
        print(f"      This helps with OpenRouter ranking and attribution")
    elif referer and title:
        print(f"   ✅ Attribution headers configured")
    else:
        print(f"   ℹ️  Partial attribution headers configured")

    print("\n" + "="*60)

    if all_good:
        print("✅ All required environment variables are properly configured!")
        print("🚀 OpenRouter should work correctly with this configuration.")
    else:
        print("❌ Some required environment variables need attention.")
        print("📝 Please check the issues above and update your .env file.")

    print("="*60)

    return all_good


def generate_env_template():
    """Generate a template .env configuration."""
    template = """
# OpenRouter Configuration Template
# Copy these lines to your .env file and fill in your values

# Required - Get your API key from https://openrouter.ai/
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Recommended - OpenRouter configuration
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_HTTP_REFERER=https://your-app-domain.com
OPENROUTER_APP_TITLE=Your App Name

# Model Configuration - GPT-5 variants for cost optimization
DEFAULT_MODEL=openai/gpt-5
MINI_MODEL=openai/gpt-5-mini
NANO_MODEL=openai/gpt-5-nano

# Free Fallback Models for budget exhaustion
FREE_FALLBACK_MODELS=openai/gpt-oss-20b:free,moonshotai/kimi-k2:free

# Optional - Metaculus proxy fallback
ENABLE_PROXY_CREDITS=true

# Budget Management
BUDGET_LIMIT=100.0
EMERGENCY_MODE_THRESHOLD=0.95
CONSERVATIVE_MODE_THRESHOLD=0.80
"""

    print("\n📝 Environment Variable Template:")
    print("="*60)
    print(template)
    print("="*60)

    try:
        with open("openrouter_env_template.txt", "w") as f:
            f.write(template)
        print("💾 Template saved to: openrouter_env_template.txt")
    except Exception as e:
        print(f"⚠️ Could not save template file: {e}")


def main():
    """Main test function."""
    print("OpenRouter Configuration Test\n")

    success = test_environment_variables()

    if not success:
        print("\n" + "="*60)
        print("Need help with configuration? Here's a template:")
        generate_env_template()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
