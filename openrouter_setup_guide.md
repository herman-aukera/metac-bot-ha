# OpenRouter Configuration Setup Guide

## Current Status
Configuration Valid: ✗ NO
Errors: 1
Warnings: 0

## ❌ Critical Errors (Must Fix)

1. OpenRouter API connectivity test failed

## 💡 Recommendations

1. Check OpenRouter API key validity and network connectivity

## Environment Variable Setup

Add these to your .env file:

```bash
# Required
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Recommended
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_HTTP_REFERER=your_app_url_here
OPENROUTER_APP_TITLE=your_app_name_here

# Model Configuration
DEFAULT_MODEL=openai/gpt-5
MINI_MODEL=openai/gpt-5-mini
NANO_MODEL=openai/gpt-5-nano

# Free Fallback Models
FREE_FALLBACK_MODELS=openai/gpt-oss-20b:free,moonshotai/kimi-k2:free
```

## Next Steps

1. Set the required environment variables
2. Restart the application
3. Run validation again to confirm setup
