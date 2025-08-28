# OpenRouter Configuration Guide

This document provides comprehensive guidance for configuring OpenRouter model availability detection and auto-configuration (Task 9.2).

## Overview

The OpenRouter configuration system provides:

- **Automatic model availability detection** on startup
- **Intelligent fallback chain configuration** based on available models
- **Health monitoring and auto-reconfiguration** during operation
- **Configuration validation and error reporting**
- **Provider routing optimization** for cost and performance

## Environment Variables

### Required Variables

```bash
# OpenRouter API Key (Required)
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### Recommended Variables

```bash
# OpenRouter Configuration
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_HTTP_REFERER=https://your-app-domain.com
OPENROUTER_APP_TITLE=Your App Name

# Model Configuration
DEFAULT_MODEL=openai/gpt-5
MINI_MODEL=openai/gpt-5-mini
NANO_MODEL=openai/gpt-5-nano

# Free Fallback Models
FREE_FALLBACK_MODELS=openai/gpt-oss-20b:free,moonshotai/kimi-k2:free
```

### Optional Variables

```bash
# Operation Mode Thresholds (Budget Utilization %)
NORMAL_MODE_THRESHOLD=70
CONSERVATIVE_MODE_THRESHOLD=85
EMERGENCY_MODE_THRESHOLD=95

# Metaculus Proxy Fallback
ENABLE_PROXY_CREDITS=true
```

## Model Availability Detection

The system automatically detects available OpenRouter models on startup:

### Tested Models

1. **Primary GPT-5 Models**:
   - `openai/gpt-5` - Full model for complex forecasting
   - `openai/gpt-5-mini` - Mini model for research synthesis
   - `openai/gpt-5-nano` - Nano model for validation tasks

2. **Free Fallback Models**:
   - `moonshotai/kimi-k2:free` - Free reasoning model
   - `openai/gpt-oss-20b:free` - Free OSS model

### Detection Process

```python
# Automatic detection on startup
router = await OpenRouterTriModelRouter.create_with_auto_configuration()

# Manual detection
availability = await router.detect_model_availability()
```

## Auto-Configuration Features

### Fallback Chain Optimization

The system automatically configures optimal fallback chains based on model availability:

```python
# Example auto-configured chains
{
    "full": [
        "openai/gpt-5",              # Primary
        "openai/gpt-5-mini",         # Downgrade
        "moonshotai/kimi-k2:free",   # Free fallback
        "metaculus/gpt-4o"           # Proxy fallback
    ],
    "mini": [
        "openai/gpt-5-mini",         # Primary
        "openai/gpt-5-nano",         # Downgrade
        "moonshotai/kimi-k2:free",   # Free fallback
        "metaculus/gpt-4o-mini"      # Proxy fallback
    ],
    "nano": [
        "openai/gpt-5-nano",         # Primary
        "openai/gpt-oss-20b:free",   # Free fallback
        "metaculus/gpt-4o-mini"      # Proxy fallback
    ]
}
```

### Provider Routing Configuration

The system configures OpenRouter provider preferences based on operation mode:

#### Normal Mode (0-70% budget)
```python
{
    "sort": "throughput",
    "order": ["fastest", "most_reliable", "cheapest"],
    "min_success_rate": 0.98,
    "prefer_streaming": True
}
```

#### Conservative Mode (70-85% budget)
```python
{
    "sort": "price",
    "max_price": {"prompt": 1.0, "completion": 2.0},
    "order": ["cheapest", "most_reliable", "fastest"],
    "min_success_rate": 0.95
}
```

#### Emergency Mode (85-95% budget)
```python
{
    "sort": "price",
    "max_price": {"prompt": 0.1, "completion": 0.1},
    "order": ["cheapest", "fastest"],
    "timeout": 30
}
```

#### Critical Mode (95-100% budget)
```python
{
    "sort": "price",
    "max_price": {"prompt": 0, "completion": 0},
    "order": ["free", "cheapest"],
    "require_parameters": True
}
```

## Health Monitoring

### Startup Health Check

```python
# Comprehensive startup validation
validation_success = await router.health_monitor_startup()

# Individual tier health checks
for tier in ["nano", "mini", "full"]:
    health_status = await router.check_model_health(tier)
    print(f"{tier}: {health_status.is_available}")
```

### Continuous Monitoring

```python
# Start continuous health monitoring (optional)
await router.continuous_health_monitoring(interval_seconds=300)
```

### Health Status Information

```python
# Get comprehensive status report
status_report = router.get_configuration_status_report()

# Check model status
for tier, status in status_report['model_status'].items():
    print(f"{tier}: {status['model_name']} - {'✓' if status['is_available'] else '✗'}")
```

## Configuration Validation

### Startup Validation

```python
from src.infrastructure.config.openrouter_startup_validator import OpenRouterStartupValidator

validator = OpenRouterStartupValidator()
validation_result = await validator.validate_configuration()

if validation_result.is_valid:
    print("✅ Configuration is valid")
else:
    print("❌ Configuration issues found:")
    for error in validation_result.errors:
        print(f"  • {error}")
```

### Validation Features

1. **API Key Validation**: Checks if OpenRouter API key is configured and valid
2. **Model Availability**: Tests connectivity to configured models
3. **Configuration Completeness**: Validates all required and recommended settings
4. **Provider Routing**: Verifies OpenRouter provider configuration
5. **Fallback Chain Validation**: Ensures fallback models are available

## Error Handling and Recovery

### Automatic Recovery

The system provides comprehensive error recovery:

1. **Model Failures**: Automatic fallback to next model in chain
2. **API Failures**: Retry with exponential backoff
3. **Budget Exhaustion**: Graceful degradation to free models
4. **Configuration Errors**: Detailed error reporting and suggestions

### Manual Recovery

```python
# Force reconfiguration
await router.auto_configure_fallback_chains()

# Reinitialize unhealthy models
for tier in ["nano", "mini", "full"]:
    if not router.model_status[tier].is_available:
        # Reinitialize this tier
        config = router.model_configs[tier]
        model, status = router._initialize_model_with_fallback(tier, config)
        router.models[tier] = model
        router.model_status[tier] = status
```

## Testing and Validation

### Simple Configuration Test

```bash
python3 simple_openrouter_test.py
```

### Comprehensive Validation Test

```bash
python3 test_openrouter_validation.py
```

### Manual Validation

```python
# Test individual components
from src.infrastructure.config.tri_model_router import OpenRouterTriModelRouter

# Create router with auto-configuration
router = await OpenRouterTriModelRouter.create_with_auto_configuration()

# Test model availability
availability = await router.detect_model_availability()
print(f"Available models: {sum(availability.values())}/{len(availability)}")

# Test health monitoring
health_success = await router.health_monitor_startup()
print(f"Health check: {'✓' if health_success else '✗'}")
```

## Troubleshooting

### Common Issues

1. **No Models Available**
   - Check OpenRouter API key validity
   - Verify network connectivity
   - Check account status and credits

2. **Configuration Errors**
   - Run `python3 simple_openrouter_test.py` to validate environment variables
   - Check `.env` file for missing or incorrect values
   - Verify OpenRouter base URL is correct

3. **Health Check Failures**
   - Check individual model availability
   - Verify API key permissions
   - Test network connectivity to OpenRouter

### Debug Information

```python
# Get detailed configuration status
status_report = router.get_configuration_status_report()

# Check environment variables
env_status = status_report['environment_variables']
for var, status in env_status.items():
    print(f"{var}: {status}")

# Check model status
model_status = status_report['model_status']
for tier, status in model_status.items():
    print(f"{tier}: {status['model_name']} - {status['is_available']}")
```

### Log Analysis

The system provides detailed logging for troubleshooting:

```
INFO - OpenRouter tri-model router initialized with actual available models and pricing
INFO - ✓ nano model initialized: openai/gpt-5-nano
INFO - ✓ mini model initialized: openai/gpt-5-mini
INFO - ✓ full model initialized: openai/gpt-5
INFO - Model availability check complete: 5/5 models available
INFO - All model tiers available - system fully operational
```

## Integration with Main Application

The OpenRouter configuration is automatically integrated into the main application:

```python
# In main.py - automatic startup validation
async def validate_openrouter_startup():
    validator = OpenRouterStartupValidator()
    validation_success = await validator.run_startup_validation(exit_on_failure=False)

    if validation_success:
        router = await OpenRouterTriModelRouter.create_with_auto_configuration()
        return router
    return None

# Run validation on startup
openrouter_router = asyncio.run(validate_openrouter_startup())
```

## Best Practices

1. **Always set attribution headers** for better OpenRouter ranking
2. **Use environment variables** for all configuration
3. **Test configuration** before production deployment
4. **Monitor health status** during operation
5. **Keep API keys secure** and rotate regularly
6. **Use free fallbacks** to extend operational capacity
7. **Configure operation mode thresholds** based on budget requirements

## Support

For issues with OpenRouter configuration:

1. Run the validation tests to identify specific problems
2. Check the generated setup guide for configuration recommendations
3. Review the logs for detailed error information
4. Verify OpenRouter account status and API key validity

The system is designed to be self-diagnosing and provides detailed feedback for any configuration issues.
