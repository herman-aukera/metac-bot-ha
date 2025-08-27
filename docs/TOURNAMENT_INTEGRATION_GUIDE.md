# Tournament Integration Guide

## 🏆 Overview

This guide covers the complete tournament integration for the Metaculus Forecasting Bot HA. The bot is now fully optimized for tournament competition with advanced features and fallback mechanisms.

## 🚀 Quick Start

### Basic Usage

```bash
# Run tournament forecasting
python3 -m src.main --tournament 32813 --max-questions 10 --dry-run

# Run with verbose logging
python3 -m src.main --tournament 32813 --max-questions 5 --verbose

# Run validation
python3 scripts/validate_tournament_integration.py

# Run comprehensive tests
python3 scripts/test_tournament_features.py
```

### Configuration

The bot automatically loads configuration from:
1. Environment variables (`.env` file)
2. Configuration files (if provided)
3. Default settings

Key environment variables:
```bash
# AskNews API (Tournament Optimization)
ASKNEWS_CLIENT_ID=your_client_id
ASKNEWS_SECRET=your_secret

# OpenRouter API (LLM Provider)
OPENROUTER_API_KEY=your_api_key

# Tournament Settings
AIB_TOURNAMENT_ID=32813
MAX_CONCURRENT_QUESTIONS=5
```

## 🏆 Tournament Features

### 1. Tournament-Optimized Research

**TournamentAskNewsClient**
- Primary: AskNews API with quota management
- Fallback 1: Perplexity via OpenRouter
- Fallback 2: Exa search via OpenRouter
- Fallback 3: OpenRouter Perplexity
- Final fallback: Basic reasoning without external research

**Features:**
- Intelligent quota management
- Usage statistics tracking
- Automatic fallback chain
- Error handling and recovery

### 2. Metaculus Proxy Integration

**MetaculusProxyClient**
- Free credits from Metaculus proxy
- Automatic fallback to OpenRouter
- Cost optimization
- Usage tracking

**Supported Models:**
- `metaculus/claude-3-5-sonnet`
- `metaculus/gpt-4o`
- `metaculus/gpt-4o-mini`

### 3. Multi-Agent Ensemble

**Agent Types:**
- `chain_of_thought`: Step-by-step reasoning
- `tree_of_thought`: Branching analysis
- `react`: Reasoning and acting
- `ensemble`: Combination of multiple agents

**Ensemble Features:**
- Weighted aggregation by confidence and research quality
- Consensus strength calculation
- Individual agent tracking
- Tournament-specific optimizations

### 4. Advanced Pipeline

**ForecastingPipeline**
- Tournament-optimized research
- Multi-agent prediction
- Confidence calibration
- Error handling and recovery
- Resource usage optimization

## 📊 Usage Examples

### Single Question Forecast

```python
from src.main import MetaculusForecastingBot
from src.infrastructure.config.settings import Config

# Initialize bot
config = Config()
bot = MetaculusForecastingBot(config)

# Forecast single question
result = await bot.forecast_question(12345, "chain_of_thought")
print(f"Prediction: {result['forecast']['prediction']:.3f}")
print(f"Confidence: {result['forecast']['confidence']:.3f}")
```

### Ensemble Forecast

```python
# Ensemble forecast with multiple agents
agents = ["chain_of_thought", "tree_of_thought", "react"]
result = await bot.forecast_question_ensemble(12345, agents)

print(f"Ensemble prediction: {result['ensemble_forecast']['prediction']:.3f}")
print(f"Consensus strength: {result['metadata']['consensus_strength']:.3f}")
```

### Tournament Mode

```python
# Run full tournament
results = await bot.run_tournament(32813, max_questions=20)
print(f"Success rate: {results['success_rate']:.1f}%")
```

### Batch Processing

```python
# Process multiple questions
question_ids = [12345, 12346, 12347]
results = await bot.forecast_questions_batch(question_ids, "ensemble")
```

## 🔧 Configuration Options

### Tournament Configuration

```python
from src.infrastructure.config.tournament_config import get_tournament_config

config = get_tournament_config()
print(f"Tournament mode: {config.mode.value}")
print(f"Tournament ID: {config.tournament_id}")
```

### Resource Management

```python
# Check AskNews usage
asknews_stats = bot.tournament_asknews.get_usage_stats()
print(f"Quota usage: {asknews_stats['quota_usage_percentage']:.1f}%")

# Check proxy usage
proxy_stats = bot.metaculus_proxy.get_usage_stats()
print(f"Proxy requests: {proxy_stats['total_requests']}")
```

## 🛠️ Troubleshooting

### Common Issues

1. **SSL Certificate Errors**
   - These are network-related and handled by fallbacks
   - Bot continues to work with alternative providers

2. **AskNews Authentication Errors**
   - Check `ASKNEWS_CLIENT_ID` and `ASKNEWS_SECRET`
   - Bot falls back to other research providers

3. **OpenRouter Connection Issues**
   - Check `OPENROUTER_API_KEY`
   - Verify network connectivity

4. **Low Success Rate**
   - Check API credentials
   - Monitor quota usage
   - Review error logs

### Debugging

```bash
# Enable verbose logging
python3 -m src.main --tournament 32813 --verbose

# Run validation
python3 scripts/validate_tournament_integration.py

# Check specific components
python3 -c "
from src.infrastructure.external_apis.tournament_asknews_client import TournamentAskNewsClient
client = TournamentAskNewsClient()
print(client.get_usage_stats())
"
```

## 📈 Performance Optimization

### Resource Usage

- **AskNews Quota**: Monitor usage to stay within limits
- **Proxy Credits**: Use free Metaculus credits when available
- **Fallback Chain**: Automatic degradation for reliability

### Batch Processing

- Process multiple questions efficiently
- Automatic error handling and recovery
- Resource usage optimization

### Caching

- Research results cached when possible
- Model responses optimized
- Reduced API calls through intelligent fallbacks

## 🏆 Tournament Strategy

### Optimal Settings

```bash
# For maximum accuracy (slower)
python3 -m src.main --tournament 32813 --max-questions 20

# For speed (faster, slightly lower accuracy)
python3 -m src.main --tournament 32813 --max-questions 50 --agent-type chain_of_thought
```

### Resource Management

1. **Monitor Quotas**: Check AskNews usage regularly
2. **Use Proxy Credits**: Leverage free Metaculus credits
3. **Fallback Strategy**: Ensure all fallbacks are working
4. **Batch Processing**: Process questions efficiently

### Quality Assurance

1. **Validation**: Run validation scripts before tournaments
2. **Testing**: Use test scripts to verify functionality
3. **Monitoring**: Track success rates and resource usage
4. **Error Handling**: Ensure graceful degradation

## 🔍 Monitoring and Alerts

### Usage Statistics

```python
# Get comprehensive stats
asknews_stats = bot.tournament_asknews.get_usage_stats()
proxy_stats = bot.metaculus_proxy.get_usage_stats()

# Monitor key metrics
print(f"AskNews quota: {asknews_stats['quota_usage_percentage']:.1f}%")
print(f"Success rate: {asknews_stats['success_rate']:.1f}%")
print(f"Proxy fallback rate: {proxy_stats['fallback_rate']:.1f}%")
```

### Health Checks

```bash
# Quick health check
python3 scripts/validate_tournament_integration.py

# Comprehensive testing
python3 scripts/test_tournament_features.py
```

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**: Configure all API keys
2. **Validation**: Run validation scripts
3. **Testing**: Execute comprehensive tests
4. **Monitoring**: Set up usage monitoring
5. **Deployment**: Deploy with confidence

### GitHub Actions

The bot includes GitHub Actions workflows for:
- Automated testing
- Tournament execution
- Resource monitoring
- Error alerting

## 📚 API Reference

### Main Classes

- `MetaculusForecastingBot`: Main bot class
- `TournamentAskNewsClient`: Tournament-optimized research
- `MetaculusProxyClient`: Proxy API integration
- `ForecastingPipeline`: Core forecasting pipeline

### Key Methods

- `forecast_question()`: Single question forecast
- `forecast_question_ensemble()`: Multi-agent ensemble
- `forecast_questions_batch()`: Batch processing
- `run_tournament()`: Full tournament mode

## 🎯 Success Metrics

### Key Performance Indicators

1. **Accuracy**: Prediction accuracy vs actual outcomes
2. **Coverage**: Percentage of questions successfully forecasted
3. **Efficiency**: Resource usage vs output quality
4. **Reliability**: Uptime and error recovery

### Monitoring Dashboard

Track these metrics:
- Success rate: >90% target
- AskNews quota usage: <80% recommended
- Proxy fallback rate: <20% optimal
- Average confidence: >0.7 good

## 🏆 Tournament Readiness Checklist

- [ ] All API keys configured
- [ ] Validation scripts pass
- [ ] Comprehensive tests pass
- [ ] Resource usage within limits
- [ ] Fallback mechanisms tested
- [ ] Monitoring setup complete
- [ ] Error handling verified
- [ ] Performance optimized

**🎉 Your tournament bot is ready to dominate the Fall 2025 Metaculus tournament!**
