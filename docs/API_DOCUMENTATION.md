# 🔌 API Documentation

## 📋 Overview

This document provides comprehensive API documentation for the Metaculus AI Forecasting Bot's new architecture (`src/main.py`).

## 🚀 CLI Interface

### Main Command

```bash
python3 -m src.main [OPTIONS]
```

### Options

| Option            | Short | Type    | Default | Description                                     |
| ----------------- | ----- | ------- | ------- | ----------------------------------------------- |
| `--config`        | `-c`  | PATH    | None    | Path to configuration file                      |
| `--tournament`    | `-t`  | INTEGER | None    | Metaculus tournament ID to forecast on          |
| `--max-questions` | `-n`  | INTEGER | 10      | Maximum number of questions to forecast         |
| `--dry-run`       |       | FLAG    | False   | Run without submitting predictions to Metaculus |
| `--verbose`       | `-v`  | FLAG    | False   | Enable verbose logging                          |

### Examples

```bash
# Basic tournament forecasting
python3 -m src.main --tournament 32813 --max-questions 5

# Dry run mode (no submissions)
python3 -m src.main --tournament 32813 --dry-run --verbose

# Custom configuration
python3 -m src.main --config config/custom.yaml --tournament 32813
```

## 🏗️ Programmatic API

### MetaculusForecastingBot Class

Main class for programmatic access to the forecasting system.

#### Constructor

```python
from src.main import MetaculusForecastingBot
from src.infrastructure.config.settings import Config

config = Config()
bot = MetaculusForecastingBot(config)
```

#### Methods

##### `forecast_question(question_id: int, agent_type: str = "ensemble") -> Dict`

Generate a forecast for a specific question.

**Parameters:**

- `question_id` (int): Metaculus question ID
- `agent_type` (str): Agent type ("ensemble", "chain_of_thought", "tree_of_thought", "react")

**Returns:**

```python
{
    "question": {
        "id": int,
        "title": str,
        "description": str,
        "url": str,
        "close_time": str,  # ISO format
        "categories": List[str]
    },
    "forecast": {
        "prediction": float,  # 0.0 to 1.0 for binary questions
        "confidence": float,  # 0.0 to 1.0
        "reasoning": str,
        "method": str
    },
    "metadata": {
        "agent_type": str,
        "question_id": int,
        "status": str,
        "execution_time": float,
        "timestamp": str  # ISO format
    }
}
```

**Example:**

```python
result = await bot.forecast_question(12345, "ensemble")
print(f"Prediction: {result['forecast']['prediction']:.3f}")
```

##### `forecast_question_ensemble(question_id: int, agent_types: List[str]) -> Dict`

Generate an ensemble forecast using multiple agents.

**Parameters:**

- `question_id` (int): Metaculus question ID
- `agent_types` (List[str]): List of agent types to use

**Returns:**

```python
{
    "question": {...},  # Same as forecast_question
    "ensemble_forecast": {
        "prediction": float,
        "confidence": float,
        "reasoning": str,
        "method": "ensemble",
        "agents_used": List[str],
        "weight_distribution": Dict[str, float]
    },
    "individual_forecasts": [
        {
            "agent": str,
            "method": str,
            "prediction": float,
            "confidence": float,
            "reasoning": str
        }
    ],
    "metadata": {
        "agent_type": "ensemble",
        "question_id": int,
        "status": str,
        "agents_used": List[str],
        "consensus_strength": float
    }
}
```

**Example:**

```python
agents = ["chain_of_thought", "tree_of_thought", "react"]
result = await bot.forecast_question_ensemble(12345, agents)
print(f"Ensemble prediction: {result['ensemble_forecast']['prediction']:.3f}")
```

##### `forecast_questions_batch(question_ids: List[int], agent_type: str = "chain_of_thought") -> List[Dict]`

Generate forecasts for multiple questions in batch.

**Parameters:**

- `question_ids` (List[int]): List of Metaculus question IDs
- `agent_type` (str): Agent type to use for all questions

**Returns:**
List of forecast results (same format as `forecast_question`) or error objects:

```python
[
    {
        # Success result (same as forecast_question)
        "question": {...},
        "forecast": {...},
        "metadata": {...}
    },
    {
        # Error result
        "question_id": int,
        "error": str,
        "status": "failed"
    }
]
```

**Example:**

```python
question_ids = [12345, 12346, 12347]
results = await bot.forecast_questions_batch(question_ids, "ensemble")
successful = [r for r in results if "error" not in r]
print(f"Successfully forecasted {len(successful)}/{len(results)} questions")
```

##### `run_tournament(tournament_id: int, max_questions: int = 10) -> List[Dict]`

Run forecasting on a complete tournament.

**Parameters:**

- `tournament_id` (int): Metaculus tournament ID
- `max_questions` (int): Maximum number of questions to process

**Returns:**
List of forecast results with tournament-specific metadata.

**Example:**

```python
results = await bot.run_tournament(32813, max_questions=20)
print(f"Tournament completed: {len(results)} questions processed")
```

## 🔧 Configuration API

### Environment Variables

#### Required

- `METACULUS_TOKEN`: Metaculus API token
- `OPENROUTER_API_KEY`: OpenRouter API key

#### Tournament Optimization

- `AIB_TOURNAMENT_ID`: Tournament ID (default: 32813)
- `TOURNAMENT_MODE`: Enable tournament mode (default: false)
- `DRY_RUN`: Dry run mode (default: true)
- `PUBLISH_REPORTS`: Publish reasoning reports (default: true)

#### Resource Management

- `ASKNEWS_CLIENT_ID`: AskNews client ID
- `ASKNEWS_SECRET`: AskNews secret key
- `ASKNEWS_QUOTA_LIMIT`: AskNews quota limit (default: 9000)
- `ENABLE_PROXY_CREDITS`: Enable Metaculus proxy credits (default: true)
- `METACULUS_DEFAULT_MODEL`: Default proxy model (default: metaculus/claude-3-5-sonnet)

### Configuration Files

#### YAML Configuration

```yaml
# config/config.production.yaml
tournament:
  id: 32813
  mode: "tournament"
  max_concurrent_questions: 5

llm:
  provider: "openrouter"
  model: "metaculus/claude-3-5-sonnet"
  temperature: 0.3
  fallback_models:
    - "openrouter/anthropic/claude-3-5-sonnet"
    - "openai/gpt-4o-mini"

search:
  provider: "multi_source"
  asknews_quota_limit: 9000
  fallback_providers:
    - "perplexity"
    - "exa"
    - "openrouter"

bot:
  name: "MetaculusBotHA"
  version: "1.0.0"
  publish_reports_to_metaculus: true
  max_concurrent_questions: 2
```

## 🏆 Tournament-Specific APIs

### TournamentAskNewsClient

Tournament-optimized AskNews client with quota management.

#### Methods

##### `get_news_research(question: str) -> str`

Get news research with intelligent fallback chain.

##### `get_usage_stats() -> Dict`

Get current usage statistics:

```python
{
    "total_requests": int,
    "successful_requests": int,
    "failed_requests": int,
    "fallback_requests": int,
    "quota_usage_percentage": float,
    "success_rate": float,
    "fallback_rate": float,
    "estimated_quota_used": int,
    "quota_limit": int,
    "asknews_available": bool
}
```

### MetaculusProxyClient

Metaculus proxy API client for free credits.

#### Methods

##### `get_llm_client(model_type: str = "default", purpose: str = "general") -> LLMClient`

Get LLM client with proxy support and automatic fallback.

##### `get_usage_stats() -> Dict`

Get proxy usage statistics:

```python
{
    "total_requests": int,
    "successful_requests": int,
    "failed_requests": int,
    "fallback_requests": int,
    "success_rate": float,
    "fallback_rate": float,
    "estimated_credits_used": float,
    "proxy_exhausted": bool,
    "proxy_credits_enabled": bool
}
```

## 📊 Response Schemas

### Question Schema

```python
{
    "id": int,                    # Metaculus question ID
    "title": str,                 # Question title
    "description": str,           # Question description
    "url": str,                   # Metaculus URL
    "close_time": str,           # ISO timestamp
    "categories": List[str],      # Question categories
    "question_type": str,        # "binary", "numeric", "multiple_choice"
    "status": str                # "open", "closed", "resolved"
}
```

### Forecast Schema

```python
{
    "prediction": float,          # Main prediction value
    "confidence": float,          # Confidence level (0.0-1.0)
    "reasoning": str,            # Detailed reasoning
    "method": str,               # Forecasting method used
    "agents_used": List[str],    # For ensemble forecasts
    "weight_distribution": Dict  # Agent weights for ensemble
}
```

### Error Schema

```python
{
    "question_id": int,
    "error": str,                # Error message
    "status": "failed",
    "timestamp": str,            # ISO timestamp
    "error_type": str           # Error classification
}
```

## 🔍 Status and Health APIs

### Health Check Endpoints

When running as a service, the system exposes health endpoints:

- `/health`: Basic service health
- `/metrics`: Prometheus metrics
- `/ready`: Readiness probe
- `/live`: Liveness probe

### Metrics Available

- `questions_processed_total`: Total questions processed
- `forecasts_generated_total`: Total forecasts generated
- `api_requests_total`: Total API requests made
- `api_request_duration_seconds`: API request duration histogram
- `tournament_quota_usage`: Current quota usage percentage
- `proxy_credits_used`: Proxy credits consumed
- `fallback_requests_total`: Fallback provider usage

## 🚨 Error Handling

### Common Error Types

1. **Configuration Errors**
   - Missing API keys
   - Invalid tournament ID
   - Malformed configuration files

2. **API Errors**
   - Rate limiting
   - Authentication failures
   - Network connectivity issues

3. **Processing Errors**
   - Invalid question format
   - Prediction generation failures
   - Ensemble aggregation errors

### Error Response Format

All errors follow a consistent format:

```python
{
    "error": str,              # Human-readable error message
    "error_type": str,         # Error classification
    "error_code": str,         # Specific error code
    "timestamp": str,          # ISO timestamp
    "context": Dict,           # Additional error context
    "retry_after": int         # Seconds to wait before retry (if applicable)
}
```

## 🔧 Advanced Usage

### Custom Agent Configuration

```python
from src.agents.chain_of_thought_agent import ChainOfThoughtAgent
from src.infrastructure.config.settings import Config

config = Config()
agent = ChainOfThoughtAgent(
    name="custom_cot",
    model_config=config.llm,
    reasoning_depth=7,
    confidence_threshold=0.8,
    enable_bias_detection=True
)

# Use custom agent
result = await agent.forecast(question)
```

### Tournament Analytics

```python
from src.domain.services.tournament_analytics import TournamentAnalytics

analytics = TournamentAnalytics()
performance = await analytics.analyze_tournament_performance(32813)
print(f"Current ranking: {performance.overall_rank}")
```

This API documentation provides comprehensive coverage of all available interfaces and methods in the new architecture.
