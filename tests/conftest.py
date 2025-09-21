"""Test configuration and fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import patch

import pytest
import yaml

from src.infrastructure.config.settings import Settings


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_config_yaml(temp_dir: Path) -> Path:
    """Create a test configuration YAML file."""
    config_data = {
        "database": {
            "url": "sqlite:///test.db",
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
            "pool_recycle": 3600,
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test-key",
            "backup_models": ["gpt-3.5-turbo"],
            "temperature": 0.7,
            "max_tokens": 4000,
            "timeout": 60,
            "max_retries": 3,
            "structured_output": True,
            "parallel_requests": 5,
        },
        "search": {
            "sources": ["duckduckgo", "wikipedia"],
            "max_results": 10,
            "timeout": 30,
            "cache_ttl": 3600,
            "deduplicate": True,
            "language": "en",
        },
        "metaculus": {
            "api_key": "test-metaculus-key",
            "base_url": "https://www.metaculus.com/api/v2",
            "timeout": 30,
            "max_retries": 3,
            "submit_predictions": False,
            "dry_run": True,
            "rate_limit_per_minute": 60,
        },
        "agent": {
            "max_iterations": 5,
            "timeout": 300,
            "confidence_threshold": 0.7,
            "use_memory": True,
            "memory_size": 100,
            "debug_mode": True,
        },
        "ensemble": {
            "aggregation_method": "weighted_average",
            "min_agents": 2,
            "max_agents": 5,
            "confidence_weights": True,
            "diversity_bonus": 0.1,
            "timeout": 600,
        },
        "pipeline": {
            "parallel_execution": True,
            "health_check_interval": 60,
            "benchmark_mode": False,
            "circuit_breaker_threshold": 5,
            "circuit_breaker_timeout": 300,
            "cache_enabled": True,
            "cache_ttl": 1800,
        },
        "bot": {
            "name": "MetaculusBot-Test",
            "version": "0.1.0-test",
            "description": "Test AI forecasting bot",
            "max_research_time": 300,
            "research_depth": "medium",
            "uncertainty_quantification": True,
            "explanation_required": True,
            "min_confidence": 0.6,
        },
        "logging": {
            "level": "DEBUG",
            "format": "structured",
            "file_enabled": True,
            "file_path": "logs/test.log",
            "file_max_size": "10MB",
            "file_backup_count": 3,
            "console_enabled": True,
            "structured_enabled": True,
        },
    }

    config_file = temp_dir / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return config_file


@pytest.fixture
def test_env_vars() -> Generator[Dict[str, str], None, None]:
    """Set test environment variables."""
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        "METACULUS_API_KEY": "test-metaculus-key",
        "SERPAPI_API_KEY": "test-serpapi-key",
        "LOG_LEVEL": "DEBUG",
        "ENVIRONMENT": "test",
        "DATABASE_URL": "sqlite:///test.db",
        "LLM_MODEL": "gpt-4",
        "LLM_TEMPERATURE": "0.7",
        "SEARCH_SOURCES": "duckduckgo,wikipedia",
        "AGENT_MAX_ITERATIONS": "5",
        "ENSEMBLE_AGGREGATION_METHOD": "weighted_average",
        "PIPELINE_PARALLEL_EXECUTION": "true",
        "BOT_NAME": "MetaculusBot-Test",
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_settings(test_config_yaml: Path, test_env_vars: Dict[str, str]) -> Settings:
    """Create a Settings instance with test configuration."""
    with patch.dict(os.environ, test_env_vars):
        return Settings.load_from_yaml(str(test_config_yaml))


@pytest.fixture
def sample_question_data() -> Dict[str, Any]:
    """Sample question data for testing."""
    return {
        "id": 12345,
        "title": "Will AI achieve AGI by 2030?",
        "description": "This question asks about the likelihood of Artificial General Intelligence being achieved by 2030.",
        "resolution_criteria": "AGI is defined as AI that can perform any intellectual task that a human can.",
        "type": "binary",
        "close_time": "2030-01-01T00:00:00Z",
        "resolve_time": "2030-12-31T23:59:59Z",
        "categories": ["AI", "Technology"],
        "tags": ["artificial-intelligence", "agi", "technology"],
        "current_prediction": 0.35,
        "prediction_count": 1000,
        "comment_count": 150,
    }


@pytest.fixture
def sample_forecast_data() -> Dict[str, Any]:
    """Sample forecast data for testing."""
    return {
        "question_id": 12345,
        "prediction": 0.42,
        "confidence": 0.75,
        "reasoning": "Based on current AI progress and expert opinions...",
        "sources": [
            "https://example.com/ai-progress-report",
            "https://example.com/expert-survey",
        ],
        "method": "chain_of_thought",
        "created_at": "2025-05-26T10:00:00Z",
    }


@pytest.fixture
def sample_research_data() -> Dict[str, Any]:
    """Sample research data for testing."""
    return {
        "query": "AI AGI progress 2025",
        "sources": [
            {
                "url": "https://example.com/ai-report",
                "title": "AI Progress Report 2025",
                "content": "Recent advances in AI have shown significant progress...",
                "relevance_score": 0.9,
            },
            {
                "url": "https://example.com/expert-opinion",
                "title": "Expert Opinion on AGI Timeline",
                "content": "Experts believe that AGI could be achieved within...",
                "relevance_score": 0.8,
            },
        ],
        "summary": "Current AI research suggests moderate progress toward AGI...",
        "key_insights": [
            "Large language models showing emergent capabilities",
            "Robotics integration improving rapidly",
            "Compute requirements still significant",
        ],
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1684924800,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Based on my analysis, I estimate a 42% probability that AGI will be achieved by 2030.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200},
    }


@pytest.fixture
def mock_metaculus_response():
    """Mock Metaculus API response."""
    return {
        "id": 12345,
        "title": "Will AI achieve AGI by 2030?",
        "description": "This question asks about the likelihood of Artificial General Intelligence being achieved by 2030.",
        "resolution_criteria": "AGI is defined as AI that can perform any intellectual task that a human can.",
        "type": "binary",
        "close_time": "2030-01-01T00:00:00Z",
        "resolve_time": "2030-12-31T23:59:59Z",
        "categories": ["AI", "Technology"],
        "tags": ["artificial-intelligence", "agi", "technology"],
        "community_prediction": {"median": 0.35, "mean": 0.38, "count": 1000},
        "my_predictions": [],
        "status": "open",
    }


@pytest.fixture
def mock_search_results():
    """Mock search API results."""
    return [
        {
            "title": "AI Progress Report 2025",
            "url": "https://example.com/ai-report",
            "snippet": "Recent advances in AI have shown significant progress toward general intelligence...",
            "source": "duckduckgo",
        },
        {
            "title": "Expert Survey on AGI Timeline",
            "url": "https://example.com/expert-survey",
            "snippet": "Survey of AI researchers indicates mixed opinions on AGI timeline...",
            "source": "duckduckgo",
        },
        {
            "title": "Artificial General Intelligence",
            "url": "https://en.wikipedia.org/wiki/Artificial_general_intelligence",
            "snippet": "Artificial general intelligence (AGI) is the intelligence of a machine that can understand...",
            "source": "wikipedia",
        },
    ]


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    from unittest.mock import AsyncMock, Mock

    client = Mock()

    # Mock generate method
    client.generate = AsyncMock(return_value="Mocked LLM response")
    client.generate_response = AsyncMock(return_value="Mocked LLM response")

    # Mock chat_completion method to return JSON responses for different calls
    async def chat_completion_side_effect(*args, **kwargs):
        # Check the messages to determine what type of response to return
        messages = kwargs.get("messages", [])
        if messages and len(messages) > 0:
            content = messages[-1].get("content", "")

            # For question deconstruction
            if "deconstruct" in content.lower() or "breakdown" in content.lower():
                return '{"research_areas": ["AI progress metrics", "expert opinions", "funding trends"]}'

            # For research analysis
            elif "analyze" in content.lower() or "research" in content.lower():
                return '{"executive_summary": "AI progress is steady", "detailed_analysis": "Detailed analysis", "key_factors": ["Compute power"], "base_rates": {"AGI by 2030": 0.3}, "confidence_level": 0.8, "reasoning_steps": ["Analyzed metrics"], "evidence_for": ["Rapid advancements"], "evidence_against": ["Complexity"], "uncertainties": ["Black swan events"]}'

            # For prediction generation
            elif "probability" in content.lower() or "prediction" in content.lower():
                return '{"probability": 0.42, "confidence": "high", "reasoning": "Based on analysis", "reasoning_steps": ["Synthesized findings"], "lower_bound": 0.30, "upper_bound": 0.55, "confidence_interval": 0.25}'

            # For meta-reasoning
            elif "meta" in content.lower():
                return "PROBABILITY: 0.42\nCONFIDENCE: 0.8\nREASONING: Meta-analysis of agent predictions suggests moderate likelihood."

        # Default response
        return '{"result": "default response", "probability": 0.5, "confidence": 0.7}'

    client.chat_completion = AsyncMock(side_effect=chat_completion_side_effect)
    return client


@pytest.fixture
def mock_search_client():
    """Mock search client."""
    from unittest.mock import AsyncMock, Mock

    client = Mock()
    client.search = AsyncMock()
    client.health_check = AsyncMock(return_value=True)
    return client
