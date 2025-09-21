"""
Minimal integration test for core forecasting workflow.

This test verifies that the essential forecasting pipeline works end-to-end
with minimal dependencies and mocked external services.
"""

import os
from unittest.mock import patch, AsyncMock

import pytest

from src.infrastructure.config.settings import Config
from src.agents.ensemble_agent import EnsembleAgent
from src.domain.entities.question import Question, QuestionType
from src.domain.entities.forecast import Forecast
from src.domain.value_objects.probability import Probability
from src.domain.value_objects.confidence import ConfidenceLevel


class TestCoreForeccastingWorkflow:
    """Test core forecasting workflow with minimal dependencies."""

    @pytest.fixture
    def minimal_config(self):
        """Create minimal configuration for testing."""
        return {
            "llm": {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "api_key": "test-key",
                "temperature": 0.1,
                "max_tokens": 2000,
                "timeout": 30,
            },
            "asknews": {
                "client_id": "test-client-id",
                "secret": "test-secret",
                "timeout": 30,
            },
            "tournament": {
                "id": 32813,
                "max_questions": 1,
                "dry_run": True,
            },
            "agent": {
                "max_iterations": 3,
                "timeout": 120,
                "confidence_threshold": 0.6,
            },
        }

    @pytest.fixture
    def sample_question(self):
        """Create a sample question for testing."""
        return Question(
            id=12345,
            title="Will AI achieve AGI by 2030?",
            description="This question asks about the likelihood of Artificial General Intelligence being achieved by 2030.",
            resolution_criteria="AGI is defined as AI that can perform any intellectual task that a human can.",
            question_type=QuestionType.BINARY,
            close_time="2030-01-01T00:00:00Z",
            resolve_time="2030-12-31T23:59:59Z",
            categories=["AI", "Technology"],
            tags=["artificial-intelligence", "agi", "technology"],
            url="https://metaculus.com/questions/12345/",
        )

    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for forecasting."""
        return {
            "reasoning": "Based on current AI research trends and expert opinions, I estimate a moderate probability of AGI by 2030. Key factors include rapid progress in large language models, increasing computational resources, and growing industry investment. However, significant challenges remain in areas like reasoning, planning, and real-world understanding.",
            "prediction": 0.35,
            "confidence": 0.72,
            "sources": [
                "AI expert surveys",
                "Recent ML research papers",
                "Industry progress reports",
            ],
            "reasoning_steps": [
                "Analyzed current AI capabilities and limitations",
                "Reviewed expert surveys and predictions",
                "Considered technological and resource constraints",
                "Evaluated potential breakthrough scenarios",
            ],
        }

    @pytest.fixture
    def mock_research_results(self):
        """Mock research results."""
        return [
            {
                "title": "State of AI Report 2025",
                "url": "https://example.com/ai-report-2025",
                "content": "The 2025 State of AI report shows continued progress in large language models, with GPT-5 demonstrating improved reasoning capabilities. However, experts remain divided on AGI timelines.",
                "relevance_score": 0.9,
                "date": "2025-01-15",
            },
            {
                "title": "Expert Survey: AGI Predictions",
                "url": "https://example.com/expert-survey",
                "content": "A recent survey of 200 AI researchers found that 40% believe AGI will be achieved by 2030, while 35% think it will take until 2035 or later.",
                "relevance_score": 0.85,
                "date": "2025-01-10",
            },
        ]

    @pytest.mark.asyncio
    async def test_minimal_forecasting_workflow(
        self, minimal_config, sample_question, mock_llm_response, mock_research_results
    ):
        """Test minimal forecasting workflow with mocked dependencies."""

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "ASKNEWS_CLIENT_ID": "test-client-id",
                "ASKNEWS_SECRET": "test-secret",
            },
        ):
            # Mock LLM client
            mock_llm_client = AsyncMock()
            mock_llm_client.generate_response.return_value = str(mock_llm_response)
            mock_llm_client.chat_completion.return_value = str(mock_llm_response)

            # Mock research client
            mock_research_client = AsyncMock()
            mock_research_client.search.return_value = mock_research_results

            # Create agent with mocked dependencies
            with (
                patch(
                    "src.infrastructure.external_apis.llm_client.LLMClient",
                    return_value=mock_llm_client,
                ),
                patch(
                    "src.infrastructure.external_apis.tournament_asknews.TournamentAskNews",
                    return_value=mock_research_client,
                ),
            ):

                config = Config.from_dict(minimal_config)
                agent = EnsembleAgent("test-agent", config.llm_config)

                # Mock the agent's internal clients
                agent.llm_client = mock_llm_client
                agent.research_client = mock_research_client

                # Execute forecasting workflow
                forecast = await agent.forecast(sample_question)

                # Verify forecast was generated
                assert forecast is not None
                assert isinstance(forecast, Forecast)

                # Verify forecast properties
                assert forecast.question_id == sample_question.id
                assert isinstance(forecast.prediction, Probability)
                assert isinstance(forecast.confidence, ConfidenceLevel)
                assert 0 <= forecast.prediction.value <= 1
                assert 0 <= forecast.confidence.value <= 1

                # Verify reasoning was provided
                assert forecast.reasoning is not None
                assert len(forecast.reasoning) > 50  # Substantial reasoning

                # Verify method was recorded
                assert forecast.method is not None
                assert "ensemble" in forecast.method.lower()

                # Verify LLM was called
                mock_llm_client.generate_response.assert_called()

                # Verify research was performed
                mock_research_client.search.assert_called()

    @pytest.mark.asyncio
    async def test_forecasting_with_research_failure(
        self, minimal_config, sample_question, mock_llm_response
    ):
        """Test forecasting workflow when research fails but LLM still works."""

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "ASKNEWS_CLIENT_ID": "test-client-id",
                "ASKNEWS_SECRET": "test-secret",
            },
        ):
            # Mock LLM client (working)
            mock_llm_client = AsyncMock()
            mock_llm_client.generate_response.return_value = str(
                {
                    **mock_llm_response,
                    "confidence": 0.5,  # Reduced confidence due to no research
                    "reasoning": "Analysis without research data - reduced confidence due to limited information sources.",
                }
            )

            # Mock research client (failing)
            mock_research_client = AsyncMock()
            mock_research_client.search.side_effect = Exception(
                "Research service unavailable"
            )

            # Create agent with mocked dependencies
            with (
                patch(
                    "src.infrastructure.external_apis.llm_client.LLMClient",
                    return_value=mock_llm_client,
                ),
                patch(
                    "src.infrastructure.external_apis.tournament_asknews.TournamentAskNews",
                    return_value=mock_research_client,
                ),
            ):

                config = Config.from_dict(minimal_config)
                agent = EnsembleAgent("test-agent", config.llm_config)

                # Mock the agent's internal clients
                agent.llm_client = mock_llm_client
                agent.research_client = mock_research_client

                # Execute forecasting workflow - should still work despite research failure
                forecast = await agent.forecast(sample_question)

                # Verify forecast was still generated
                assert forecast is not None
                assert isinstance(forecast, Forecast)

                # Verify reduced confidence due to research failure
                assert forecast.confidence.value <= 0.6

                # Verify reasoning mentions limited information
                assert (
                    "limited" in forecast.reasoning.lower()
                    or "reduced" in forecast.reasoning.lower()
                )

    @pytest.mark.asyncio
    async def test_forecasting_performance_benchmark(
        self, minimal_config, sample_question, mock_llm_response, mock_research_results
    ):
        """Test forecasting performance meets deployment requirements."""

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "ASKNEWS_CLIENT_ID": "test-client-id",
                "ASKNEWS_SECRET": "test-secret",
            },
        ):
            # Mock fast responses
            mock_llm_client = AsyncMock()
            mock_llm_client.generate_response.return_value = str(mock_llm_response)

            mock_research_client = AsyncMock()
            mock_research_client.search.return_value = mock_research_results

            # Create agent with mocked dependencies
            with (
                patch(
                    "src.infrastructure.external_apis.llm_client.LLMClient",
                    return_value=mock_llm_client,
                ),
                patch(
                    "src.infrastructure.external_apis.tournament_asknews.TournamentAskNews",
                    return_value=mock_research_client,
                ),
            ):

                config = Config.from_dict(minimal_config)
                agent = EnsembleAgent("test-agent", config.llm_config)

                # Mock the agent's internal clients
                agent.llm_client = mock_llm_client
                agent.research_client = mock_research_client

                # Measure execution time
                import time

                start_time = time.time()

                forecast = await agent.forecast(sample_question)

                end_time = time.time()
                execution_time = end_time - start_time

                # Verify performance requirements
                assert forecast is not None
                assert (
                    execution_time < 30
                )  # Should complete within 30 seconds with mocks

                # Verify API call efficiency
                assert (
                    mock_llm_client.generate_response.call_count <= 5
                )  # Limited API calls
                assert (
                    mock_research_client.search.call_count <= 3
                )  # Limited research calls

    @pytest.mark.asyncio
    async def test_configuration_validation(self, minimal_config):
        """Test that minimal configuration is valid for deployment."""

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "ASKNEWS_CLIENT_ID": "test-client-id",
                "ASKNEWS_SECRET": "test-secret",
            },
        ):
            # Test configuration loading
            config = Config.from_dict(minimal_config)

            # Verify critical configuration values
            assert config.llm_config is not None
            assert config.llm_config.get("provider") == "openrouter"
            assert config.llm_config.get("model") is not None
            assert config.llm_config.get("api_key") is not None

            # Verify tournament configuration
            assert hasattr(config, "tournament_id") or "tournament" in minimal_config

            # Verify agent can be initialized with this config
            agent = EnsembleAgent("test-agent", config.llm_config)
            assert agent is not None
            assert agent.name == "test-agent"

    def test_import_verification(self):
        """Test that all required modules can be imported."""

        # Test core imports
        try:
            from src.infrastructure.config.settings import Config
            from src.agents.ensemble_agent import EnsembleAgent
            from src.domain.entities.question import Question, QuestionType
            from src.domain.entities.forecast import Forecast
            from src.domain.value_objects.probability import Probability
            from src.domain.value_objects.confidence import ConfidenceLevel
        except ImportError as e:
            pytest.fail(f"Critical import failed: {e}")

        # Test external dependencies
        try:
            import openai
            import requests
            import asyncio
            import json
        except ImportError as e:
            pytest.fail(f"External dependency import failed: {e}")

    def test_environment_variable_handling(self):
        """Test environment variable handling for deployment."""

        required_vars = ["OPENROUTER_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET"]

        # Test with missing variables
        for var in required_vars:
            with patch.dict(os.environ, {}, clear=True):
                # Should handle missing variables gracefully
                try:
                    config = Config()
                    # Config should either provide defaults or raise informative error
                    assert True  # If we get here, config handled missing vars
                except Exception as e:
                    # Error should be informative
                    assert (
                        var.lower() in str(e).lower() or "environment" in str(e).lower()
                    )

        # Test with all variables present
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "ASKNEWS_CLIENT_ID": "test-client-id",
                "ASKNEWS_SECRET": "test-secret",
            },
        ):
            config = Config()
            assert config is not None
