"""End-to-end tests for the complete forecasting system."""

import os
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
import yaml

from main import TemplateForecaster


class TestEndToEndForecasting:
    """End-to-end tests for the complete forecasting system."""

    @pytest.fixture
    def e2e_config_file(self):
        """Create a configuration file for E2E testing."""
        config_data = {
            "database": {"url": "sqlite:///test_e2e.db"},
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",  # Use cheaper model for testing
                "temperature": 0.1,  # Lower temperature for consistency
                "max_tokens": 2000,
            },
            "search": {"sources": ["duckduckgo"], "max_results": 5, "timeout": 15},
            "metaculus": {
                "base_url": "https://www.metaculus.com/api/v2",
                "timeout": 30,
                "submit_predictions": False,
                "dry_run": True,
            },
            "agent": {"max_iterations": 3, "timeout": 120, "confidence_threshold": 0.6},
            "ensemble": {
                "aggregation_method": "weighted_average",
                "min_agents": 2,
                "max_agents": 3,
            },
            "pipeline": {
                "parallel_execution": False,  # Simpler for testing
                "cache_enabled": True,
            },
            "bot": {
                "name": "MetaculusBot-E2E-Test",
                "max_research_time": 60,
                "research_depth": "light",
            },
            "logging": {
                "level": "INFO",
                "console_enabled": True,
                "file_enabled": False,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            return f.name

    @pytest.fixture
    def e2e_bot(self, e2e_config_file):
        """Create bot instance for E2E testing."""
        # Set required environment variables
        test_env = {
            "OPENAI_API_KEY": "test-key-for-e2e",
            "METACULUS_API_KEY": "test-metaculus-key",
        }

        with patch.dict(os.environ, test_env):
            # TemplateForecaster inherits from ForecastBot and doesn't need a config parameter
            return TemplateForecaster()

    @pytest.fixture
    def mock_real_apis(self):
        """Mock real API responses for E2E testing."""
        # Mock OpenAI API
        openai_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1684924800,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": """{
                            "reasoning": "Based on current AI research trends, expert surveys, and recent breakthroughs in large language models and robotics, I estimate there is a moderate probability of achieving AGI by 2030. Key factors include: 1) Rapid progress in transformer architectures, 2) Increasing computational resources, 3) Growing industry investment. However, significant challenges remain in areas like reasoning, planning, and real-world understanding.",
                            "prediction": 0.35,
                            "confidence": 0.72,
                            "sources": ["AI expert surveys", "Recent ML research papers", "Industry progress reports"]
                        }""",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 250,
                "completion_tokens": 100,
                "total_tokens": 350,
            },
        }

        # Mock Metaculus API
        metaculus_question = {
            "id": 12345,
            "title": "Will artificial general intelligence be achieved by 2030?",
            "description": "This question resolves positively if by January 1, 2030, there exists an AI system that can perform any cognitive task at least as well as a human with comparable training.",
            "resolution_criteria": "AGI is defined as AI that can perform any intellectual task that a human can, with training comparable to what a human would need.",
            "type": "binary",
            "close_time": "2029-12-01T00:00:00Z",
            "resolve_time": "2030-01-01T00:00:00Z",
            "categories": ["AI", "Technology"],
            "tags": ["artificial-intelligence", "agi", "technology"],
            "url": "https://metaculus.com/questions/12345/",
            "community_prediction": {"median": 0.32, "mean": 0.35, "count": 1500},
            "status": "open",
        }

        # Mock search results
        search_results = [
            {
                "title": "State of AI Report 2025",
                "url": "https://example.com/ai-report-2025",
                "snippet": "The 2025 State of AI report shows continued progress in large language models, with GPT-5 demonstrating improved reasoning capabilities. However, experts remain divided on AGI timelines.",
                "source": "duckduckgo",
            },
            {
                "title": "Expert Survey: AGI Predictions",
                "url": "https://example.com/expert-survey",
                "snippet": "A recent survey of 200 AI researchers found that 40% believe AGI will be achieved by 2030, while 35% think it will take until 2035 or later.",
                "source": "duckduckgo",
            },
            {
                "title": "Recent AI Breakthroughs",
                "url": "https://example.com/ai-breakthroughs",
                "snippet": "Recent breakthroughs in multimodal AI and robotics have accelerated progress toward general intelligence, though significant challenges remain.",
                "source": "duckduckgo",
            },
        ]

        return {
            "openai": openai_response,
            "metaculus": metaculus_question,
            "search": search_results,
        }

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_single_question_forecast(self, e2e_bot, mock_real_apis):
        """Test complete forecasting workflow for a single question."""
        with (
            patch("aiohttp.ClientSession.post") as mock_openai_post,
            patch("aiohttp.ClientSession.get") as mock_metaculus_get,
            patch.object(e2e_bot.search_client, "search") as mock_search,
        ):

            # Setup API mocks
            mock_openai_response = Mock()
            mock_openai_response.status = 200
            mock_openai_response.json = Mock(return_value=mock_real_apis["openai"])
            mock_openai_response.__aenter__ = Mock(return_value=mock_openai_response)
            mock_openai_response.__aexit__ = Mock(return_value=None)
            mock_openai_post.return_value = mock_openai_response

            mock_metaculus_response = Mock()
            mock_metaculus_response.status = 200
            mock_metaculus_response.json = Mock(
                return_value=mock_real_apis["metaculus"]
            )
            mock_metaculus_response.__aenter__ = Mock(
                return_value=mock_metaculus_response
            )
            mock_metaculus_response.__aexit__ = Mock(return_value=None)
            mock_metaculus_get.return_value = mock_metaculus_response

            mock_search.return_value = mock_real_apis["search"]

            # Run complete forecast
            result = await e2e_bot.forecast_question(
                question_id=12345, agent_type="chain_of_thought"
            )

            # Verify complete workflow
            assert result is not None
            assert "question" in result
            assert "forecast" in result
            assert "metadata" in result

            # Verify question data
            assert result["question"]["id"] == 12345
            assert "AGI" in result["question"]["title"]

            # Verify forecast data
            forecast = result["forecast"]
            assert "prediction" in forecast
            assert "confidence" in forecast
            assert "reasoning" in forecast
            assert "method" in forecast

            # Verify prediction is valid probability
            assert 0 <= forecast["prediction"] <= 1
            assert 0 <= forecast["confidence"] <= 1

            # Verify reasoning contains analysis
            assert len(forecast["reasoning"]) > 50  # Substantial reasoning
            assert forecast["method"] == "chain_of_thought"

            # Verify metadata
            assert "execution_time" in result["metadata"]
            assert "timestamp" in result["metadata"]
            assert result["metadata"]["execution_time"] > 0

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_ensemble_forecast_workflow(self, e2e_bot, mock_real_apis):
        """Test complete ensemble forecasting workflow."""
        with (
            patch("aiohttp.ClientSession.post") as mock_openai_post,
            patch("aiohttp.ClientSession.get") as mock_metaculus_get,
            patch.object(e2e_bot.search_client, "search") as mock_search,
        ):

            # Setup API mocks with different responses for each agent
            agent_responses = [
                # Chain of Thought response
                {
                    "reasoning": "CoT analysis: Based on recent AI progress...",
                    "prediction": 0.32,
                    "confidence": 0.75,
                },
                # Tree of Thought response
                {
                    "reasoning": "ToT synthesis: Considering multiple perspectives...",
                    "prediction": 0.38,
                    "confidence": 0.82,
                },
                # ReAct response
                {
                    "reasoning": "ReAct conclusion: After researching expert opinions...",
                    "prediction": 0.35,
                    "confidence": 0.78,
                },
            ]

            # Create response cycle for different agent calls
            openai_responses = []
            for response_data in agent_responses:
                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = Mock(
                    return_value={
                        "choices": [{"message": {"content": str(response_data)}}]
                    }
                )
                mock_response.__aenter__ = Mock(return_value=mock_response)
                mock_response.__aexit__ = Mock(return_value=None)
                openai_responses.append(mock_response)

            mock_openai_post.side_effect = openai_responses

            # Setup Metaculus mock
            mock_metaculus_response = Mock()
            mock_metaculus_response.status = 200
            mock_metaculus_response.json = Mock(
                return_value=mock_real_apis["metaculus"]
            )
            mock_metaculus_response.__aenter__ = Mock(
                return_value=mock_metaculus_response
            )
            mock_metaculus_response.__aexit__ = Mock(return_value=None)
            mock_metaculus_get.return_value = mock_metaculus_response

            mock_search.return_value = mock_real_apis["search"]

            # Run ensemble forecast
            result = await e2e_bot.forecast_question_ensemble(
                question_id=12345,
                agent_types=["chain_of_thought", "tree_of_thought", "react"],
            )

            # Verify ensemble results
            assert result is not None
            assert "question" in result
            assert "ensemble_forecast" in result
            assert "individual_forecasts" in result
            assert "metadata" in result

            # Verify individual forecasts
            individual_forecasts = result["individual_forecasts"]
            assert len(individual_forecasts) == 3

            for forecast in individual_forecasts:
                assert "prediction" in forecast
                assert "confidence" in forecast
                assert "method" in forecast
                assert 0 <= forecast["prediction"] <= 1
                assert 0 <= forecast["confidence"] <= 1

            # Verify ensemble forecast
            ensemble = result["ensemble_forecast"]
            assert "prediction" in ensemble
            assert "confidence" in ensemble
            assert "method" in ensemble
            assert ensemble["method"] == "ensemble"

            # Ensemble prediction should be aggregation of individual predictions
            individual_predictions = [f["prediction"] for f in individual_forecasts]
            min_pred, max_pred = min(individual_predictions), max(
                individual_predictions
            )
            assert min_pred <= ensemble["prediction"] <= max_pred

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_batch_forecasting_workflow(self, e2e_bot, mock_real_apis):
        """Test batch forecasting of multiple questions."""
        with (
            patch("aiohttp.ClientSession.post") as mock_openai_post,
            patch("aiohttp.ClientSession.get") as mock_metaculus_get,
            patch.object(e2e_bot.search_client, "search") as mock_search,
        ):

            # Create multiple question responses
            questions = [
                {**mock_real_apis["metaculus"], "id": 12345, "title": "AGI by 2030?"},
                {
                    **mock_real_apis["metaculus"],
                    "id": 12346,
                    "title": "Climate change impact by 2030?",
                },
                {
                    **mock_real_apis["metaculus"],
                    "id": 12347,
                    "title": "Space colony by 2035?",
                },
            ]

            # Setup API mocks
            def mock_get_question(url, **kwargs):
                # Extract question ID from URL
                for question in questions:
                    if str(question["id"]) in url:
                        mock_response = Mock()
                        mock_response.status = 200
                        mock_response.json = Mock(return_value=question)
                        mock_response.__aenter__ = Mock(return_value=mock_response)
                        mock_response.__aexit__ = Mock(return_value=None)
                        return mock_response
                raise ValueError("Question not found")

            mock_metaculus_get.side_effect = mock_get_question

            # Setup OpenAI responses
            openai_responses = []
            for i in range(len(questions)):
                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = Mock(
                    return_value={
                        "choices": [
                            {
                                "message": {
                                    "content": f'{{"reasoning": "Analysis for question {i+1}...", "prediction": {0.3 + i*0.1}, "confidence": 0.75}}'
                                }
                            }
                        ]
                    }
                )
                mock_response.__aenter__ = Mock(return_value=mock_response)
                mock_response.__aexit__ = Mock(return_value=None)
                openai_responses.append(mock_response)

            mock_openai_post.side_effect = openai_responses
            mock_search.return_value = mock_real_apis["search"]

            # Run batch forecast
            results = await e2e_bot.forecast_questions_batch(
                question_ids=[12345, 12346, 12347], agent_type="chain_of_thought"
            )

            # Verify batch results
            assert len(results) == 3

            for i, result in enumerate(results):
                assert "question" in result
                assert "forecast" in result
                assert result["question"]["id"] == questions[i]["id"]
                assert 0 <= result["forecast"]["prediction"] <= 1
                assert 0 <= result["forecast"]["confidence"] <= 1

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_error_recovery_workflow(self, e2e_bot, mock_real_apis):
        """Test error recovery in the complete workflow."""
        with (
            patch("aiohttp.ClientSession.post") as mock_openai_post,
            patch("aiohttp.ClientSession.get") as mock_metaculus_get,
            patch.object(e2e_bot.search_client, "search") as mock_search,
        ):

            # Setup partial failure scenario
            mock_metaculus_response = Mock()
            mock_metaculus_response.status = 200
            mock_metaculus_response.json = Mock(
                return_value=mock_real_apis["metaculus"]
            )
            mock_metaculus_response.__aenter__ = Mock(
                return_value=mock_metaculus_response
            )
            mock_metaculus_response.__aexit__ = Mock(return_value=None)
            mock_metaculus_get.return_value = mock_metaculus_response

            # Search fails
            mock_search.side_effect = Exception("Search service unavailable")

            # LLM still works but with reduced confidence
            mock_openai_response = Mock()
            mock_openai_response.status = 200
            mock_openai_response.json = Mock(
                return_value={
                    "choices": [
                        {
                            "message": {
                                "content": '{"reasoning": "Analysis without search data - reduced confidence", "prediction": 0.35, "confidence": 0.5}'
                            }
                        }
                    ]
                }
            )
            mock_openai_response.__aenter__ = Mock(return_value=mock_openai_response)
            mock_openai_response.__aexit__ = Mock(return_value=None)
            mock_openai_post.return_value = mock_openai_response

            # Should still produce forecast despite search failure
            result = await e2e_bot.forecast_question(
                question_id=12345, agent_type="chain_of_thought"
            )

            assert result is not None
            assert "forecast" in result
            assert result["forecast"]["confidence"] <= 0.6  # Reduced confidence
            assert "error" in result["metadata"]  # Error logged

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_performance_benchmarks(self, e2e_bot, mock_real_apis):
        """Test performance benchmarks for the complete system."""
        with (
            patch("aiohttp.ClientSession.post") as mock_openai_post,
            patch("aiohttp.ClientSession.get") as mock_metaculus_get,
            patch.object(e2e_bot.search_client, "search") as mock_search,
        ):

            # Setup fast mock responses
            mock_openai_response = Mock()
            mock_openai_response.status = 200
            mock_openai_response.json = Mock(return_value=mock_real_apis["openai"])
            mock_openai_response.__aenter__ = Mock(return_value=mock_openai_response)
            mock_openai_response.__aexit__ = Mock(return_value=None)
            mock_openai_post.return_value = mock_openai_response

            mock_metaculus_response = Mock()
            mock_metaculus_response.status = 200
            mock_metaculus_response.json = Mock(
                return_value=mock_real_apis["metaculus"]
            )
            mock_metaculus_response.__aenter__ = Mock(
                return_value=mock_metaculus_response
            )
            mock_metaculus_response.__aexit__ = Mock(return_value=None)
            mock_metaculus_get.return_value = mock_metaculus_response

            mock_search.return_value = mock_real_apis["search"]

            # Measure performance
            start_time = datetime.utcnow()

            result = await e2e_bot.forecast_question(
                question_id=12345, agent_type="chain_of_thought"
            )

            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()

            # Verify performance benchmarks
            assert result is not None
            assert execution_time < 30  # Should complete within 30 seconds with mocks
            assert "execution_time" in result["metadata"]

            # Verify resource usage is reasonable
            if "resource_usage" in result["metadata"]:
                assert (
                    result["metadata"]["resource_usage"]["memory_mb"] < 1000
                )  # Under 1GB
                assert (
                    result["metadata"]["resource_usage"]["api_calls"] <= 5
                )  # Limited API calls

    def teardown_method(self, method):
        """Clean up after each test."""
        # Clean up any temporary files
        for filename in ["test_e2e.db", "test_e2e.db-journal"]:
            if os.path.exists(filename):
                os.unlink(filename)
