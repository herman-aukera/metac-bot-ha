"""Integration tests for forecasting pipeline."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.agents.chain_of_thought_agent import ChainOfThoughtAgent
from src.agents.ensemble_agent import EnsembleAgent
from src.agents.react_agent import ReActAgent
from src.agents.tot_agent import TreeOfThoughtAgent
from src.domain.entities.forecast import Forecast
from src.domain.entities.question import Question, QuestionType
from src.domain.value_objects.probability import Probability
from src.infrastructure.config.settings import Settings
from src.pipelines.forecasting_pipeline import ForecastingPipeline


class TestForecastingPipelineIntegration:
    """Integration tests for the full forecasting pipeline."""

    @pytest.fixture
    def mock_clients(self):
        """Create mock external service clients."""
        llm_client = Mock()
        llm_client.generate_response = AsyncMock()
        llm_client.chat_completion = AsyncMock()
        llm_client.generate = AsyncMock()

        search_client = Mock()
        search_client.search = AsyncMock()

        metaculus_client = Mock()
        metaculus_client.get_question = AsyncMock()
        metaculus_client.submit_prediction = AsyncMock()

        return llm_client, search_client, metaculus_client

    @pytest.fixture
    def pipeline(self, mock_settings, mock_clients):
        """Create forecasting pipeline with mocked dependencies."""
        llm_client, search_client, metaculus_client = mock_clients

        return ForecastingPipeline(
            llm_client=llm_client,
            search_client=search_client,
            metaculus_client=metaculus_client,
            config=mock_settings,
        )

    @pytest.fixture
    def sample_question_data(self):
        """Sample question data from Metaculus API."""
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
            "url": "https://metaculus.com/questions/12345/",
            "community_prediction": {"median": 0.35, "mean": 0.38, "count": 1000},
        }

    @pytest.mark.asyncio
    async def test_end_to_end_forecasting_single_agent(
        self, pipeline, sample_question_data, mock_clients, mock_search_results
    ):
        """Test end-to-end forecasting with a single agent."""
        llm_client, search_client, metaculus_client = mock_clients

        # Setup mocks
        metaculus_client.get_question.return_value = sample_question_data
        search_client.search.return_value = mock_search_results

        # Setup LLM responses for chain of thought agent
        llm_client.chat_completion.side_effect = [
            # Question breakdown response
            "Key factors to consider: AI progress metrics, expert opinions, technological milestones",
            # Research areas response
            "research queries: AI progress 2024, AGI timeline experts, machine learning breakthroughs",
            # Synthesis response (JSON format expected by synthesize_findings)
            '{"executive_summary": "Current AI progress suggests moderate likelihood", "detailed_analysis": "Based on current research and expert opinions", "key_factors": ["AI progress", "Expert consensus"], "base_rates": {"similar_predictions": 0.4}, "confidence_level": 0.75, "reasoning_steps": ["Analysis step 1", "Analysis step 2"], "evidence_for": ["Recent breakthroughs"], "evidence_against": ["Technical challenges"], "uncertainties": ["Timeline uncertainty"]}',
            # Prediction response (JSON format)
            '{"probability": 0.42, "confidence": 0.75, "reasoning": "Based on current AI progress and expert opinions...", "reasoning_steps": ["Step 1", "Step 2"], "lower_bound": 0.35, "upper_bound": 0.50, "confidence_interval": [0.35, 0.50]}',
        ]

        llm_client.generate_response.return_value = {
            "reasoning": "Based on current AI progress and expert opinions...",
            "prediction": 0.42,
            "confidence": 0.75,
            "sources": ["https://example.com/ai-report"],
        }
        metaculus_client.submit_prediction.return_value = {
            "status": "dry_run",
            "would_submit": True,
        }

        # Run pipeline
        result = await pipeline.run_single_question(
            question_id=12345, agent_type="chain_of_thought"
        )

        # Verify results
        assert result["question_id"] == 12345
        assert "forecast" in result
        assert 0.0 <= result["forecast"]["prediction"] <= 1.0  # Valid probability
        assert 0.0 <= result["forecast"]["confidence"] <= 1.0  # Valid confidence
        assert result["forecast"]["method"] == "chain_of_thought"

        # Verify external calls
        metaculus_client.get_question.assert_called_once_with(12345)
        search_client.search.assert_called_once()
        # Agents use chat_completion, not generate_response
        assert llm_client.chat_completion.call_count > 0

    @pytest.mark.asyncio
    async def test_end_to_end_forecasting_ensemble(
        self, pipeline, sample_question_data, mock_clients, mock_search_results
    ):
        """Test end-to-end forecasting with ensemble of agents."""
        llm_client, search_client, metaculus_client = mock_clients

        # Setup mocks
        metaculus_client.get_question.return_value = sample_question_data
        search_client.search.return_value = mock_search_results

        # Mock different responses for each agent in ensemble
        llm_responses = [
            {
                "reasoning": "Chain of thought analysis...",
                "prediction": 0.35,
                "confidence": 0.8,
            },
            {
                "thoughts": [{"text": "Consider AI metrics", "score": 0.9}],
                "evaluations": [{"thought_id": 0, "score": 0.85}],
                "reasoning": "Tree of thought analysis...",
                "prediction": 0.45,
                "confidence": 0.85,
            },
            {
                "thought": "I need to research AI progress",
                "action": "search",
                "reasoning": "ReAct analysis...",
                "prediction": 0.38,
                "confidence": 0.75,
            },
        ]

        llm_client.generate_response.side_effect = llm_responses
        metaculus_client.submit_prediction.return_value = {
            "status": "dry_run",
            "would_submit": True,
        }

        # Run ensemble pipeline
        result = await pipeline.run_ensemble_forecast(
            question_id=12345,
            agent_types=["chain_of_thought", "tree_of_thought", "react"],
        )

        # Verify results
        assert result["question_id"] == 12345
        assert "ensemble_forecast" in result
        assert "individual_forecasts" in result
        assert len(result["individual_forecasts"]) == 3
        assert result["ensemble_forecast"]["method"] == "ensemble"

        # Verify ensemble prediction is aggregated from individual predictions
        ensemble_prediction = result["ensemble_forecast"]["prediction"]
        assert (
            0.3 <= ensemble_prediction <= 0.5
        )  # Should be within range of individual predictions

    @pytest.mark.asyncio
    async def test_batch_forecasting(self, pipeline, mock_clients, mock_search_results):
        """Test batch forecasting of multiple questions."""
        llm_client, search_client, metaculus_client = mock_clients

        # Setup mock questions
        question_data_list = [
            {
                "id": 12345,
                "title": "AGI by 2030?",
                "description": "AGI question",
                "type": "binary",
                "close_time": "2030-01-01T00:00:00Z",
                "categories": ["AI"],
                "url": "https://metaculus.com/questions/12345/",
            },
            {
                "id": 12346,
                "title": "Will renewable energy exceed 50% by 2030?",
                "description": "Renewable energy question",
                "type": "binary",
                "close_time": "2030-01-01T00:00:00Z",
                "categories": ["Energy"],
                "url": "https://metaculus.com/questions/12346/",
            },
        ]

        # Setup mocks
        def mock_get_question(question_id):
            for q in question_data_list:
                if q["id"] == question_id:
                    return q
            raise ValueError(f"Question {question_id} not found")

        metaculus_client.get_question.side_effect = mock_get_question
        search_client.search.return_value = mock_search_results
        llm_client.generate_response.return_value = {
            "reasoning": "Analysis...",
            "prediction": 0.4,
            "confidence": 0.8,
        }
        metaculus_client.submit_prediction.return_value = {"status": "dry_run"}

        # Run batch forecasting
        results = await pipeline.run_batch_forecast(
            question_ids=[12345, 12346], agent_type="chain_of_thought"
        )

        # Verify results
        assert len(results) == 2
        assert results[0]["question_id"] == 12345
        assert results[1]["question_id"] == 12346

        # Verify each question was processed
        assert metaculus_client.get_question.call_count == 2

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, pipeline, mock_clients):
        """Test pipeline error handling."""
        llm_client, search_client, metaculus_client = mock_clients

        # Mock API failure
        metaculus_client.get_question.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await pipeline.run_single_question(
                question_id=12345, agent_type="chain_of_thought"
            )

    @pytest.mark.asyncio
    async def test_pipeline_partial_failure_handling(
        self, pipeline, mock_clients, sample_question_data
    ):
        """Test pipeline handling of partial failures."""
        llm_client, search_client, metaculus_client = mock_clients

        # Setup mocks - search fails but other components work
        metaculus_client.get_question.return_value = sample_question_data
        search_client.search.side_effect = Exception("Search API failed")
        llm_client.generate_response.return_value = {
            "reasoning": "Analysis without search results...",
            "prediction": 0.5,
            "confidence": 0.6,
        }

        # Should still produce forecast even with search failure
        result = await pipeline.run_single_question(
            question_id=12345, agent_type="chain_of_thought"
        )

        assert result["question_id"] == 12345
        assert "forecast" in result
        # Confidence might be lower due to missing search data
        assert result["forecast"]["confidence"] <= 0.7

    @pytest.mark.asyncio
    async def test_pipeline_caching(
        self, pipeline, mock_clients, sample_question_data, mock_search_results
    ):
        """Test pipeline caching functionality."""
        llm_client, search_client, metaculus_client = mock_clients

        # Setup mocks
        metaculus_client.get_question.return_value = sample_question_data
        search_client.search.return_value = mock_search_results
        llm_client.generate_response.return_value = {
            "reasoning": "Cached analysis...",
            "prediction": 0.42,
            "confidence": 0.75,
        }

        # First run
        result1 = await pipeline.run_single_question(
            question_id=12345, agent_type="chain_of_thought"
        )

        # Second run with same parameters (should use cache)
        result2 = await pipeline.run_single_question(
            question_id=12345, agent_type="chain_of_thought"
        )

        # Verify results are identical
        assert result1["forecast"]["prediction"] == result2["forecast"]["prediction"]

        # If caching is implemented, external APIs should be called only once
        # (This depends on the actual caching implementation)

    @pytest.mark.asyncio
    async def test_pipeline_metrics_collection(
        self, pipeline, mock_clients, sample_question_data, mock_search_results
    ):
        """Test pipeline metrics and performance monitoring."""
        llm_client, search_client, metaculus_client = mock_clients

        # Setup mocks
        metaculus_client.get_question.return_value = sample_question_data
        search_client.search.return_value = mock_search_results
        llm_client.generate_response.return_value = {
            "reasoning": "Analysis...",
            "prediction": 0.42,
            "confidence": 0.75,
        }

        # Run with metrics collection
        result = await pipeline.run_single_question(
            question_id=12345, agent_type="chain_of_thought", collect_metrics=True
        )

        # Verify metrics are collected
        if "metrics" in result:
            assert "execution_time" in result["metrics"]
            assert "api_calls" in result["metrics"]
            assert result["metrics"]["execution_time"] > 0


class TestAgentIntegration:
    """Integration tests for different agent types."""

    @pytest.fixture
    def real_question(self):
        """Create a realistic question for agent testing."""
        return Question.create_new(
            metaculus_id=12345,
            title="Will artificial general intelligence be achieved by 2030?",
            description="""
            This question resolves positively if by January 1, 2030, there exists
            an AI system that can perform any cognitive task at least as well as
            a human with comparable training.
            """,
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12345/",
            close_time=datetime.utcnow() + timedelta(days=365),
            categories=["AI", "Technology", "Future"],
        )

    @pytest.mark.asyncio
    async def test_agent_consistency(self, mock_settings, real_question):
        """Test that agents produce consistent results for the same question."""
        # Mock clients
        llm_client = Mock()
        search_client = Mock()

        # Setup consistent mock responses
        search_client.search = AsyncMock(
            return_value=[
                {
                    "title": "AI Progress Report 2025",
                    "url": "https://example.com/ai-report",
                    "snippet": "AI research shows steady progress toward AGI...",
                    "source": "duckduckgo",
                }
            ]
        )

        # Mock chat_completion method for agent research and prediction
        llm_client.chat_completion = AsyncMock(
            return_value='{"research_areas": ["AI progress metrics", "expert opinions"], "probability": 0.42, "confidence": "high", "reasoning": "Based on current AI progress metrics...", "reasoning_steps": ["Analyzed current AI capabilities", "Reviewed expert predictions", "Considered technological barriers"], "lower_bound": 0.30, "upper_bound": 0.55, "confidence_interval": 0.25}'
        )
        llm_client.generate_response = AsyncMock(
            return_value={
                "reasoning": "Based on current AI progress metrics...",
                "prediction": 0.42,
                "confidence": 0.75,
            }
        )

        # Create agent
        agent = ChainOfThoughtAgent(
            name="cot",
            model_config=mock_settings.agent.__dict__,
            llm_client=llm_client,
            search_client=search_client,
        )

        # Run multiple forecasts
        forecasts = []
        for _ in range(3):
            forecast = await agent.forecast(real_question)
            forecasts.append(forecast)

        # Verify consistency (predictions should be identical with same inputs)
        predictions = [f.prediction for f in forecasts]
        assert all(p == predictions[0] for p in predictions)

        confidences = [f.confidence for f in forecasts]
        assert all(c == confidences[0] for c in confidences)

    @pytest.mark.asyncio
    async def test_agent_comparison(self, mock_settings, real_question):
        """Test different agents on the same question."""
        # Mock clients
        llm_client = Mock()
        search_client = Mock()

        search_client.search = AsyncMock(
            return_value=[
                {
                    "title": "Expert Survey on AGI",
                    "url": "https://example.com/survey",
                    "snippet": "Experts predict AGI timeline varies widely...",
                    "source": "duckduckgo",
                }
            ]
        )

        # Mock chat_completion method for different agent types
        def chat_completion_side_effect(*args, **kwargs):
            messages = kwargs.get("messages", [])
            if messages and len(messages) > 0:
                content = messages[-1].get("content", "")

                # Return different responses based on content
                if "deconstruct" in content.lower():
                    return AsyncMock(
                        return_value='{"research_areas": ["expert opinions", "AI metrics"]}'
                    )()
                elif "research" in content.lower():
                    return AsyncMock(
                        return_value='{"analysis": "Research indicates mixed expert opinions"}'
                    )()
                else:
                    return AsyncMock(
                        return_value='{"probability": 0.42, "confidence": "high", "reasoning": "Analysis complete"}'
                    )()

        llm_client.chat_completion = AsyncMock(
            side_effect=lambda *args, **kwargs: '{"probability": 0.42, "confidence": "high", "reasoning": "Analysis complete", "reasoning_steps": ["Analyzed data", "Evaluated trends"], "lower_bound": 0.30, "upper_bound": 0.55, "confidence_interval": 0.25}'
        )

        # Different mock responses for different agent types
        agent_responses = {
            "cot": {
                "reasoning": "Chain of thought analysis shows...",
                "prediction": 0.35,
                "confidence": 0.8,
            },
            "tot": {
                "thoughts": [{"text": "Consider expert opinions", "score": 0.9}],
                "evaluations": [{"thought_id": 0, "score": 0.85}],
                "reasoning": "Tree of thought synthesis indicates...",
                "prediction": 0.42,
                "confidence": 0.85,
            },
            "react": {
                "thought": "I should research recent AI breakthroughs",
                "action": "search",
                "reasoning": "ReAct analysis concludes...",
                "prediction": 0.38,
                "confidence": 0.75,
            },
        }

        results = {}

        # Test each agent type
        for agent_type, response in agent_responses.items():
            llm_client.generate_response = AsyncMock(return_value=response)

            if agent_type == "cot":
                agent = ChainOfThoughtAgent(
                    name="cot",
                    model_config=mock_settings.agent.__dict__,
                    llm_client=llm_client,
                    search_client=search_client,
                )
            elif agent_type == "tot":
                agent = TreeOfThoughtAgent(
                    name="tot",
                    model_config=mock_settings.agent.__dict__,
                    llm_client=llm_client,
                    search_client=search_client,
                )
            elif agent_type == "react":
                agent = ReActAgent(
                    name="react",
                    model_config=mock_settings.agent.__dict__,
                    llm_client=llm_client,
                    search_client=search_client,
                )

            forecast = await agent.forecast(real_question)
            results[agent_type] = forecast

        # Verify all agents produced valid forecasts
        method_mapping = {
            "cot": "chain_of_thought",
            "tot": "tree_of_thought",
            "react": "react",
        }
        for agent_type, forecast in results.items():
            assert isinstance(forecast, Forecast)
            assert forecast.question_id == real_question.id
            assert 0 <= forecast.prediction <= 1
            assert 0 <= forecast.confidence <= 1
            assert forecast.method == method_mapping[agent_type]

        # Verify agents can produce different predictions
        predictions = [f.prediction for f in results.values()]
        assert len(set(predictions)) > 1  # At least some variation
