"""Unit tests for AI agents."""

from unittest.mock import AsyncMock, Mock, patch
import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

from src.agents.base_agent import BaseAgent
from src.agents.chain_of_thought_agent import ChainOfThoughtAgent
from src.agents.tot_agent import TreeOfThoughtAgent
from src.agents.react_agent import ReActAgent
from src.agents.ensemble_agent import EnsembleAgent
from src.domain.entities.question import Question, QuestionType
from src.domain.entities.forecast import Forecast
from src.domain.value_objects.probability import Probability
from src.infrastructure.config.settings import Settings, AggregationMethod


class TestBaseAgent:
    """Test BaseAgent abstract class."""
    
    def test_base_agent_abstract(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent()


class TestChainOfThoughtAgent:
    """Test ChainOfThoughtAgent implementation."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client."""
        client = Mock()
        client.generate_response = AsyncMock()
        client.chat_completion = AsyncMock()
        client.generate = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_search_client(self):
        """Mock search client."""
        client = Mock()
        client.search = AsyncMock()
        return client
    
    @pytest.fixture
    def cot_agent(self, mock_llm_client, mock_search_client, mock_settings):
        """Create ChainOfThoughtAgent instance."""
        return ChainOfThoughtAgent(
            name="test-cot-agent",
            model_config={"temperature": 0.7, "max_tokens": 1000},
            llm_client=mock_llm_client,
            search_client=mock_search_client
        )
    
    @pytest.fixture
    def sample_question(self):
        """Create a sample question for testing."""
        return Question.create_new(
            metaculus_id=12345,
            title="Will AI achieve AGI by 2030?",
            description="This question asks about AGI timeline.",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12345",
            close_time=datetime.utcnow() + timedelta(days=365),
            categories=["AI", "Technology"]
        )
    
    @pytest.mark.asyncio
    async def test_cot_agent_forecast(self, cot_agent, sample_question, mock_search_results):
        """Test ChainOfThought agent forecasting."""
        # Mock search results
        cot_agent.search_client.search.return_value = mock_search_results
        
        # Mock LLM responses for research and prediction
        mock_breakdown_response = {
            "choices": [{"message": {"content": "Question breakdown: AGI timeline, technological milestones"}}]
        }
        mock_research_areas_response = {
            "choices": [{"message": {"content": "research queries: AI progress 2024, AGI timeline experts, machine learning breakthroughs"}}]
        }
        mock_final_response = {
            "choices": [{"message": {"content": '''{"probability": 0.42, "confidence": "high", "reasoning": "Based on current AI progress"}'''}}]
        }
        
        # Set up the mock to return these responses in sequence
        cot_agent.llm_client.chat_completion.side_effect = [
            mock_breakdown_response,
            mock_research_areas_response, 
            mock_final_response
        ]
        
        # Test forecasting
        forecast = await cot_agent.forecast(sample_question)
        
        assert isinstance(forecast, Forecast)
        assert forecast.question_id == sample_question.id
        
        # Assertions for the final_prediction within the Forecast object
        final_prediction = forecast.final_prediction
        assert final_prediction is not None
        assert final_prediction.probability.value == 0.42
        assert final_prediction.confidence == PredictionConfidence.HIGH # Ensure this matches the parsed value
        assert "Based on current AI progress" in final_prediction.reasoning
        assert final_prediction.method == PredictionMethod.CHAIN_OF_THOUGHT
        
        # Verify search was called (likely multiple times by conduct_research)
        assert cot_agent.search_client.search.call_count > 0
        
        # Verify LLM was called for each step
        assert cot_agent.llm_client.generate.call_count == 3
    
    @pytest.mark.asyncio
    async def test_cot_agent_research_gathering(self, cot_agent, sample_question, mock_search_results):
        """Test research gathering functionality."""
        cot_agent.search_client.search.return_value = mock_search_results
        
        research = await cot_agent._gather_research(sample_question.title)
        
        assert len(research) > 0
        assert "title" in research[0]
        assert "snippet" in research[0]
        assert "url" in research[0]
    
    @pytest.mark.asyncio
    async def test_cot_agent_error_handling(self, cot_agent, sample_question):
        """Test error handling in agent."""
        # Mock search to raise exception
        cot_agent.search_client.search.side_effect = Exception("Search failed")
        
        with pytest.raises(Exception, match="Search failed"):
            await cot_agent.forecast(sample_question)


class TestTreeOfThoughtAgent:
    """Test TreeOfThoughtAgent implementation."""
    
    @pytest.fixture
    def tot_agent(self, mock_llm_client, mock_search_client, mock_settings):
        """Create TreeOfThoughtAgent instance."""
        return TreeOfThoughtAgent(
            name="test-tot-agent",
            model_config={"temperature": 0.7, "max_tokens": 1000},
            llm_client=mock_llm_client,
            search_client=mock_search_client
        )
    
    @pytest.fixture
    def sample_question(self):
        """Create a sample question for testing."""
        return Question.create_new(
            metaculus_id=12345,
            title="Will AI achieve AGI by 2030?",
            description="This question asks about AGI timeline.",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12345",
            close_time=datetime.utcnow() + timedelta(days=365),
            categories=["AI", "Technology"]
        )
    
    @pytest.mark.asyncio
    async def test_tot_agent_forecast(self, tot_agent, sample_question, mock_search_results):
        """Test TreeOfThought agent forecasting."""
        # Mock search results
        tot_agent.search_client.search.return_value = mock_search_results
        
        # Mock LLM responses for different stages
        mock_responses = [
            # Initial thoughts generation
            {
                "thoughts": [
                    {"text": "Consider AI progress metrics", "score": 0.8},
                    {"text": "Analyze expert opinions", "score": 0.9},
                    {"text": "Review funding trends", "score": 0.7}
                ]
            },
            # Thought evaluation
            {
                "evaluations": [
                    {"thought_id": 0, "score": 0.85, "reasoning": "Good approach"},
                    {"thought_id": 1, "score": 0.9, "reasoning": "Most reliable"},
                    {"thought_id": 2, "score": 0.75, "reasoning": "Relevant but secondary"}
                ]
            },
            # Final synthesis
            {
                "reasoning": "Based on expert analysis and progress metrics...",
                "prediction": 0.42,
                "confidence": 0.8,
                "sources": ["https://example.com/expert-survey"]
            }
        ]
        
        tot_agent.llm_client.generate_response.side_effect = mock_responses
        
        # Test forecasting
        forecast = await tot_agent.forecast(sample_question)
        
        assert isinstance(forecast, Forecast)
        assert forecast.question_id == sample_question.id
        assert forecast.prediction.value == 0.42
        assert forecast.confidence.value == 0.8
        assert forecast.method == "tree_of_thought"
        
        # Verify multiple LLM calls were made
        assert tot_agent.llm_client.generate_response.call_count >= 2


class TestReActAgent:
    """Test ReActAgent implementation."""
    
    @pytest.fixture
    def react_agent(self, mock_llm_client, mock_search_client, mock_settings):
        """Create ReActAgent instance."""
        return ReActAgent(
            name="test-react-agent",
            model_config={"temperature": 0.7},
            llm_client=mock_llm_client,
            search_client=mock_search_client
        )
    
    @pytest.fixture
    def sample_question(self):
        """Create a sample question for testing."""
        return Question.create_new(
            metaculus_id=12345,
            title="Will AI achieve AGI by 2030?",
            description="This question asks about AGI timeline.",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12345",
            close_time=datetime.utcnow() + timedelta(days=365),
            categories=["AI", "Technology"]
        )
    
    @pytest.mark.asyncio
    async def test_react_agent_forecast(self, react_agent, sample_question, mock_search_results):
        """Test ReAct agent forecasting."""
        # Mock search results
        react_agent.search_client.search.return_value = mock_search_results
        
        # Mock LLM responses for ReAct iterations
        mock_responses = [
            # First iteration - Thought + Action
            {
                "thought": "I need to research recent AI progress",
                "action": "search",
                "action_input": "AI AGI progress 2025"
            },
            # Observation processing
            {
                "thought": "The search results show mixed expert opinions",
                "action": "search",
                "action_input": "expert survey AGI timeline 2030"
            },
            # Final answer
            {
                "thought": "Based on my research, I can now make a prediction",
                "action": "answer",
                "reasoning": "Expert surveys and progress metrics suggest...",
                "prediction": 0.38,
                "confidence": 0.72
            }
        ]
        
        react_agent.llm_client.generate_response.side_effect = mock_responses
        
        # Test forecasting
        forecast = await react_agent.forecast(sample_question)
        
        assert isinstance(forecast, Forecast)
        assert forecast.question_id == sample_question.id
        assert forecast.prediction.value == 0.38
        assert forecast.confidence.value == 0.72
        assert forecast.method == "react"
        
        # Verify search was called during ReAct process
        assert react_agent.search_client.search.call_count >= 1


class TestEnsembleAgent:
    """Test EnsembleAgent implementation."""
    
    @pytest.fixture
    def mock_agents(self, mock_llm_client, mock_search_client, mock_settings):
        """Create mock agent instances."""
        agents = []
        for i in range(3):
            agent = Mock()
            agent.forecast = AsyncMock()
            agents.append(agent)
        return agents
    
    @pytest.fixture
    def ensemble_agent(self, mock_agents, mock_settings):
        """Create EnsembleAgent instance."""
        from src.domain.services.forecasting_service import ForecastingService
        from unittest.mock import Mock
        
        # Create mock forecasting service
        mock_forecasting_service = Mock(spec=ForecastingService)
        mock_forecasting_service.confidence_weighted_average = Mock(return_value=0.42)
        
        return EnsembleAgent(
            name="test-ensemble-agent",
            model_config={"temperature": 0.7},
            agents=mock_agents,
            forecasting_service=mock_forecasting_service
        )
    
    @pytest.fixture
    def sample_question(self):
        """Create a sample question for testing."""
        return Question.create_new(
            metaculus_id=12345,
            title="Will AI achieve AGI by 2030?",
            description="This question asks about AGI timeline.",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12345",
            close_time=datetime.utcnow() + timedelta(days=365),
            categories=["AI", "Technology"]
        )
    
    @pytest.mark.asyncio
    async def test_ensemble_weighted_average(self, ensemble_agent, sample_question, mock_agents):
        """Test ensemble with weighted average aggregation."""
        # Mock individual agent forecasts
        forecasts = [
            Forecast.create_new(
                question_id=sample_question.id,
                prediction=Probability(0.3),
                confidence=Probability(0.8),
                reasoning="Agent 1 reasoning",
                method="cot"
            ),
            Forecast.create_new(
                question_id=sample_question.id,
                prediction=Probability(0.5),
                confidence=Probability(0.9),
                reasoning="Agent 2 reasoning",
                method="tot"
            ),
            Forecast.create_new(
                question_id=sample_question.id,
                prediction=Probability(0.4),
                confidence=Probability(0.7),
                reasoning="Agent 3 reasoning",
                method="react"
            )
        ]
        
        for i, agent in enumerate(mock_agents):
            agent.forecast.return_value = forecasts[i]
        
        # Test ensemble forecasting
        ensemble_forecast = await ensemble_agent.forecast(sample_question)
        
        assert isinstance(ensemble_forecast, Forecast)
        assert ensemble_forecast.question_id == sample_question.id
        assert ensemble_forecast.method == "ensemble"
        
        # Verify all agents were called
        for agent in mock_agents:
            agent.forecast.assert_called_once_with(sample_question)
        
        # Check that prediction is within expected range
        assert 0.3 <= ensemble_forecast.prediction.value <= 0.5
    
    @pytest.mark.asyncio
    async def test_ensemble_simple_average(self, mock_settings):
        """Test ensemble with simple average aggregation."""
        # Update config for simple average
        mock_settings.ensemble.aggregation_method = AggregationMethod.SIMPLE_AVERAGE
        
        # Create agents and ensemble
        mock_agents = []
        for i in range(3):
            agent = Mock()
            agent.forecast = AsyncMock()
            mock_agents.append(agent)
        
        ensemble_agent = EnsembleAgent(
            agents=mock_agents,
            config=mock_settings.ensemble
        )
        
        sample_question = Question.create_new(
            metaculus_id=12345,
            title="Test Question",
            description="Test",
            question_type=QuestionType.BINARY,
            url="https://metaculus.com/questions/12345",
            close_time=datetime.utcnow() + timedelta(days=365),
            categories=["Test"]
        )
        
        # Mock forecasts with equal predictions for easy verification
        forecasts = [
            Forecast.create_new(
                question_id=sample_question.id,
                prediction=Probability(0.4),
                confidence=Probability(0.8),
                reasoning="Reasoning 1",
                method="cot"
            ),
            Forecast.create_new(
                question_id=sample_question.id,
                prediction=Probability(0.6),
                confidence=Probability(0.8),
                reasoning="Reasoning 2",
                method="tot"
            )
        ]
        
        for i, agent in enumerate(mock_agents[:2]):  # Use only 2 agents
            agent.forecast.return_value = forecasts[i]
        
        ensemble_forecast = await ensemble_agent.forecast(sample_question)
        
        # Simple average of 0.4 and 0.6 should be 0.5
        assert abs(ensemble_forecast.prediction.value - 0.5) < 0.01
    
    @pytest.mark.asyncio
    async def test_ensemble_error_handling(self, ensemble_agent, sample_question, mock_agents):
        """Test ensemble error handling when agents fail."""
        # Make first agent fail
        mock_agents[0].forecast.side_effect = Exception("Agent failed")
        
        # Other agents succeed
        for agent in mock_agents[1:]:
            agent.forecast.return_value = Forecast.create_new(
                question_id=sample_question.id,
                prediction=Probability(0.5),
                confidence=Probability(0.8),
                reasoning="Working agent",
                method="test"
            )
        
        # Should still produce forecast with remaining agents
        ensemble_forecast = await ensemble_agent.forecast(sample_question)
        
        assert isinstance(ensemble_forecast, Forecast)
        assert ensemble_forecast.method == "ensemble"
