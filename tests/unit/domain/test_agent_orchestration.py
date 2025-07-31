"""Unit tests for agent orchestration system."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from uuid import uuid4

from src.domain.services.agent_orchestration import (
    BaseAgent, ChainOfThoughtAgent, TreeOfThoughtAgent, ReActAgent, AutoCoTAgent,
    EnsembleAgent, AgentOrchestrator, AggregationMethod, ConsensusMetrics,
    ReasoningBranch, ReasoningTrace, ResearchReport
)
from src.domain.entities.question import Question, QuestionType, QuestionCategory
from src.domain.entities.forecast import Forecast
from src.domain.entities.agent import ReasoningStyle
from src.domain.value_objects.confidence import Confidence
from src.domain.value_objects.reasoning_step import ReasoningStep


class TestBaseAgent:
    """Test cases for BaseAgent abstract class."""

    def test_base_agent_initialization(self):
        """Test BaseAgent initialization with valid parameters."""
        # Create a concrete implementation for testing
        class TestAgent(BaseAgent):
            async def conduct_research(self, question):
                return Mock()
            async def generate_prediction(self, question, research):
                return {"prediction": 0.5}
            async def forecast(self, question):
                return Mock()

        agent = TestAgent(
            agent_id="test_agent",
            name="Test Agent",
            reasoning_style=ReasoningStyle.CHAIN_OF_THOUGHT,
            knowledge_domains=["technology", "ai_development"]
        )

        assert agent.agent_id == "test_agent"
        assert agent.name == "Test Agent"
        assert agent.reasoning_style == ReasoningStyle.CHAIN_OF_THOUGHT
        assert agent.knowledge_domains == ["technology", "ai_development"]
        assert agent.performance_history == []

    def test_specialization_score_calculation(self):
        """Test specialization score calculation for different categories."""
        class TestAgent(BaseAgent):
            async def conduct_research(self, question):
                return Mock()
            async def generate_prediction(self, question, research):
                return {"prediction": 0.5}
            async def forecast(self, question):
                return Mock()

        agent = TestAgent(
            agent_id="test_agent",
            name="Test Agent",
            reasoning_style=ReasoningStyle.CHAIN_OF_THOUGHT,
            knowledge_domains=["technology", "ai_development"]
        )

        # Direct match
        assert agent.get_specialization_score("technology") == 0.9
        assert agent.get_specialization_score("ai_development") == 0.9

        # Related domain
        assert agent.get_specialization_score("science") == 0.6

        # Unrelated domain
        assert agent.get_specialization_score("politics") == 0.3

    def test_performance_tracking(self):
        """Test performance tracking and recent performance calculation."""
        class TestAgent(BaseAgent):
            async def conduct_research(self, question):
                return Mock()
            async def generate_prediction(self, question, research):
                return {"prediction": 0.5}
            async def forecast(self, question):
                return Mock()

        agent = TestAgent(
            agent_id="test_agent",
            name="Test Agent",
            reasoning_style=ReasoningStyle.CHAIN_OF_THOUGHT,
            knowledge_domains=["technology"]
        )

        # Add performance data
        agent.update_performance(0.8, 0.7)
        agent.update_performance(0.9, 0.8)

        assert len(agent.performance_history) == 2

        recent_performance = agent.get_recent_performance()
        assert abs(recent_performance['accuracy'] - 0.85) < 0.001  # Average of 0.8 and 0.9
        assert abs(recent_performance['confidence_calibration'] - 0.75) < 0.001  # Average of 0.7 and 0.8
        assert recent_performance['sample_size'] == 2


class TestChainOfThoughtAgent:
    """Test cases for ChainOfThoughtAgent."""

    @pytest.fixture
    def sample_question(self):
        """Create a sample question for testing."""
        return Question(
            id=1,
            text="Will AI achieve AGI by 2030?",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.AI_DEVELOPMENT,
            deadline=datetime.utcnow() + timedelta(days=30),
            background="Background information about AGI development",
            resolution_criteria="Clear criteria for AGI achievement",
            scoring_weight=1.0
        )

    @pytest.fixture
    def cot_agent(self):
        """Create a ChainOfThoughtAgent for testing."""
        return ChainOfThoughtAgent(
            agent_id="cot_agent",
            name="Chain of Thought Agent",
            knowledge_domains=["ai_development", "technology"],
            llm_client=Mock()
        )

    async def test_conduct_research(self, cot_agent, sample_question):
        """Test research conduct functionality."""
        research = await cot_agent.conduct_research(sample_question)

        assert isinstance(research, ResearchReport)
        assert research.question_id == sample_question.id
        assert research.research_quality_score > 0
        assert isinstance(research.timestamp, datetime)

    async def test_generate_prediction_binary(self, cot_agent, sample_question):
        """Test prediction generation for binary questions."""
        research = await cot_agent.conduct_research(sample_question)
        prediction_result = await cot_agent.generate_prediction(sample_question, research)

        assert "prediction" in prediction_result
        assert "confidence" in prediction_result
        assert "reasoning_steps" in prediction_result

        assert isinstance(prediction_result["confidence"], Confidence)
        assert isinstance(prediction_result["reasoning_steps"], list)
        assert len(prediction_result["reasoning_steps"]) == 3  # Expected number of steps

        # Verify reasoning steps
        for i, step in enumerate(prediction_result["reasoning_steps"]):
            assert isinstance(step, ReasoningStep)
            assert step.step_number == i + 1

    async def test_complete_forecast_workflow(self, cot_agent, sample_question):
        """Test complete forecasting workflow."""
        forecast = await cot_agent.forecast(sample_question)

        assert isinstance(forecast, Forecast)
        assert forecast.question_id == sample_question.id
        assert forecast.agent_id == cot_agent.agent_id
        assert len(forecast.reasoning_trace) > 0


class TestTreeOfThoughtAgent:
    """Test cases for TreeOfThoughtAgent."""

    @pytest.fixture
    def sample_question(self):
        """Create a sample question for testing."""
        return Question(
            id=2,
            text="What will be the global temperature increase by 2050?",
            question_type=QuestionType.NUMERIC,
            category=QuestionCategory.CLIMATE,
            deadline=datetime.utcnow() + timedelta(days=60),
            background="Climate change background",
            resolution_criteria="Temperature measurement criteria",
            scoring_weight=1.5,
            min_value=0.0,
            max_value=10.0
        )

    @pytest.fixture
    def tot_agent(self):
        """Create a TreeOfThoughtAgent for testing."""
        return TreeOfThoughtAgent(
            agent_id="tot_agent",
            name="Tree of Thought Agent",
            knowledge_domains=["climate", "science"],
            configuration={"max_branches": 3, "max_depth": 3}
        )

    async def test_explore_branches(self, tot_agent, sample_question):
        """Test branch exploration functionality."""
        branches = await tot_agent.explore_branches(sample_question, depth=3)

        assert len(branches) == 3  # max_branches

        for branch in branches:
            assert isinstance(branch, ReasoningBranch)
            assert len(branch.reasoning_steps) == 3  # depth
            assert branch.get_branch_quality_score() > 0

    async def test_generate_prediction_with_branches(self, tot_agent, sample_question):
        """Test prediction generation using branch exploration."""
        research = await tot_agent.conduct_research(sample_question)
        prediction_result = await tot_agent.generate_prediction(sample_question, research)

        assert "prediction" in prediction_result
        assert "confidence" in prediction_result
        assert "reasoning_steps" in prediction_result
        assert "branches_explored" in prediction_result
        assert "best_branch_id" in prediction_result

        assert prediction_result["branches_explored"] == 3
        assert prediction_result["best_branch_id"].startswith("branch_")


class TestReActAgent:
    """Test cases for ReActAgent."""

    @pytest.fixture
    def sample_question(self):
        """Create a sample question for testing."""
        return Question(
            id=3,
            text="Which party will win the 2024 election?",
            question_type=QuestionType.MULTIPLE_CHOICE,
            category=QuestionCategory.POLITICS,
            deadline=datetime.utcnow() + timedelta(days=90),
            background="Election background",
            resolution_criteria="Election results",
            scoring_weight=2.0,
            choices=["Party A", "Party B", "Party C"]
        )

    @pytest.fixture
    def react_agent(self):
        """Create a ReActAgent for testing."""
        return ReActAgent(
            agent_id="react_agent",
            name="ReAct Agent",
            knowledge_domains=["politics", "social"],
            configuration={"max_iterations": 5}
        )

    async def test_reason_act_cycle(self, react_agent, sample_question):
        """Test reasoning-action cycle functionality."""
        reasoning_trace = await react_agent.reason_act_cycle(sample_question, max_iterations=3)

        assert isinstance(reasoning_trace, ReasoningTrace)
        assert len(reasoning_trace.iterations) <= 3
        assert reasoning_trace.convergence_score > 0
        assert len(reasoning_trace.final_reasoning) > 0

        # Check iteration structure
        for iteration in reasoning_trace.iterations:
            assert "iteration" in iteration
            assert "reasoning" in iteration
            assert "action" in iteration
            assert "convergence" in iteration


class TestEnsembleAgent:
    """Test cases for EnsembleAgent."""

    @pytest.fixture
    def sample_agents(self):
        """Create sample agents for ensemble testing."""
        agents = []

        # Create mock agents with different characteristics
        for i in range(3):
            agent = Mock(spec=BaseAgent)
            agent.agent_id = f"agent_{i}"
            agent.name = f"Agent {i}"
            agent.reasoning_style = ReasoningStyle.CHAIN_OF_THOUGHT

            # Mock forecast method
            async def mock_forecast(question):
                return Forecast.create_binary(
                    question_id=question.id,
                    probability=0.5 + (i * 0.1),  # Different predictions
                    confidence_level=0.7 + (i * 0.05),
                    confidence_basis=f"Agent {i} reasoning",
                    reasoning_trace=[],
                    evidence_sources=[],
                    agent_id=f"agent_{i}"
                )

            agent.forecast = mock_forecast
            agents.append(agent)

        return agents

    @pytest.fixture
    def sample_question(self):
        """Create a sample binary question."""
        return Question(
            id=4,
            text="Will the stock market go up next month?",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.ECONOMICS,
            deadline=datetime.utcnow() + timedelta(days=30),
            background="Market analysis background",
            resolution_criteria="Market index criteria",
            scoring_weight=1.0
        )

    def test_ensemble_initialization(self, sample_agents):
        """Test EnsembleAgent initialization."""
        ensemble = EnsembleAgent(sample_agents, AggregationMethod.SIMPLE_AVERAGE)

        assert len(ensemble.agents) == 3
        assert ensemble.aggregation_method == AggregationMethod.SIMPLE_AVERAGE
        assert len(ensemble.agent_weights) == 3

    def test_ensemble_initialization_empty_agents(self):
        """Test EnsembleAgent initialization with empty agent list."""
        with pytest.raises(ValueError, match="Ensemble must have at least one agent"):
            EnsembleAgent([], AggregationMethod.SIMPLE_AVERAGE)

    async def test_generate_ensemble_forecast(self, sample_agents, sample_question):
        """Test ensemble forecast generation."""
        ensemble = EnsembleAgent(sample_agents, AggregationMethod.SIMPLE_AVERAGE)

        forecast = await ensemble.generate_ensemble_forecast(sample_question)

        assert isinstance(forecast, Forecast)
        assert forecast.question_id == sample_question.id
        assert forecast.agent_id == "ensemble"
        assert 0.0 <= forecast.get_binary_probability() <= 1.0

    def test_consensus_metrics_calculation(self, sample_agents):
        """Test consensus metrics calculation."""
        ensemble = EnsembleAgent(sample_agents, AggregationMethod.SIMPLE_AVERAGE)

        # Create sample forecasts with different predictions
        forecasts = [
            Forecast.create_binary(1, 0.6, 0.8, "Test", [], [], "agent_0"),
            Forecast.create_binary(1, 0.7, 0.7, "Test", [], [], "agent_1"),
            Forecast.create_binary(1, 0.5, 0.9, "Test", [], [], "agent_2")
        ]

        metrics = ensemble.calculate_consensus_metrics(forecasts)

        assert isinstance(metrics, ConsensusMetrics)
        assert 0.0 <= metrics.consensus_strength <= 1.0
        assert metrics.prediction_variance >= 0.0
        assert 0.0 <= metrics.agent_diversity_score <= 1.0
        assert 0.0 <= metrics.confidence_alignment <= 1.0

    def test_simple_average_aggregation(self, sample_agents, sample_question):
        """Test simple average aggregation method."""
        ensemble = EnsembleAgent(sample_agents, AggregationMethod.SIMPLE_AVERAGE)

        # Create test forecasts
        forecasts = [
            Forecast.create_binary(1, 0.6, 0.8, "Test", [], [], "agent_0"),
            Forecast.create_binary(1, 0.8, 0.7, "Test", [], [], "agent_1"),
            Forecast.create_binary(1, 0.4, 0.9, "Test", [], [], "agent_2")
        ]

        result = ensemble._simple_average(forecasts, sample_question)

        expected_avg = (0.6 + 0.8 + 0.4) / 3
        assert abs(result["value"] - expected_avg) < 0.001

        expected_confidence = (0.8 + 0.7 + 0.9) / 3
        assert abs(result["confidence"] - expected_confidence) < 0.001

    def test_confidence_weighted_aggregation(self, sample_agents, sample_question):
        """Test confidence-weighted aggregation method."""
        ensemble = EnsembleAgent(sample_agents, AggregationMethod.CONFIDENCE_WEIGHTED)

        # Create test forecasts with different confidences
        forecasts = [
            Forecast.create_binary(1, 0.6, 0.9, "High confidence", [], [], "agent_0"),  # High confidence
            Forecast.create_binary(1, 0.8, 0.5, "Low confidence", [], [], "agent_1"),   # Low confidence
            Forecast.create_binary(1, 0.4, 0.7, "Medium confidence", [], [], "agent_2") # Medium confidence
        ]

        result = ensemble._confidence_weighted_average(forecasts, sample_question)

        # High confidence prediction (0.6) should have more weight
        assert result["value"] != (0.6 + 0.8 + 0.4) / 3  # Should not be simple average
        assert 0.0 <= result["value"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0


class TestAgentOrchestrator:
    """Test cases for AgentOrchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create an AgentOrchestrator for testing."""
        return AgentOrchestrator()

    @pytest.fixture
    def sample_agents(self):
        """Create sample agents for orchestrator testing."""
        agents = []

        for i in range(3):
            agent = Mock(spec=BaseAgent)
            agent.agent_id = f"agent_{i}"
            agent.name = f"Agent {i}"
            agent.reasoning_style = ReasoningStyle.CHAIN_OF_THOUGHT
            agent.knowledge_domains = ["technology"] if i == 0 else ["politics"]

            # Mock methods
            agent.get_specialization_score = Mock(return_value=0.8 if i == 0 else 0.3)
            agent.get_recent_performance = Mock(return_value={
                'accuracy': 0.7 + (i * 0.1),
                'confidence_calibration': 0.6 + (i * 0.1),
                'sample_size': 10
            })
            agent.update_performance = Mock()
            agent.performance_history = []

            agents.append(agent)

        return agents

    def test_agent_registration(self, orchestrator, sample_agents):
        """Test agent registration functionality."""
        for agent in sample_agents:
            orchestrator.register_agent(agent)

        assert len(orchestrator.agents) == 3
        assert "agent_0" in orchestrator.agents
        assert "agent_1" in orchestrator.agents
        assert "agent_2" in orchestrator.agents

    def test_ensemble_creation(self, orchestrator, sample_agents):
        """Test ensemble creation from registered agents."""
        # Register agents
        for agent in sample_agents:
            orchestrator.register_agent(agent)

        # Create ensemble
        ensemble = orchestrator.create_ensemble(
            "test_ensemble",
            ["agent_0", "agent_1"],
            AggregationMethod.CONFIDENCE_WEIGHTED
        )

        assert isinstance(ensemble, EnsembleAgent)
        assert len(ensemble.agents) == 2
        assert ensemble.aggregation_method == AggregationMethod.CONFIDENCE_WEIGHTED
        assert "test_ensemble" in orchestrator.ensembles

    def test_ensemble_creation_invalid_agent(self, orchestrator):
        """Test ensemble creation with unregistered agent."""
        with pytest.raises(ValueError, match="Agent invalid_agent not registered"):
            orchestrator.create_ensemble("test_ensemble", ["invalid_agent"])

    def test_best_agents_selection(self, orchestrator, sample_agents):
        """Test selection of best agents for a question."""
        # Register agents
        for agent in sample_agents:
            orchestrator.register_agent(agent)

        # Create a technology question
        question = Question(
            id=5,
            text="Will quantum computing breakthrough happen?",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.TECHNOLOGY,
            deadline=datetime.utcnow() + timedelta(days=30),
            background="Quantum computing background",
            resolution_criteria="Breakthrough criteria",
            scoring_weight=1.0
        )

        best_agents = orchestrator.get_best_agents_for_question(question, max_agents=2)

        assert len(best_agents) <= 2
        # Agent 0 should be selected first due to technology specialization
        assert best_agents[0].agent_id == "agent_0"

    def test_performance_tracking_update(self, orchestrator, sample_agents):
        """Test performance tracking updates."""
        # Register agents
        for agent in sample_agents:
            orchestrator.register_agent(agent)

        # Update performance
        orchestrator.update_performance_tracking("agent_0", 1, 0.9, 0.8)

        # Verify agent's update_performance was called
        sample_agents[0].update_performance.assert_called_once_with(0.9, 0.8)


class TestConsensusMetrics:
    """Test cases for ConsensusMetrics value object."""

    def test_valid_consensus_metrics_creation(self):
        """Test creating valid consensus metrics."""
        metrics = ConsensusMetrics(
            consensus_strength=0.8,
            prediction_variance=0.1,
            agent_diversity_score=0.7,
            confidence_alignment=0.9
        )

        assert metrics.consensus_strength == 0.8
        assert metrics.prediction_variance == 0.1
        assert metrics.agent_diversity_score == 0.7
        assert metrics.confidence_alignment == 0.9

    def test_consensus_metrics_validation(self):
        """Test consensus metrics validation."""
        # Invalid consensus strength
        with pytest.raises(ValueError, match="Consensus strength must be between 0.0 and 1.0"):
            ConsensusMetrics(
                consensus_strength=1.5,
                prediction_variance=0.1,
                agent_diversity_score=0.7,
                confidence_alignment=0.9
            )

        # Invalid prediction variance
        with pytest.raises(ValueError, match="Prediction variance cannot be negative"):
            ConsensusMetrics(
                consensus_strength=0.8,
                prediction_variance=-0.1,
                agent_diversity_score=0.7,
                confidence_alignment=0.9
            )

    def test_consensus_quality_checks(self):
        """Test consensus quality checking methods."""
        high_consensus = ConsensusMetrics(
            consensus_strength=0.9,
            prediction_variance=0.05,
            agent_diversity_score=0.8,
            confidence_alignment=0.85
        )

        low_consensus = ConsensusMetrics(
            consensus_strength=0.3,
            prediction_variance=0.4,
            agent_diversity_score=0.4,
            confidence_alignment=0.5
        )

        assert high_consensus.is_high_consensus()
        assert not low_consensus.is_high_consensus()

        assert high_consensus.is_diverse_ensemble()
        assert not low_consensus.is_diverse_ensemble()


# Integration test for complete agent orchestration workflow
class TestAgentOrchestrationIntegration:
    """Integration tests for complete agent orchestration workflows."""

    @pytest.fixture
    def complete_setup(self):
        """Set up complete agent orchestration system."""
        orchestrator = AgentOrchestrator()

        # Create real agents (not mocks)
        cot_agent = ChainOfThoughtAgent(
            agent_id="cot_agent",
            name="CoT Agent",
            knowledge_domains=["technology", "ai_development"]
        )

        tot_agent = TreeOfThoughtAgent(
            agent_id="tot_agent",
            name="ToT Agent",
            knowledge_domains=["science", "climate"]
        )

        react_agent = ReActAgent(
            agent_id="react_agent",
            name="ReAct Agent",
            knowledge_domains=["politics", "economics"]
        )

        # Register agents
        orchestrator.register_agent(cot_agent)
        orchestrator.register_agent(tot_agent)
        orchestrator.register_agent(react_agent)

        return orchestrator, [cot_agent, tot_agent, react_agent]

    async def test_complete_orchestration_workflow(self, complete_setup):
        """Test complete orchestration workflow from question to forecast."""
        orchestrator, agents = complete_setup

        # Create a technology question
        question = Question(
            id=6,
            text="Will AI surpass human performance in coding by 2025?",
            question_type=QuestionType.BINARY,
            category=QuestionCategory.AI_DEVELOPMENT,
            deadline=datetime.utcnow() + timedelta(days=365),
            background="AI coding capabilities background",
            resolution_criteria="Performance benchmark criteria",
            scoring_weight=1.0
        )

        # Generate optimal forecast
        forecast = await orchestrator.generate_optimal_forecast(question)

        assert isinstance(forecast, Forecast)
        assert forecast.question_id == question.id
        assert 0.0 <= forecast.get_binary_probability() <= 1.0
        assert forecast.confidence.level > 0.0
        assert len(forecast.reasoning_trace) > 0
