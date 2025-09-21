"""
Tests for enhanced ReActAgent with dynamic reasoning-acting cycles.
"""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from src.agents.react_agent import (
    ActionContext,
    ActionType,
    ActionValidationResult,
    ReActAgent,
    ReActStep,
)
from src.domain.entities.prediction import PredictionMethod
from src.domain.entities.question import Question, QuestionType
from src.domain.value_objects.reasoning_trace import ReasoningStepType
from src.infrastructure.external_apis.llm_client import LLMClient
from src.infrastructure.external_apis.search_client import SearchClient


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = AsyncMock(spec=LLMClient)
    client.chat_completion = AsyncMock(return_value="Test response")
    return client


@pytest.fixture
def mock_search_client():
    """Mock search client for testing."""
    client = AsyncMock(spec=SearchClient)
    client.search = AsyncMock(
        return_value=[
            {
                "title": "Test Result",
                "snippet": "Test snippet",
                "url": "http://test.com",
            }
        ]
    )
    return client


@pytest.fixture
def sample_question():
    """Sample question for testing."""
    return Question.create(
        title="Will AI achieve AGI by 2030?",
        description="A question about artificial general intelligence timeline",
        question_type=QuestionType.BINARY,
        resolution_criteria="AGI is defined as AI that can perform any intellectual task that a human can do",
        close_time=datetime(2030, 1, 1),
        metadata={"categories": ["AI", "Technology"]},
    )


@pytest.fixture
def react_agent(mock_llm_client, mock_search_client):
    """Enhanced ReActAgent instance for testing."""
    return ReActAgent(
        name="test_react_agent",
        model_config={"model": "gpt-4", "temperature": 0.7},
        llm_client=mock_llm_client,
        search_client=mock_search_client,
        max_steps=8,
        confidence_threshold=0.8,
        enable_bias_checks=True,
        enable_uncertainty_assessment=True,
    )


class TestEnhancedReActAgent:
    """Test cases for enhanced ReActAgent functionality."""

    def test_initialization(self, react_agent):
        """Test that ReActAgent initializes with enhanced features."""
        assert react_agent.confidence_threshold == 0.8
        assert react_agent.enable_bias_checks is True
        assert react_agent.enable_uncertainty_assessment is True
        assert react_agent.adaptive_threshold is True
        assert isinstance(react_agent.action_success_rates, dict)

    @pytest.mark.asyncio
    async def test_action_validation_valid(self, react_agent, sample_question):
        """Test action validation for valid actions."""
        context = ActionContext(
            previous_actions=[ActionType.THINK],
            information_gathered={"some info"},
            confidence_threshold=0.7,
        )

        result = await react_agent._validate_action(
            ActionType.SEARCH, "AI AGI timeline", context, sample_question
        )

        assert result == ActionValidationResult.VALID

    @pytest.mark.asyncio
    async def test_action_validation_redundant(self, react_agent, sample_question):
        """Test action validation detects redundant actions."""
        context = ActionContext(
            previous_actions=[ActionType.THINK, ActionType.THINK, ActionType.THINK],
            information_gathered=set(),
            confidence_threshold=0.7,
        )

        result = await react_agent._validate_action(
            ActionType.THINK, "thinking again", context, sample_question
        )

        assert result == ActionValidationResult.REDUNDANT

    def test_is_action_redundant_search(self, react_agent):
        """Test redundancy detection for search actions."""
        context = ActionContext(
            information_gathered={"AI artificial intelligence research"}
        )

        is_redundant = react_agent._is_action_redundant(
            ActionType.SEARCH, "AI artificial intelligence", context
        )

        assert is_redundant is True

    def test_is_action_contextually_appropriate(self, react_agent, sample_question):
        """Test contextual appropriateness of actions."""
        context = ActionContext(
            previous_actions=[ActionType.SEARCH, ActionType.ANALYZE],
            information_gathered={"info1", "info2", "info3"},
        )

        # Bias check should be appropriate after some analysis
        assert (
            react_agent._is_action_contextually_appropriate(
                ActionType.BIAS_CHECK, context, sample_question
            )
            is True
        )

        # Synthesis should be appropriate with multiple pieces of info
        assert (
            react_agent._is_action_contextually_appropriate(
                ActionType.SYNTHESIZE, context, sample_question
            )
            is True
        )

    def test_action_needs_refinement(self, react_agent):
        """Test detection of actions that need refinement."""
        # Short search query needs refinement
        assert (
            react_agent._action_needs_refinement(
                ActionType.SEARCH, "AI", ActionContext()
            )
            is True
        )

        # General analysis needs refinement
        assert (
            react_agent._action_needs_refinement(
                ActionType.ANALYZE, "general analysis", ActionContext()
            )
            is True
        )

        # Specific search is fine
        assert (
            react_agent._action_needs_refinement(
                ActionType.SEARCH,
                "artificial general intelligence timeline research",
                ActionContext(),
            )
            is False
        )

    @pytest.mark.asyncio
    async def test_adapt_reasoning_strategy(self, react_agent, sample_question):
        """Test adaptive reasoning strategy adjustment."""
        context = ActionContext(confidence_threshold=0.8, question_complexity=0.5)
        steps = [
            ReActStep(
                1,
                "thought",
                ActionType.SEARCH,
                "input",
                "obs",
                "reasoning",
                confidence_change=0.1,
            ),
            ReActStep(
                2,
                "thought",
                ActionType.ANALYZE,
                "input",
                "obs",
                "reasoning",
                confidence_change=0.05,
            ),
            ReActStep(
                3,
                "thought",
                ActionType.THINK,
                "input",
                "obs",
                "reasoning",
                confidence_change=0.02,
            ),
        ]

        adapted_context = await react_agent._adapt_reasoning_strategy(
            context, steps, sample_question
        )

        # Should adjust threshold based on low recent confidence changes
        assert adapted_context.confidence_threshold <= context.confidence_threshold

    def test_assess_question_complexity(self, react_agent):
        """Test question complexity assessment."""
        simple_question = Question.create(
            title="Simple question?",
            description="Short description",
            question_type=QuestionType.BINARY,
            resolution_criteria="Simple criteria",
            close_time=datetime(2025, 1, 1),
        )

        complexity = react_agent._assess_question_complexity(simple_question)
        assert 0.0 <= complexity <= 1.0

        # More complex question should have higher complexity
        complex_question = Question.create(
            title="Complex multi-faceted question about various interconnected factors?",
            description=" ".join(["Complex description with many words"] * 20),
            question_type=QuestionType.BINARY,
            resolution_criteria=" ".join(["Detailed resolution criteria"] * 15),
            close_time=datetime(2025, 1, 1),
            metadata={"categories": ["Cat1", "Cat2", "Cat3", "Cat4"]},
        )

        complex_complexity = react_agent._assess_question_complexity(complex_question)
        assert complex_complexity > complexity

    @pytest.mark.asyncio
    async def test_should_terminate_reasoning_loop(self, react_agent, sample_question):
        """Test intelligent reasoning loop termination."""
        context = ActionContext(current_confidence=0.9, confidence_threshold=0.8)

        # Should terminate with high confidence and sufficient steps
        steps = [
            ReActStep(i, "thought", ActionType.SEARCH, "input", "obs", "reasoning")
            for i in range(1, 6)
        ]

        should_terminate = await react_agent._should_terminate_reasoning_loop(
            steps, context, sample_question
        )

        assert should_terminate is True

    def test_has_sufficient_analysis(self, react_agent):
        """Test detection of sufficient analysis."""
        # Insufficient analysis
        steps = [ReActStep(1, "thought", ActionType.THINK, "input", "obs", "reasoning")]
        assert react_agent._has_sufficient_analysis(steps) is False

        # Sufficient analysis
        steps = [
            ReActStep(1, "thought", ActionType.SEARCH, "input", "obs", "reasoning"),
            ReActStep(2, "thought", ActionType.ANALYZE, "input", "obs", "reasoning"),
            ReActStep(3, "thought", ActionType.SYNTHESIZE, "input", "obs", "reasoning"),
            ReActStep(4, "thought", ActionType.BIAS_CHECK, "input", "obs", "reasoning"),
        ]
        assert react_agent._has_sufficient_analysis(steps) is True

    def test_is_stuck_in_loop(self, react_agent):
        """Test detection of reasoning loops."""
        # Not stuck - diverse actions
        diverse_steps = [
            ReActStep(1, "thought1", ActionType.SEARCH, "input", "obs", "reasoning"),
            ReActStep(2, "thought2", ActionType.ANALYZE, "input", "obs", "reasoning"),
            ReActStep(
                3, "thought3", ActionType.SYNTHESIZE, "input", "obs", "reasoning"
            ),
            ReActStep(4, "thought4", ActionType.VALIDATE, "input", "obs", "reasoning"),
        ]
        assert react_agent._is_stuck_in_loop(diverse_steps) is False

        # Stuck - repetitive actions
        repetitive_steps = [
            ReActStep(
                1,
                "similar thought about AI",
                ActionType.THINK,
                "input",
                "obs",
                "reasoning",
            ),
            ReActStep(
                2,
                "similar thought about AI",
                ActionType.THINK,
                "input",
                "obs",
                "reasoning",
            ),
            ReActStep(
                3,
                "similar thought about AI",
                ActionType.THINK,
                "input",
                "obs",
                "reasoning",
            ),
            ReActStep(
                4,
                "similar thought about AI",
                ActionType.THINK,
                "input",
                "obs",
                "reasoning",
            ),
        ]
        assert react_agent._is_stuck_in_loop(repetitive_steps) is True

    @pytest.mark.asyncio
    async def test_create_reasoning_trace(self, react_agent, sample_question):
        """Test creation of reasoning trace from ReAct steps."""
        steps = [
            ReActStep(
                1,
                "thought1",
                ActionType.SEARCH,
                "search query",
                "search results",
                "reasoning1",
            ),
            ReActStep(
                2,
                "thought2",
                ActionType.ANALYZE,
                "analysis target",
                "analysis results",
                "reasoning2",
            ),
            ReActStep(
                3,
                "thought3",
                ActionType.BIAS_CHECK,
                "bias check",
                "bias results",
                "reasoning3",
            ),
        ]

        trace = await react_agent._create_reasoning_trace(sample_question, steps)

        assert trace.question_id == sample_question.id
        assert trace.agent_id == react_agent.name
        assert trace.reasoning_method == "react_enhanced"
        assert len(trace.steps) == 3
        assert len(trace.bias_checks) == 1  # One bias check step
        assert 0.0 <= trace.overall_confidence <= 1.0

    def test_map_action_to_reasoning_type(self, react_agent):
        """Test mapping of ReAct actions to reasoning step types."""
        mappings = [
            (ActionType.SEARCH, ReasoningStepType.OBSERVATION),
            (ActionType.THINK, ReasoningStepType.HYPOTHESIS),
            (ActionType.ANALYZE, ReasoningStepType.ANALYSIS),
            (ActionType.SYNTHESIZE, ReasoningStepType.SYNTHESIS),
            (ActionType.BIAS_CHECK, ReasoningStepType.BIAS_CHECK),
            (ActionType.UNCERTAINTY_ASSESS, ReasoningStepType.UNCERTAINTY_ASSESSMENT),
            (ActionType.FINALIZE, ReasoningStepType.CONCLUSION),
        ]

        for action_type, expected_reasoning_type in mappings:
            result = react_agent._map_action_to_reasoning_type(action_type)
            assert result == expected_reasoning_type

    @pytest.mark.asyncio
    async def test_execute_new_action_types(self, react_agent, sample_question):
        """Test execution of new action types."""
        # Test validate action
        result = await react_agent._execute_validate_action(
            "test validation", sample_question
        )
        assert "Validation of 'test validation'" in result

        # Test bias check action
        result = await react_agent._execute_bias_check_action(
            "test bias check", sample_question
        )
        assert "Bias check for 'test bias check'" in result

        # Test uncertainty assessment action
        result = await react_agent._execute_uncertainty_assess_action(
            "test uncertainty", sample_question
        )
        assert "Uncertainty assessment for 'test uncertainty'" in result

    def test_calculate_confidence_change(self, react_agent):
        """Test confidence change calculation."""
        context = ActionContext()

        # Positive change for good evidence
        change = react_agent._calculate_confidence_change(
            ActionType.SEARCH,
            "Found strong evidence and research data",
            "This supports the hypothesis",
            context,
        )
        assert change > 0

        # Negative change for poor results
        change = react_agent._calculate_confidence_change(
            ActionType.SEARCH,
            "No results found, unclear information",
            "This is uncertain and conflicting",
            context,
        )
        assert change < 0

    def test_suggest_alternative_action(self, react_agent):
        """Test alternative action suggestion."""
        context = ActionContext(
            previous_actions=[ActionType.THINK, ActionType.THINK, ActionType.SEARCH]
        )
        steps = []

        alternative = react_agent._suggest_alternative_action(context, steps)

        # Should suggest something other than recent actions
        assert alternative not in [ActionType.THINK, ActionType.SEARCH]
        assert alternative in [
            ActionType.ANALYZE,
            ActionType.SYNTHESIZE,
            ActionType.VALIDATE,
        ]

    @pytest.mark.asyncio
    async def test_refine_action_input(self, react_agent, sample_question):
        """Test action input refinement."""
        # Search refinement
        refined = await react_agent._refine_action_input(
            ActionType.SEARCH, "AI", sample_question
        )
        assert "AI" in refined
        assert len(refined) > len("AI")

        # Analysis refinement
        refined = await react_agent._refine_action_input(
            ActionType.ANALYZE, "general stuff", sample_question
        )
        assert "detailed analysis" in refined.lower()

    @pytest.mark.asyncio
    async def test_full_prediction_cycle(
        self, react_agent, sample_question, mock_llm_client
    ):
        """Test complete prediction cycle with enhanced features."""
        # Mock LLM responses for different steps - need more responses for enhanced agent
        mock_responses = [
            "Thought: I need to search for information\nAction: search\nAction Input: AI AGI timeline research",
            "This search revealed important information about AI development",
            "Thought: I should analyze this information\nAction: analyze\nAction Input: AI development trends",
            "Analysis shows mixed evidence for AGI timeline",
            "Thought: Ready to finalize\nAction: finalize\nAction Input: make prediction",
            "PROBABILITY: 0.3\nCONFIDENCE: 0.7\nREASONING: Based on analysis, 30% chance of AGI by 2030",
        ]
        # Add more responses to handle potential additional steps
        mock_responses.extend(
            [
                "Additional reasoning step response",
                "Bias check completed - no significant biases detected",
                "Uncertainty assessment shows moderate confidence",
                "Validation confirms reasoning quality",
                "Final synthesis of all information",
            ]
            * 3
        )  # Multiply to ensure we have enough responses

        mock_llm_client.chat_completion.side_effect = mock_responses

        prediction = await react_agent.predict(sample_question)

        assert prediction is not None
        assert prediction.method == PredictionMethod.REACT
        assert prediction.created_by == react_agent.name
        assert "react_enhanced" in prediction.method_metadata["agent_type"]
        assert "validation_results" in prediction.method_metadata
        assert "reasoning_trace_id" in prediction.method_metadata


class TestActionContext:
    """Test cases for ActionContext class."""

    def test_action_context_initialization(self):
        """Test ActionContext initialization with defaults."""
        context = ActionContext()

        assert context.previous_actions == []
        assert context.information_gathered == set()
        assert context.confidence_threshold == 0.7
        assert context.time_remaining is None
        assert context.question_complexity == 0.5
        assert context.current_confidence == 0.0

    def test_action_context_with_values(self):
        """Test ActionContext initialization with custom values."""
        context = ActionContext(
            previous_actions=[ActionType.SEARCH],
            information_gathered={"info1", "info2"},
            confidence_threshold=0.8,
            time_remaining=120.0,
            question_complexity=0.7,
            current_confidence=0.5,
        )

        assert context.previous_actions == [ActionType.SEARCH]
        assert context.information_gathered == {"info1", "info2"}
        assert context.confidence_threshold == 0.8
        assert context.time_remaining == 120.0
        assert context.question_complexity == 0.7
        assert context.current_confidence == 0.5


class TestReActStep:
    """Test cases for enhanced ReActStep class."""

    def test_react_step_initialization(self):
        """Test ReActStep initialization with enhanced fields."""
        step = ReActStep(
            step_number=1,
            thought="Test thought",
            action=ActionType.SEARCH,
            action_input="test query",
            observation="test observation",
            reasoning="test reasoning",
            validation_result=ActionValidationResult.VALID,
            confidence_change=0.1,
            execution_time=1.5,
            metadata={"test": "value"},
        )

        assert step.step_number == 1
        assert step.thought == "Test thought"
        assert step.action == ActionType.SEARCH
        assert step.validation_result == ActionValidationResult.VALID
        assert step.confidence_change == 0.1
        assert step.execution_time == 1.5
        assert step.metadata == {"test": "value"}

    def test_react_step_defaults(self):
        """Test ReActStep with default values."""
        step = ReActStep(
            step_number=1,
            thought="Test thought",
            action=ActionType.SEARCH,
            action_input="test query",
            observation="test observation",
            reasoning="test reasoning",
        )

        assert step.validation_result == ActionValidationResult.VALID
        assert step.confidence_change == 0.0
        assert step.execution_time == 0.0
        assert step.metadata == {}
