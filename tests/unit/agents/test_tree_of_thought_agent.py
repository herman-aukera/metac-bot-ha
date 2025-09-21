"""Tests for the enhanced TreeOfThoughtAgent."""

from unittest.mock import AsyncMock

import pytest

from src.agents.tree_of_thought_agent import (
    PathEvaluationCriteria,
    ReasoningPath,
    ReasoningPathType,
    TreeExplorationConfig,
    TreeOfThoughtAgent,
)
from src.domain.entities.prediction import Prediction, PredictionMethod
from src.domain.entities.question import Question, QuestionType
from src.domain.entities.research_report import ResearchReport
from src.domain.value_objects.reasoning_trace import ReasoningStep, ReasoningStepType


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = AsyncMock()
    client.chat_completion = AsyncMock()
    return client


@pytest.fixture
def mock_search_client():
    """Mock search client."""
    client = AsyncMock()
    client.search = AsyncMock(
        return_value=[
            {
                "url": "http://example.com/1",
                "title": "Test Result 1",
                "snippet": "Test snippet 1",
            },
            {
                "url": "http://example.com/2",
                "title": "Test Result 2",
                "snippet": "Test snippet 2",
            },
        ]
    )
    return client


@pytest.fixture
def sample_question():
    """Sample question for testing."""
    return Question.create(
        title="Will AI achieve AGI by 2030?",
        description="Question about artificial general intelligence timeline",
        question_type=QuestionType.BINARY,
        resolution_criteria="AGI is defined as AI that can perform any intellectual task that a human can do",
    )


@pytest.fixture
def sample_research_report(sample_question):
    """Sample research report for testing."""
    return ResearchReport.create_new(
        question_id=sample_question.id,
        title="Research on AGI Timeline",
        executive_summary="Research indicates mixed expert opinions on AGI timeline",
        detailed_analysis="Detailed analysis of AGI development",
        sources=[],
        created_by="test_agent",
        key_factors=["Technical progress", "Funding", "Regulatory environment"],
    )


@pytest.fixture
def exploration_config():
    """Test exploration configuration."""
    return TreeExplorationConfig(
        max_depth=3,
        max_breadth=2,
        max_parallel_paths=4,
        evaluation_threshold=0.5,
        path_selection_top_k=2,
    )


@pytest.fixture
def tree_agent(mock_llm_client, mock_search_client, exploration_config):
    """TreeOfThoughtAgent instance for testing."""
    return TreeOfThoughtAgent(
        name="test_tot_agent",
        model_config={"model": "test-model"},
        llm_client=mock_llm_client,
        search_client=mock_search_client,
        exploration_config=exploration_config,
    )


class TestReasoningPath:
    """Test ReasoningPath functionality."""

    def test_reasoning_path_creation(self):
        """Test creating a reasoning path."""
        path = ReasoningPath(path_type=ReasoningPathType.ANALYTICAL)

        assert path.path_type == ReasoningPathType.ANALYTICAL
        assert path.depth == 0
        assert len(path.steps) == 0
        assert not path.is_complete
        assert path.confidence == 0.5

    def test_add_step(self):
        """Test adding steps to a reasoning path."""
        path = ReasoningPath()

        step = ReasoningStep.create(
            step_type=ReasoningStepType.ANALYSIS,
            content="Test reasoning step",
            confidence=0.8,
        )

        path.add_step(step)

        assert len(path.steps) == 1
        assert path.depth == 1
        assert path.steps[0] == step

    def test_get_overall_score_with_evaluations(self):
        """Test calculating overall score with evaluation criteria."""
        path = ReasoningPath()
        path.evaluation_scores = {
            PathEvaluationCriteria.LOGICAL_COHERENCE: 0.8,
            PathEvaluationCriteria.EVIDENCE_STRENGTH: 0.7,
            PathEvaluationCriteria.ACCURACY_POTENTIAL: 0.9,
        }

        score = path.get_overall_score()

        # Should be weighted average
        assert 0.7 < score < 0.9
        assert score != path.confidence  # Should be different from default confidence

    def test_get_overall_score_without_evaluations(self):
        """Test calculating overall score without evaluation criteria."""
        path = ReasoningPath(confidence=0.75)

        score = path.get_overall_score()

        assert score == 0.75  # Should return confidence when no evaluations

    def test_get_reasoning_summary(self):
        """Test getting reasoning summary."""
        path = ReasoningPath(
            path_type=ReasoningPathType.EMPIRICAL,
            sub_components=["Component 1", "Component 2"],
        )

        step1 = ReasoningStep.create(
            step_type=ReasoningStepType.ANALYSIS,
            content="First reasoning step with detailed analysis",
            confidence=0.8,
        )
        step2 = ReasoningStep.create(
            step_type=ReasoningStepType.HYPOTHESIS,
            content="Second reasoning step with hypothesis formation",
            confidence=0.7,
        )

        path.add_step(step1)
        path.add_step(step2)

        summary = path.get_reasoning_summary()

        assert "empirical" in summary.lower()
        assert "Component 1" in summary
        assert "Component 2" in summary
        assert "First reasoning step" in summary
        assert "Second reasoning step" in summary


class TestTreeExplorationConfig:
    """Test TreeExplorationConfig functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TreeExplorationConfig()

        assert config.max_depth == 4
        assert config.max_breadth == 3
        assert config.max_parallel_paths == 6
        assert config.evaluation_threshold == 0.6
        assert config.path_selection_top_k == 2
        assert config.enable_sub_component_analysis is True
        assert config.enable_parallel_exploration is True
        assert len(config.reasoning_path_types) == 3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TreeExplorationConfig(
            max_depth=5,
            max_breadth=4,
            evaluation_threshold=0.7,
            reasoning_path_types=[
                ReasoningPathType.ANALYTICAL,
                ReasoningPathType.CAUSAL,
            ],
        )

        assert config.max_depth == 5
        assert config.max_breadth == 4
        assert config.evaluation_threshold == 0.7
        assert len(config.reasoning_path_types) == 2


class TestTreeOfThoughtAgent:
    """Test TreeOfThoughtAgent functionality."""

    @pytest.mark.asyncio
    async def test_decompose_question(
        self, tree_agent, sample_question, mock_llm_client
    ):
        """Test question decomposition."""
        mock_llm_client.chat_completion.return_value = """
        1. Technical feasibility of AGI
        2. Current AI research progress
        3. Funding and investment trends
        4. Regulatory and ethical considerations
        5. Expert consensus and predictions
        """

        components = await tree_agent._decompose_question(sample_question)

        assert len(components) == 5
        assert "Technical feasibility of AGI" in components
        assert "Current AI research progress" in components
        assert "Expert consensus and predictions" in components

        # Verify LLM was called with appropriate prompt
        mock_llm_client.chat_completion.assert_called_once()
        call_args = mock_llm_client.chat_completion.call_args
        assert "decompose" in call_args[1]["messages"][1]["content"].lower()

    @pytest.mark.asyncio
    async def test_research_component(
        self, tree_agent, sample_question, mock_search_client
    ):
        """Test researching a specific component."""
        component = "Technical feasibility of AGI"

        sources, summary = await tree_agent._research_component(
            component, sample_question
        )

        assert len(sources) == 2  # Limited to 3 per component, mock returns 2
        assert sources[0]["component"] == component
        assert "Technical feasibility of AGI" in summary

        # Verify search was called
        mock_search_client.search.assert_called_once()
        search_query = mock_search_client.search.call_args[0][0]
        assert component in search_query
        assert sample_question.title in search_query

    @pytest.mark.asyncio
    async def test_conduct_research(
        self, tree_agent, sample_question, mock_llm_client, mock_search_client
    ):
        """Test systematic research conduct."""
        # Mock question decomposition
        mock_llm_client.chat_completion.return_value = """
        1. Technical progress
        2. Expert opinions
        3. Timeline factors
        """

        research_report = await tree_agent.conduct_research(sample_question)

        assert research_report.question_id == sample_question.id
        assert "systematic research" in research_report.executive_summary.lower()
        assert len(research_report.key_factors) == 3
        assert "Technical progress" in research_report.key_factors

        # Verify decomposition and search were called
        mock_llm_client.chat_completion.assert_called()
        assert mock_search_client.search.call_count == 3  # One per component

    @pytest.mark.asyncio
    async def test_initialize_reasoning_tree(
        self, tree_agent, sample_question, sample_research_report, mock_llm_client
    ):
        """Test reasoning tree initialization."""
        # Mock initial step generation
        mock_llm_client.chat_completion.return_value = (
            "Initial analytical reasoning step"
        )

        await tree_agent._initialize_reasoning_tree(
            sample_question, sample_research_report
        )

        # Should create paths for each configured reasoning type
        assert len(tree_agent.reasoning_paths) == len(
            tree_agent.exploration_config.reasoning_path_types
        )

        # Each path should have one initial step
        for path in tree_agent.reasoning_paths.values():
            assert len(path.steps) == 1
            assert path.depth == 1
            assert not path.is_complete

    @pytest.mark.asyncio
    async def test_create_initial_reasoning_path(
        self, tree_agent, sample_question, sample_research_report, mock_llm_client
    ):
        """Test creating initial reasoning path."""
        mock_llm_client.chat_completion.return_value = (
            "Empirical analysis of AGI development"
        )

        path = await tree_agent._create_initial_reasoning_path(
            sample_question, sample_research_report, ReasoningPathType.EMPIRICAL
        )

        assert path.path_type == ReasoningPathType.EMPIRICAL
        assert len(path.steps) == 1
        assert path.depth == 1
        assert len(path.sub_components) == 3  # From sample research report
        assert "Technical progress" in path.sub_components

    @pytest.mark.asyncio
    async def test_generate_initial_step(
        self, tree_agent, sample_question, sample_research_report, mock_llm_client
    ):
        """Test generating initial reasoning step."""
        mock_llm_client.chat_completion.return_value = (
            "Analytical approach to AGI timeline"
        )

        step = await tree_agent._generate_initial_step(
            sample_question, sample_research_report, ReasoningPathType.ANALYTICAL
        )

        assert step.step_type == ReasoningStepType.ANALYSIS
        assert "Analytical approach to AGI timeline" in step.content
        assert step.confidence == 0.6
        assert step.metadata["path_type"] == "analytical"
        assert step.metadata["step_number"] == 1

    @pytest.mark.asyncio
    async def test_evaluate_single_path(self, tree_agent, mock_llm_client):
        """Test evaluating a single reasoning path."""
        # Create a path with steps
        path = ReasoningPath(path_type=ReasoningPathType.ANALYTICAL)
        step = ReasoningStep.create(
            step_type=ReasoningStepType.ANALYSIS,
            content="Detailed analytical reasoning about AGI development",
            confidence=0.8,
        )
        path.add_step(step)

        # Mock evaluation response
        mock_llm_client.chat_completion.return_value = """
        LOGICAL_COHERENCE: 0.8
        EVIDENCE_STRENGTH: 0.7
        NOVELTY: 0.6
        COMPLETENESS: 0.5
        ACCURACY_POTENTIAL: 0.9
        UNCERTAINTY_HANDLING: 0.7
        """

        await tree_agent._evaluate_single_path(path)

        # Check evaluation scores were parsed correctly
        assert len(path.evaluation_scores) == 6
        assert path.evaluation_scores[PathEvaluationCriteria.LOGICAL_COHERENCE] == 0.8
        assert path.evaluation_scores[PathEvaluationCriteria.EVIDENCE_STRENGTH] == 0.7
        assert path.evaluation_scores[PathEvaluationCriteria.ACCURACY_POTENTIAL] == 0.9

    def test_select_paths_for_expansion(self, tree_agent):
        """Test selecting paths for expansion."""
        # Create paths with different scores
        path1 = ReasoningPath(path_type=ReasoningPathType.ANALYTICAL)
        path1.evaluation_scores = {
            PathEvaluationCriteria.LOGICAL_COHERENCE: 0.9,
            PathEvaluationCriteria.ACCURACY_POTENTIAL: 0.8,
        }

        path2 = ReasoningPath(path_type=ReasoningPathType.EMPIRICAL)
        path2.evaluation_scores = {
            PathEvaluationCriteria.LOGICAL_COHERENCE: 0.5,
            PathEvaluationCriteria.ACCURACY_POTENTIAL: 0.4,
        }

        path3 = ReasoningPath(path_type=ReasoningPathType.PROBABILISTIC)
        path3.evaluation_scores = {
            PathEvaluationCriteria.LOGICAL_COHERENCE: 0.7,
            PathEvaluationCriteria.ACCURACY_POTENTIAL: 0.8,
        }

        paths = [path1, path2, path3]
        selected = tree_agent._select_paths_for_expansion(paths)

        # Should select top 2 paths above threshold (0.5)
        assert len(selected) == 2
        assert path1 in selected  # Highest score
        assert path3 in selected  # Second highest above threshold
        assert path2 not in selected  # Below threshold

    @pytest.mark.asyncio
    async def test_generate_next_reasoning_step(
        self, tree_agent, sample_question, sample_research_report, mock_llm_client
    ):
        """Test generating next reasoning step."""
        # Create path with existing step
        path = ReasoningPath(
            path_type=ReasoningPathType.CAUSAL, sub_components=["Factor 1", "Factor 2"]
        )
        initial_step = ReasoningStep.create(
            step_type=ReasoningStepType.ANALYSIS,
            content="Initial causal analysis",
            confidence=0.7,
        )
        path.add_step(initial_step)

        mock_llm_client.chat_completion.return_value = (
            "Deeper causal analysis building on previous step"
        )

        next_step = await tree_agent._generate_next_reasoning_step(
            path, sample_question, sample_research_report
        )

        assert next_step.step_type == ReasoningStepType.ANALYSIS
        assert "Deeper causal analysis" in next_step.content
        assert next_step.confidence == 0.7
        assert next_step.metadata["path_type"] == "causal"
        assert next_step.metadata["step_number"] == 2
        assert next_step.metadata["parent_path_id"] == str(path.id)

    @pytest.mark.asyncio
    async def test_is_path_complete(self, tree_agent):
        """Test checking if path is complete."""
        path = ReasoningPath()

        # Empty path should not be complete
        assert not await tree_agent._is_path_complete(path)

        # Path with conclusion step should be complete
        conclusion_step = ReasoningStep.create(
            step_type=ReasoningStepType.CONCLUSION,
            content="Final conclusion",
            confidence=0.8,
        )
        path.add_step(conclusion_step)

        assert await tree_agent._is_path_complete(path)

        # Path with completion indicators should be complete
        path2 = ReasoningPath()
        indicator_step = ReasoningStep.create(
            step_type=ReasoningStepType.ANALYSIS,
            content="In conclusion, the analysis shows...",
            confidence=0.8,
        )
        path2.add_step(indicator_step)

        assert await tree_agent._is_path_complete(path2)

    def test_parse_synthesis_response(self, tree_agent):
        """Test parsing synthesis response."""
        response = """
        PROBABILITY: 0.65
        CONFIDENCE: 0.8
        REASONING: Based on the analysis of multiple reasoning paths,
        the evidence suggests a moderate probability of AGI by 2030.
        The analytical path showed strong technical progress indicators.
        UNCERTAINTIES: Regulatory changes could impact timeline.
        Funding availability remains uncertain.
        """

        prob, conf, reasoning, uncertainties = tree_agent._parse_synthesis_response(
            response
        )

        assert prob == 0.65
        assert conf == 0.8
        assert "moderate probability" in reasoning
        assert "analytical path" in reasoning
        assert "Regulatory changes" in uncertainties
        assert "Funding availability" in uncertainties

    @pytest.mark.asyncio
    async def test_generate_prediction_integration(
        self, tree_agent, sample_question, sample_research_report, mock_llm_client
    ):
        """Test full prediction generation integration."""

        # Mock all LLM responses - provide enough responses for the full exploration
        def mock_response(*args, **kwargs):
            # Check the content to determine what type of response to give
            content = kwargs.get("messages", [{}])[-1].get("content", "")

            # Check for synthesis first (most specific)
            if "Synthesize a final prediction" in content:
                return """PROBABILITY: 0.42
CONFIDENCE: 0.75
REASONING: Synthesis of analytical, empirical, and probabilistic approaches suggests moderate probability
UNCERTAINTIES: Technical challenges and timeline uncertainty"""
            elif (
                "LOGICAL_COHERENCE" in content
                or "Evaluate this reasoning path" in content
            ):
                return """
                LOGICAL_COHERENCE: 0.8
                EVIDENCE_STRENGTH: 0.7
                NOVELTY: 0.6
                COMPLETENESS: 0.7
                ACCURACY_POTENTIAL: 0.8
                UNCERTAINTY_HANDLING: 0.6
                """
            elif "Continue this" in content or "next logical step" in content:
                return "Next reasoning step building on previous analysis"
            elif "Initial" in content or "initial" in content:
                return "Initial reasoning step for this approach"
            else:
                return "Default reasoning response"

        mock_llm_client.chat_completion.side_effect = mock_response

        prediction = await tree_agent.generate_prediction(
            sample_question, sample_research_report
        )

        assert isinstance(prediction, Prediction)
        assert prediction.method == PredictionMethod.TREE_OF_THOUGHT
        assert prediction.result.binary_probability == 0.42
        assert prediction.created_by == "test_tot_agent"
        assert "tree_of_thought_enhanced" in prediction.method_metadata["agent_type"]
        assert prediction.method_metadata["paths_explored"] >= 3
        assert len(prediction.method_metadata["path_types_used"]) >= 2

    @pytest.mark.asyncio
    async def test_conduct_research_without_search_client(self):
        """Test research conduct without search client."""
        agent = TreeOfThoughtAgent(
            name="test_agent",
            model_config={"model": "test"},
            llm_client=AsyncMock(),
            search_client=None,  # No search client
        )

        question = Question.create(
            title="Test question",
            description="Test description",
            question_type=QuestionType.BINARY,
        )

        research_report = await agent.conduct_research(question)

        assert research_report.question_id == question.id
        assert "search client not available" in research_report.executive_summary
        assert len(research_report.sources) == 0

    @pytest.mark.asyncio
    async def test_error_handling_in_path_evaluation(self, tree_agent, mock_llm_client):
        """Test error handling during path evaluation."""
        path = ReasoningPath()
        step = ReasoningStep.create(
            step_type=ReasoningStepType.ANALYSIS, content="Test content", confidence=0.5
        )
        path.add_step(step)

        # Mock LLM to raise exception
        mock_llm_client.chat_completion.side_effect = Exception("LLM error")

        await tree_agent._evaluate_single_path(path)

        # Should have default scores after error
        assert len(path.evaluation_scores) == len(PathEvaluationCriteria)
        for score in path.evaluation_scores.values():
            assert score == 0.5  # Default score

    @pytest.mark.asyncio
    async def test_prune_reasoning_paths(self, tree_agent):
        """Test pruning reasoning paths."""
        # Create more paths than max_parallel_paths (4)
        for i in range(6):
            path = ReasoningPath()
            path.evaluation_scores = {
                PathEvaluationCriteria.ACCURACY_POTENTIAL: 0.5 + (i * 0.1)
            }
            tree_agent.reasoning_paths[path.id] = path

        assert len(tree_agent.reasoning_paths) == 6

        await tree_agent._prune_reasoning_paths()

        # Should be pruned to max_parallel_paths
        assert (
            len(tree_agent.reasoning_paths)
            == tree_agent.exploration_config.max_parallel_paths
        )

        # Should keep the highest scoring paths
        remaining_scores = [
            path.get_overall_score() for path in tree_agent.reasoning_paths.values()
        ]
        # Top 4 should have scores >= 0.6 (since we created scores from 0.5 to 1.0 in increments of 0.1)
        assert all(score >= 0.6 for score in remaining_scores)


if __name__ == "__main__":
    pytest.main([__file__])
