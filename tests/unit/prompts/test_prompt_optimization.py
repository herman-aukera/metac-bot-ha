"""
Tests for prompt optimization and token usage efficiency.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock


from src.domain.entities.question import Question, QuestionType
from src.infrastructure.config.token_tracker import TokenTracker
from src.prompts.calibrated_forecasting_prompts import (
    CalibratedForecastingPrompts,
    CalibrationPromptManager,
)


class TestPromptOptimization:
    """Test prompt optimization for token efficiency."""

    def setup_method(self):
        """Set up test environment."""
        self.prompts = CalibratedForecastingPrompts()
        self.prompt_manager = CalibrationPromptManager()
        self.token_tracker = TokenTracker()

        # Create mock question
        self.mock_question = Mock(spec=Question)
        self.mock_question.title = "Will AI achieve AGI by 2030?"
        self.mock_question.question_type = QuestionType.BINARY
        self.mock_question.close_time = datetime.now() + timedelta(days=365)

    def test_basic_calibrated_prompt_efficiency(self):
        """Test basic calibrated prompt token efficiency."""
        research_summary = (
            "Recent AI developments show rapid progress in large language models."
        )

        # Generate prompt
        prompt = self.prompts.generate_basic_calibrated_prompt(
            self.mock_question, research_summary
        )

        # Count tokens
        token_count = self.token_tracker.count_tokens(prompt, "gpt-4o-mini")

        # Verify prompt is reasonably sized (not too verbose)
        assert token_count < 500, f"Basic prompt too long: {token_count} tokens"
        assert token_count > 100, f"Basic prompt too short: {token_count} tokens"

        # Verify essential calibration elements are present
        assert "BASE RATE" in prompt
        assert "CONFIDENCE" in prompt
        assert "overconfidence" in prompt.lower()
        assert "probability" in prompt.lower()

    def test_scenario_analysis_prompt_structure(self):
        """Test scenario analysis prompt structure and efficiency."""
        research_summary = (
            "Market analysis shows mixed signals for technology adoption."
        )

        # Generate scenario analysis prompt
        prompt = self.prompts.generate_scenario_analysis_prompt(
            self.mock_question, research_summary
        )

        # Count tokens
        token_count = self.token_tracker.count_tokens(prompt, "gpt-4o")

        # Verify prompt structure
        assert "OPTIMISTIC" in prompt
        assert "PESSIMISTIC" in prompt
        assert "BASELINE" in prompt
        assert "UNCERTAINTY FACTORS" in prompt

        # Should be longer than basic but not excessive
        assert token_count < 800, f"Scenario prompt too long: {token_count} tokens"
        assert token_count > 200, f"Scenario prompt too short: {token_count} tokens"

    def test_overconfidence_reduction_prompt(self):
        """Test overconfidence reduction prompt effectiveness."""
        research_summary = "Strong evidence suggests positive outcome."

        # Generate overconfidence reduction prompt
        prompt = self.prompts.generate_overconfidence_reduction_prompt(
            self.mock_question, research_summary
        )

        # Verify overconfidence mitigation elements
        assert "DEVIL'S ADVOCATE" in prompt
        assert "OUTSIDE VIEW" in prompt
        assert "REFERENCE CLASS" in prompt
        assert "contradicts" in prompt.lower()

        # Count tokens for efficiency
        token_count = self.token_tracker.count_tokens(prompt, "gpt-4o-mini")
        assert (
            token_count < 600
        ), f"Overconfidence prompt too long: {token_count} tokens"

    def test_prompt_selection_optimization(self):
        """Test optimal prompt selection based on question characteristics."""
        # Test different question types and contexts
        test_cases = [
            {
                "question_complexity": "simple",
                "available_research": "limited",
                "expected_prompt_type": "basic_calibrated",
            },
            {
                "question_complexity": "complex",
                "available_research": "comprehensive",
                "expected_prompt_type": "scenario_analysis",
            },
            {
                "question_complexity": "medium",
                "available_research": "moderate",
                "expected_prompt_type": "overconfidence_reduction",
            },
        ]

        for case in test_cases:
            selected_prompt = self.prompt_manager.select_optimal_prompt(
                complexity=case["question_complexity"],
                research_quality=case["available_research"],
                budget_constraint="normal",
            )

            # Verify appropriate prompt selection
            assert selected_prompt is not None
            assert isinstance(selected_prompt, str)

    def test_token_usage_comparison(self):
        """Test token usage comparison between different prompt types."""
        research_summary = (
            "Comprehensive research data with multiple data points and analysis."
        )

        # Generate different prompt types
        basic_prompt = self.prompts.generate_basic_calibrated_prompt(
            self.mock_question, research_summary
        )
        scenario_prompt = self.prompts.generate_scenario_analysis_prompt(
            self.mock_question, research_summary
        )
        overconfidence_prompt = self.prompts.generate_overconfidence_reduction_prompt(
            self.mock_question, research_summary
        )

        # Count tokens for each
        basic_tokens = self.token_tracker.count_tokens(basic_prompt, "gpt-4o-mini")
        scenario_tokens = self.token_tracker.count_tokens(
            scenario_prompt, "gpt-4o-mini"
        )
        overconfidence_tokens = self.token_tracker.count_tokens(
            overconfidence_prompt, "gpt-4o-mini"
        )

        # Verify expected token hierarchy
        assert (
            basic_tokens < scenario_tokens
        ), "Basic prompt should be shorter than scenario"
        assert (
            basic_tokens < overconfidence_tokens
        ), "Basic prompt should be shorter than overconfidence"

        # All should be within reasonable bounds
        for tokens in [basic_tokens, scenario_tokens, overconfidence_tokens]:
            assert tokens < 1000, f"Prompt too long: {tokens} tokens"
            assert tokens > 50, f"Prompt too short: {tokens} tokens"

    def test_budget_aware_prompt_selection(self):
        """Test prompt selection under different budget constraints."""
        # Test budget-constrained selection
        budget_prompt = self.prompt_manager.select_optimal_prompt(
            complexity="complex",
            research_quality="comprehensive",
            budget_constraint="tight",
        )

        # Test normal budget selection
        normal_prompt = self.prompt_manager.select_optimal_prompt(
            complexity="complex",
            research_quality="comprehensive",
            budget_constraint="normal",
        )

        # Budget-constrained should prefer shorter prompts
        budget_tokens = self.token_tracker.count_tokens(budget_prompt, "gpt-4o-mini")
        normal_tokens = self.token_tracker.count_tokens(normal_prompt, "gpt-4o-mini")

        # Budget version should be more concise
        assert (
            budget_tokens <= normal_tokens
        ), "Budget prompt should not be longer than normal"

    def teardown_method(self):
        """Clean up test environment."""
        pass
