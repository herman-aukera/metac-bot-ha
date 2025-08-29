"""
Tests for the task complexity analyzer.
"""

import pytest

from src.infrastructure.config.task_complexity_analyzer import (
    ComplexityAssessment,
    ComplexityLevel,
    TaskComplexityAnalyzer,
)


class TestTaskComplexityAnalyzer:
    """Test cases for the TaskComplexityAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TaskComplexityAnalyzer()

    def test_simple_question_assessment(self):
        """Test assessment of a simple binary question."""
        question_text = "Will the next iPhone be released before December 31, 2024?"
        background = "Apple typically releases new iPhones in September each year."

        assessment = self.analyzer.assess_question_complexity(question_text, background)

        assert assessment.level == ComplexityLevel.SIMPLE
        assert assessment.score < 5.0  # Should have low complexity score
        assert "simple" in assessment.reasoning.lower()
        assert assessment.recommended_model == "openai/gpt-4o-mini"

    def test_medium_question_assessment(self):
        """Test assessment of a medium complexity question."""
        question_text = "Will the S&P 500 close above 5000 by the end of 2024?"
        background = "The S&P 500 is currently trading around 4800. Market conditions are volatile."

        assessment = self.analyzer.assess_question_complexity(question_text, background)

        assert assessment.level in [ComplexityLevel.MEDIUM, ComplexityLevel.SIMPLE]
        # Check that the assessment has reasonable factors
        assert (
            assessment.factors["medium_score"] > 0
        )  # Should detect market-related content
        assert assessment.score > 2.0  # Should have some complexity

    def test_complex_question_assessment(self):
        """Test assessment of a complex geopolitical question."""
        question_text = "Will there be a major geopolitical conflict involving multiple international actors before 2025?"
        background = """This question involves complex geopolitical dynamics, multiple factors including
        economic conditions, diplomatic relations, and various international stakeholders. The outcome
        depends on uncertain and interdependent variables across multiple regions."""

        assessment = self.analyzer.assess_question_complexity(question_text, background)

        assert assessment.level == ComplexityLevel.COMPLEX
        assert assessment.score > 5.0  # Should have high complexity score
        assert "complex" in assessment.reasoning.lower()

    def test_model_selection_by_complexity(self):
        """Test model selection based on complexity level."""
        # Simple task should use mini model
        simple_assessment = ComplexityAssessment(
            level=ComplexityLevel.SIMPLE,
            score=2.0,
            factors={},
            recommended_model="openai/gpt-4o-mini",
            reasoning="Simple task",
        )

        model = self.analyzer.get_model_for_task(
            "forecast", simple_assessment, "normal"
        )
        assert model == "openai/gpt-4o-mini"

        # Complex task should use full model in normal budget
        complex_assessment = ComplexityAssessment(
            level=ComplexityLevel.COMPLEX,
            score=8.0,
            factors={},
            recommended_model="openai/gpt-4o",
            reasoning="Complex task",
        )

        model = self.analyzer.get_model_for_task(
            "forecast", complex_assessment, "normal"
        )
        assert model == "openai/gpt-4o"

        # Complex task should downgrade in conservative budget
        model = self.analyzer.get_model_for_task(
            "forecast", complex_assessment, "conservative"
        )
        assert model == "openai/gpt-4o-mini"

    def test_cost_estimation(self):
        """Test cost estimation based on complexity."""
        assessment = ComplexityAssessment(
            level=ComplexityLevel.MEDIUM,
            score=4.0,
            factors={},
            recommended_model="openai/gpt-4o-mini",
            reasoning="Medium complexity",
        )

        cost_estimate = self.analyzer.estimate_cost_per_task(
            assessment, "forecast", "normal"
        )

        assert cost_estimate["complexity"] == "medium"
        assert cost_estimate["task_type"] == "forecast"
        assert cost_estimate["estimated_cost"] > 0
        assert cost_estimate["input_tokens"] > 0
        assert cost_estimate["output_tokens"] > 0

    def test_research_vs_forecast_task_types(self):
        """Test different model selection for research vs forecast tasks."""
        assessment = ComplexityAssessment(
            level=ComplexityLevel.COMPLEX,
            score=8.0,
            factors={},
            recommended_model="openai/gpt-4o",
            reasoning="Complex task",
        )

        # Research should generally use cheaper models
        research_model = self.analyzer.get_model_for_task(
            "research", assessment, "normal"
        )
        forecast_model = self.analyzer.get_model_for_task(
            "forecast", assessment, "normal"
        )

        # In normal budget, complex forecasts can use GPT-4o but research uses mini
        assert research_model == "openai/gpt-4o-mini"
        assert forecast_model == "openai/gpt-4o"

    def test_budget_status_impact(self):
        """Test how budget status affects model selection."""
        assessment = ComplexityAssessment(
            level=ComplexityLevel.COMPLEX,
            score=8.0,
            factors={},
            recommended_model="openai/gpt-4o",
            reasoning="Complex task",
        )

        # Normal budget allows premium model
        normal_model = self.analyzer.get_model_for_task(
            "forecast", assessment, "normal"
        )
        assert normal_model == "openai/gpt-4o"

        # Conservative budget downgrades to cheaper model
        conservative_model = self.analyzer.get_model_for_task(
            "forecast", assessment, "conservative"
        )
        assert conservative_model == "openai/gpt-4o-mini"

        # Emergency budget always uses cheapest model
        emergency_model = self.analyzer.get_model_for_task(
            "forecast", assessment, "emergency"
        )
        assert emergency_model == "openai/gpt-4o-mini"

    def test_complexity_indicators(self):
        """Test specific complexity indicators."""
        # Test simple indicators
        simple_text = "Will the official announcement be made before January 1, 2025?"
        assessment = self.analyzer.assess_question_complexity(simple_text)
        assert assessment.level == ComplexityLevel.SIMPLE

        # Test complex indicators
        complex_text = """Will there be a systemic financial crisis involving multiple
        interdependent factors and geopolitical complications affecting global markets?"""
        assessment = self.analyzer.assess_question_complexity(complex_text)
        assert assessment.level == ComplexityLevel.COMPLEX

        # Test uncertainty language
        uncertain_text = """This outcome depends on various uncertain factors and might be
        contingent on multiple variables that are unclear at this time."""
        assessment = self.analyzer.assess_question_complexity(uncertain_text)
        assert assessment.score > 3.0  # Should increase complexity score

    def test_length_based_complexity(self):
        """Test how text length affects complexity assessment."""
        short_text = "Will X happen?"
        long_text = (
            "Will X happen? " + "This is additional context. " * 100
        )  # Very long text

        short_assessment = self.analyzer.assess_question_complexity(short_text)
        long_assessment = self.analyzer.assess_question_complexity(long_text)

        # Longer text should generally have higher complexity score
        assert long_assessment.score >= short_assessment.score

    def test_assessment_reasoning(self):
        """Test that assessment reasoning is informative."""
        question_text = "Will there be a complex geopolitical situation involving uncertain factors?"
        assessment = self.analyzer.assess_question_complexity(question_text)

        assert len(assessment.reasoning) > 20  # Should have substantial reasoning
        assert assessment.level.value.upper() in assessment.reasoning
        assert "score" in assessment.reasoning.lower()


if __name__ == "__main__":
    pytest.main([__file__])
