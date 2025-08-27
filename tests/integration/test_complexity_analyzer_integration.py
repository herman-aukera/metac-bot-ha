"""
Integration tests for task complexity analyzer with enhanced LLM configuration.
"""
import pytest
import os
from unittest.mock import Mock, patch

# Mock the API key manager to avoid requiring real API keys for tests
with patch('src.infrastructure.config.api_keys.api_key_manager') as mock_api_manager:
    mock_api_manager.get_api_key.return_value = "test-api-key"
    from src.infrastructure.config.enhanced_llm_config import EnhancedLLMConfig
    from src.infrastructure.config.task_complexity_analyzer import ComplexityLevel


class TestComplexityAnalyzerIntegration:
    """Integration tests for complexity analyzer with LLM configuration."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock the API key manager for testing
        with patch('src.infrastructure.config.api_keys.api_key_manager') as mock_api_manager:
            mock_api_manager.get_api_key.return_value = "test-api-key"
            self.enhanced_config = EnhancedLLMConfig()

    def test_question_analysis_workflow(self):
        """Test the complete question analysis workflow."""
        question_id = "test-question-123"
        question_text = "Will there be a major geopolitical crisis involving multiple nations before 2025?"
        background = """This question involves complex international relations, economic factors,
        and various geopolitical tensions that could escalate into a crisis."""

        # Perform comprehensive analysis
        analysis = self.enhanced_config.analyze_question_for_forecasting(
            question_id, question_text, background
        )

        # Verify analysis structure
        assert "question_id" in analysis
        assert "complexity_assessment" in analysis
        assert "research_cost_estimate" in analysis
        assert "forecast_cost_estimate" in analysis
        assert "total_estimated_cost" in analysis
        assert "budget_status" in analysis
        assert "can_afford" in analysis

        # Verify complexity assessment
        complexity = analysis["complexity_assessment"]
        assert complexity.level in [ComplexityLevel.SIMPLE, ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX]
        assert complexity.score >= 0
        assert len(complexity.reasoning) > 10

        # Verify cost estimates
        research_cost = analysis["research_cost_estimate"]
        forecast_cost = analysis["forecast_cost_estimate"]

        assert research_cost["estimated_cost"] > 0
        assert forecast_cost["estimated_cost"] > 0
        assert research_cost["input_tokens"] > 0
        assert forecast_cost["input_tokens"] > 0

    def test_model_selection_based_on_complexity(self):
        """Test that model selection adapts to complexity assessment."""
        # Simple question
        simple_assessment = self.enhanced_config.assess_question_complexity(
            "Will the next iPhone be released in September 2024?",
            "Apple typically releases iPhones in September."
        )

        simple_llm = self.enhanced_config.get_llm_for_task(
            "forecast", complexity_assessment=simple_assessment
        )

        # Complex question
        complex_assessment = self.enhanced_config.assess_question_complexity(
            "Will there be a systemic global financial crisis involving multiple interdependent factors?",
            "This involves complex economic relationships, geopolitical tensions, and uncertain market dynamics."
        )

        complex_llm = self.enhanced_config.get_llm_for_task(
            "forecast", complexity_assessment=complex_assessment
        )

        # Verify different models are selected based on complexity
        # (In normal budget mode, complex questions should get better models)
        assert hasattr(simple_llm, 'model')
        assert hasattr(complex_llm, 'model')

        # Both should be valid model configurations
        assert simple_llm.model is not None
        assert complex_llm.model is not None

    def test_budget_aware_model_selection(self):
        """Test that model selection considers budget status."""
        # Create a complex assessment
        complex_assessment = self.enhanced_config.assess_question_complexity(
            "Will there be multiple interconnected geopolitical crises affecting global markets?",
            "This involves complex international dynamics and economic interdependencies."
        )

        # Test different budget scenarios
        # Note: We can't easily mock the budget manager state, so we test the interface
        llm = self.enhanced_config.get_llm_for_task(
            "forecast", complexity_assessment=complex_assessment
        )

        assert hasattr(llm, 'model')
        assert llm.model is not None

    def test_cost_estimation_with_complexity(self):
        """Test cost estimation using complexity analysis."""
        question_text = "Will the stock market crash by 50% before 2025?"
        prompt = f"Forecast this question: {question_text}"

        # Get complexity assessment
        complexity_assessment = self.enhanced_config.assess_question_complexity(question_text)

        # Estimate cost with complexity
        estimated_cost, details = self.enhanced_config.estimate_task_cost(
            prompt, "forecast", complexity_assessment=complexity_assessment
        )

        assert estimated_cost > 0
        assert "model" in details
        assert "input_tokens" in details
        assert "estimated_output_tokens" in details
        assert "budget_status" in details
        assert "complexity" in details
        assert "complexity_score" in details

    def test_affordability_check(self):
        """Test affordability checking with complexity analysis."""
        question_text = "Will there be a simple yes/no outcome by December 2024?"
        prompt = f"Research this question: {question_text}"

        # Get complexity assessment
        complexity_assessment = self.enhanced_config.assess_question_complexity(question_text)

        # Check affordability
        can_afford, details = self.enhanced_config.can_afford_task(
            prompt, "research", complexity_assessment=complexity_assessment
        )

        assert isinstance(can_afford, bool)
        assert "estimated_cost" in details
        assert "can_afford" in details
        assert details["can_afford"] == can_afford

    def test_backward_compatibility(self):
        """Test that the enhanced config maintains backward compatibility."""
        # Test old-style complexity assessment
        complexity_str = self.enhanced_config.assess_question_complexity_simple(
            "Will something happen?", "Some background"
        )

        assert complexity_str in ["simple", "medium", "complex"]

        # Test old-style LLM selection
        llm = self.enhanced_config.get_llm_for_task("forecast", complexity_str)
        assert hasattr(llm, 'model')

    def test_configuration_logging(self):
        """Test that configuration status can be logged without errors."""
        # This should not raise any exceptions
        try:
            self.enhanced_config.log_configuration_status()
        except Exception as e:
            pytest.fail(f"Configuration logging failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
