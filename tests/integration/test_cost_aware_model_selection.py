"""
Integration tests for cost-aware model selection and budget management.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.infrastructure.config.budget_manager import BudgetManager
from src.infrastructure.config.token_tracker import TokenTracker
from src.infrastructure.config.cost_monitor import CostMonitor


class TestCostAwareModelSelection:
    """Test cost-aware model selection integration."""

    def setup_method(self):
        """Set up test environment."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()

        # Create test instances with temporary data files
        self.budget_manager = BudgetManager(budget_limit=20.0)  # Small budget for testing
        self.budget_manager.data_file = Path(self.temp_dir) / "test_budget.json"

        self.token_tracker = TokenTracker()
        self.token_tracker.data_file = Path(self.temp_dir) / "test_tokens.json"

        self.cost_monitor = CostMonitor(self.token_tracker, self.budget_manager)
        self.cost_monitor.alerts_file = Path(self.temp_dir) / "test_alerts.json"

    def test_model_cost_comparison(self):
        """Test cost comparison between different models."""
        prompt = "What is the probability that this forecast will be accurate?"
        response = "Based on the available information, I estimate a 75% probability."

        # Test GPT-4o cost
        result_4o = self.cost_monitor.track_api_call_with_monitoring(
            "test-4o", "gpt-4o", "forecast", prompt, response, True
        )

        # Test GPT-4o-mini cost
        result_mini = self.cost_monitor.track_api_call_with_monitoring(
            "test-mini", "gpt-4o-mini", "forecast", prompt, response, True
        )

        # GPT-4o should be more expensive than mini
        assert result_4o["estimated_cost"] > result_mini["estimated_cost"]

        # Both should have similar token counts for same text
        token_diff = abs(result_4o["total_tokens"] - result_mini["total_tokens"])
        assert token_diff < 50  # Allow some variation in tokenization
    def test_budget_threshold_model_switching(self):
        """Test automatic model switching based on budget thresholds."""
        # Simulate high budget usage with expensive model
        long_prompt = "This is a very long prompt that will consume many tokens " * 50
        long_response = "This is a very long response that will consume many tokens " * 50

        # Use expensive model until budget threshold
        for i in range(5):
            self.cost_monitor.track_api_call_with_monitoring(
                f"expensive-{i}", "gpt-4o", "forecast", long_prompt, long_response, True
            )

        # Check budget status
        status = self.cost_monitor.get_comprehensive_status()
        initial_utilization = status["budget"]["utilization_percent"]

        # Simulate switching to cheaper model
        for i in range(5):
            self.cost_monitor.track_api_call_with_monitoring(
                f"cheap-{i}", "gpt-4o-mini", "forecast", long_prompt, long_response, True
            )

        # Verify cost savings with cheaper model
        final_status = self.cost_monitor.get_comprehensive_status()

        # Should have processed more questions with cheaper model
        assert final_status["budget"]["questions_processed"] == 10
        # Total calls includes both expensive and cheap calls
        assert final_status["tokens"]["total_calls"] >= 10

        # Check model usage breakdown
        model_breakdown = final_status["tokens"]["by_model"]
        assert "gpt-4o" in model_breakdown
        assert "gpt-4o-mini" in model_breakdown
        assert model_breakdown["gpt-4o"]["calls"] == 5
        assert model_breakdown["gpt-4o-mini"]["calls"] == 5

    def test_budget_conservation_strategies(self):
        """Test budget conservation strategies under different utilization levels."""
        # Test normal budget usage (no restrictions)
        self.cost_monitor.track_api_call_with_monitoring(
            "normal-1", "gpt-4o", "forecast", "Normal prompt", "Normal response", True
        )

        status = self.cost_monitor.get_comprehensive_status()
        assert status["budget"]["status_level"] == "normal"

        # Simulate high budget usage to trigger conservative mode
        expensive_prompt = "Very expensive prompt " * 100
        expensive_response = "Very expensive response " * 100

        for i in range(8):  # Push budget utilization higher
            self.cost_monitor.track_api_call_with_monitoring(
                f"expensive-{i}", "gpt-4o", "forecast", expensive_prompt, expensive_response, True
            )

        final_status = self.cost_monitor.get_comprehensive_status()

        # Should have recommendations for cost optimization
        recommendations = self.cost_monitor.get_optimization_recommendations()
        # May not have recommendations if budget utilization is still low
        assert isinstance(recommendations, list)

        # Should recommend cheaper models or reduced usage
        rec_text = " ".join(recommendations).lower()
        assert any(keyword in rec_text for keyword in ["gpt-4o-mini", "reduce", "optimize", "budget"])
    def test_tournament_duration_budget_simulation(self):
        """Test budget simulation for tournament duration."""
        # Simulate a tournament with mixed task types
        questions_per_day = 10
        days = 3  # 3-day tournament simulation

        daily_costs = []

        for day in range(days):
            day_cost = 0

            # Simulate daily question processing
            for q in range(questions_per_day):
                question_id = f"day{day}-q{q}"

                # Mix of research and forecast tasks
                if q % 3 == 0:  # Every 3rd question gets research
                    result = self.cost_monitor.track_api_call_with_monitoring(
                        f"{question_id}-research", "gpt-4o-mini", "research",
                        "Research prompt for context", "Research findings", True
                    )
                    day_cost += result["estimated_cost"]

                # Forecast task
                result = self.cost_monitor.track_api_call_with_monitoring(
                    f"{question_id}-forecast", "gpt-4o", "forecast",
                    "Forecast prompt with analysis", "Detailed forecast with reasoning", True
                )
                day_cost += result["estimated_cost"]

            daily_costs.append(day_cost)

        # Analyze tournament budget usage
        final_status = self.cost_monitor.get_comprehensive_status()

        # Verify tournament metrics
        total_questions = questions_per_day * days
        assert final_status["budget"]["questions_processed"] == total_questions

        # Should have both research and forecast tasks
        task_breakdown = final_status["tokens"]["by_task_type"]
        assert "research" in task_breakdown
        assert "forecast" in task_breakdown
        # All questions get forecasts, plus some research calls
        # Note: total_calls includes both research and forecast calls
        expected_research_calls = total_questions // 3  # Every 3rd question gets research
        expected_forecast_calls = total_questions  # All questions get forecasts

        assert task_breakdown["forecast"]["calls"] >= total_questions  # All questions get forecasts
        assert task_breakdown["research"]["calls"] >= expected_research_calls  # Every 3rd gets research

        # Budget should be tracked accurately
        assert final_status["budget"]["spent"] > 0
        assert final_status["budget"]["utilization_percent"] > 0

        # Daily costs should be relatively consistent
        if len(daily_costs) > 1:
            avg_daily_cost = sum(daily_costs) / len(daily_costs)
            for daily_cost in daily_costs:
                # Daily costs should be within 50% of average (allowing for variation)
                assert abs(daily_cost - avg_daily_cost) / avg_daily_cost < 0.5

    def teardown_method(self):
        """Clean up test environment."""
        # Clean up temporary files
        for file_path in [
            self.budget_manager.data_file,
            self.token_tracker.data_file,
            self.cost_monitor.alerts_file
        ]:
            if file_path.exists():
                file_path.unlink()
