"""
Integration tests for the enhanced cost tracking system.
Tests integration with existing forecasting components.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.config.budget_manager import BudgetManager
from src.infrastructure.config.cost_monitor import CostMonitor
from src.infrastructure.config.token_tracker import TokenTracker


class TestCostTrackingIntegration:
    """Test cost tracking system integration."""

    def setup_method(self):
        """Set up test environment."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()

        # Create test instances with temporary data files
        self.token_tracker = TokenTracker()
        self.token_tracker.data_file = Path(self.temp_dir) / "test_tokens.json"
        # Reset to clean state
        self.token_tracker.usage_records = []
        self.token_tracker.total_tokens_used = {"input": 0, "output": 0, "total": 0}
        self.token_tracker.total_estimated_cost = 0.0

        self.budget_manager = BudgetManager(budget_limit=100.0)
        self.budget_manager.data_file = Path(self.temp_dir) / "test_budget.json"
        # Reset to clean state
        self.budget_manager.current_spend = 0.0
        self.budget_manager.questions_processed = 0
        self.budget_manager.cost_records = []

        self.cost_monitor = CostMonitor(self.token_tracker, self.budget_manager)
        self.cost_monitor.alerts_file = Path(self.temp_dir) / "test_alerts.json"
        # Reset to clean state
        self.cost_monitor.alerts = []

    def test_full_forecasting_workflow_tracking(self):
        """Test cost tracking through a complete forecasting workflow."""
        # Simulate a complete forecasting workflow
        question_id = "integration-test-question"

        # Step 1: Research phase
        research_prompt = """
        Research the following forecasting question:
        Will renewable energy account for more than 50% of global electricity generation by 2030?

        Please find:
        1. Current renewable energy statistics
        2. Growth trends and projections
        3. Policy initiatives and targets
        4. Technical and economic factors
        """

        research_response = """
        Current Status (2024):
        - Renewables account for ~30% of global electricity generation
        - Solar and wind are fastest growing sources
        - Hydroelectric remains largest renewable source

        Growth Trends:
        - Annual growth rate of 8-12% for solar/wind
        - Declining costs making renewables competitive
        - Grid integration challenges being addressed

        Policy Support:
        - Paris Agreement commitments driving adoption
        - Many countries have 2030 renewable targets
        - Significant investment in clean energy infrastructure

        Assessment: Strong momentum but 50% by 2030 is ambitious given current trajectory.
        """

        # Track research phase
        research_result = self.cost_monitor.track_api_call_with_monitoring(
            question_id=question_id,
            model="gpt-4o-mini",
            task_type="research",
            prompt=research_prompt,
            response=research_response,
            success=True,
        )

        # Step 2: Forecasting phase
        forecast_prompt = """
        Based on the research provided, make a probability forecast:

        Question: Will renewable energy account for more than 50% of global electricity generation by 2030?

        Research Summary: [research_response content]

        Please provide:
        1. Analysis of key factors
        2. Scenario consideration
        3. Probability estimate with reasoning
        4. Confidence assessment
        """

        forecast_response = """
        Analysis:
        Current trajectory shows strong growth but faces challenges:
        - Need to double renewable share in 6 years
        - Grid infrastructure limitations
        - Intermittency challenges
        - Policy implementation gaps

        Scenarios:
        - Optimistic: Accelerated deployment, breakthrough storage tech (65% chance)
        - Base case: Continued growth at current pace (45% chance)
        - Pessimistic: Economic/political headwinds slow progress (35% chance)

        Forecast: 45% probability

        Confidence: Medium (significant uncertainty in policy implementation and tech development)
        """

        # Track forecasting phase
        forecast_result = self.cost_monitor.track_api_call_with_monitoring(
            question_id=question_id,
            model="gpt-4o",
            task_type="forecast",
            prompt=forecast_prompt,
            response=forecast_response,
            success=True,
        )

        # Verify tracking results
        assert research_result["success"] is True
        assert forecast_result["success"] is True
        assert research_result["estimated_cost"] > 0
        assert forecast_result["estimated_cost"] > 0

        # Verify data was tracked in all components
        assert len(self.token_tracker.usage_records) == 2
        assert len(self.budget_manager.cost_records) == 2
        assert self.budget_manager.questions_processed == 1  # Only forecast counts

        # Verify cost tracking consistency
        total_token_cost = sum(
            r.estimated_cost for r in self.token_tracker.usage_records
        )
        total_budget_cost = self.budget_manager.current_spend
        assert abs(total_token_cost - total_budget_cost) < 0.0001

        # Get comprehensive status
        status = self.cost_monitor.get_comprehensive_status()

        # Verify status completeness
        assert status["budget"]["questions_processed"] == 1
        assert status["tokens"]["total_calls"] == 2
        assert status["tokens"]["by_task_type"]["research"]["calls"] == 1
        assert status["tokens"]["by_task_type"]["forecast"]["calls"] == 1
        assert status["tokens"]["by_model"]["gpt-4o-mini"]["calls"] == 1
        assert status["tokens"]["by_model"]["gpt-4o"]["calls"] == 1

    def test_cost_optimization_workflow(self):
        """Test cost optimization recommendations in a realistic scenario."""
        # Simulate high-cost usage pattern
        expensive_scenarios = [
            ("climate-q1", "gpt-4o", "research", "Long research prompt " * 100),
            ("climate-q1", "gpt-4o", "forecast", "Long forecast prompt " * 80),
            ("tech-q2", "gpt-4o", "research", "Another long research " * 120),
            ("tech-q2", "gpt-4o", "forecast", "Another long forecast " * 90),
            ("econ-q3", "gpt-4o", "research", "Economic analysis " * 110),
            ("econ-q3", "gpt-4o", "forecast", "Economic forecast " * 85),
        ]

        # Track all scenarios
        for question_id, model, task_type, prompt in expensive_scenarios:
            response = f"Detailed {task_type} response for {question_id} " * 50

            self.cost_monitor.track_api_call_with_monitoring(
                question_id=question_id,
                model=model,
                task_type=task_type,
                prompt=prompt,
                response=response,
                success=True,
            )

        # Get optimization recommendations
        recommendations = self.cost_monitor.get_optimization_recommendations()

        # Should have recommendations due to high GPT-4o usage
        assert isinstance(recommendations, list)

        # Check budget utilization
        status = self.cost_monitor.get_comprehensive_status()
        budget_utilization = status["budget"]["utilization_percent"]

        # Should be tracking significant usage
        assert budget_utilization > 0
        assert status["tokens"]["total_calls"] == 6
        assert status["budget"]["questions_processed"] == 3  # 3 forecast tasks

    def test_error_handling_and_recovery(self):
        """Test error handling in cost tracking system."""
        # Test failed API call tracking
        failed_result = self.cost_monitor.track_api_call_with_monitoring(
            question_id="failed-test",
            model="gpt-4o",
            task_type="forecast",
            prompt="Test prompt",
            response="",  # Empty response indicates failure
            success=False,
        )

        assert failed_result["success"] is False

        # Verify failed calls are tracked but don't affect totals incorrectly
        assert len(self.token_tracker.usage_records) == 1
        assert len(self.budget_manager.cost_records) == 1

        # Failed forecast shouldn't count as processed question
        assert self.budget_manager.questions_processed == 0

        # Test recovery with successful call
        success_result = self.cost_monitor.track_api_call_with_monitoring(
            question_id="recovery-test",
            model="gpt-4o-mini",
            task_type="forecast",
            prompt="Recovery test prompt",
            response="Successful recovery response",
            success=True,
        )

        assert success_result["success"] is True
        assert self.budget_manager.questions_processed == 1

    def test_data_persistence_across_sessions(self):
        """Test data persistence and loading across sessions."""
        # Add some data
        self.cost_monitor.track_api_call_with_monitoring(
            "persist-test",
            "gpt-4o-mini",
            "research",
            "Test prompt",
            "Test response",
            True,
        )

        # Force save data
        self.token_tracker._save_data()
        self.budget_manager._save_data()

        # Create new instances and load data
        new_token_tracker = TokenTracker()
        new_token_tracker.data_file = self.token_tracker.data_file
        new_token_tracker._load_existing_data()

        new_budget_manager = BudgetManager(budget_limit=100.0)
        new_budget_manager.data_file = self.budget_manager.data_file
        new_budget_manager._load_existing_data()

        # Verify data was loaded
        assert len(new_token_tracker.usage_records) == 1
        assert len(new_budget_manager.cost_records) == 1
        assert new_token_tracker.total_estimated_cost > 0
        assert new_budget_manager.current_spend > 0

    def test_model_selection_cost_impact(self):
        """Test cost impact of different model selections."""
        # Same content with different models
        prompt = "Analyze this forecasting question in detail " * 30
        response = "Detailed analysis with comprehensive reasoning " * 25

        models_to_test = [
            ("gpt-4o-mini", "research"),
            ("gpt-4o", "forecast"),
            ("claude-3-haiku", "research"),
            ("claude-3-5-sonnet", "forecast"),
        ]

        costs = {}

        for model, task_type in models_to_test:
            result = self.cost_monitor.track_api_call_with_monitoring(
                question_id=f"model-test-{model}",
                model=model,
                task_type=task_type,
                prompt=prompt,
                response=response,
                success=True,
            )

            costs[model] = result["estimated_cost"]

        # Verify cost differences
        assert costs["gpt-4o-mini"] < costs["gpt-4o"]
        assert costs["claude-3-haiku"] < costs["claude-3-5-sonnet"]

        # Get efficiency metrics
        metrics = self.token_tracker.get_cost_efficiency_metrics()
        assert "model_efficiency" in metrics

        # Verify efficiency tracking
        for model in ["gpt-4o-mini", "gpt-4o", "claude-3-haiku", "claude-3-5-sonnet"]:
            if model in metrics["model_efficiency"]:
                efficiency = metrics["model_efficiency"][model]
                assert "cost_per_token" in efficiency
                assert "tokens_per_call" in efficiency
                assert efficiency["cost_per_token"] > 0

    def teardown_method(self):
        """Clean up test environment."""
        # Clean up temporary files
        for file_path in [
            self.token_tracker.data_file,
            self.budget_manager.data_file,
            self.cost_monitor.alerts_file,
        ]:
            if file_path.exists():
                file_path.unlink()
