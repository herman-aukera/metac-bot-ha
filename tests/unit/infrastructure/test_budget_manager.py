"""
Tests for BudgetManager budget tracking and cost management.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.config.budget_manager import (
    BudgetManager,
    BudgetStatus,
    CostTrackingRecord,
)


class TestBudgetManager:
    """Test BudgetManager functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_file = Path(self.temp_dir) / "test_budget.json"

        # Create budget manager with test data file
        self.budget_manager = BudgetManager(budget_limit=50.0)
        self.budget_manager.data_file = self.test_data_file

    def test_initialization(self):
        """Test BudgetManager initialization."""
        assert self.budget_manager.budget_limit == 50.0
        assert self.budget_manager.current_spend == 0.0
        assert self.budget_manager.questions_processed == 0
        assert len(self.budget_manager.cost_records) == 0
        assert "gpt-4o" in self.budget_manager.cost_per_token
        assert "gpt-4o-mini" in self.budget_manager.cost_per_token

    def test_cost_estimation(self):
        """Test cost estimation for different models."""
        # Test GPT-4o cost estimation
        cost_4o = self.budget_manager.estimate_cost("gpt-4o", 1000, 500)
        expected_4o = (1000 * 0.0025 / 1000) + (
            500 * 0.01 / 1000
        )  # $0.0025 + $0.005 = $0.0075
        assert abs(cost_4o - expected_4o) < 0.0001

        # Test GPT-4o-mini cost estimation
        cost_mini = self.budget_manager.estimate_cost("gpt-4o-mini", 1000, 500)
        expected_mini = (1000 * 0.00015 / 1000) + (
            500 * 0.0006 / 1000
        )  # $0.00015 + $0.0003 = $0.00045
        assert abs(cost_mini - expected_mini) < 0.0001

        # Test unknown model defaults to GPT-4o pricing
        cost_unknown = self.budget_manager.estimate_cost("unknown-model", 1000, 500)
        assert abs(cost_unknown - expected_4o) < 0.0001

    def test_model_name_normalization(self):
        """Test model name normalization for cost lookup."""
        # Test provider prefix removal
        assert self.budget_manager._normalize_model_name("openai/gpt-4o") == "gpt-4o"
        assert (
            self.budget_manager._normalize_model_name("anthropic/claude-3-5-sonnet")
            == "claude-3-5-sonnet"
        )

        # Test direct model names
        assert self.budget_manager._normalize_model_name("gpt-4o-mini") == "gpt-4o-mini"
        assert (
            self.budget_manager._normalize_model_name("claude-3-haiku")
            == "claude-3-haiku"
        )

    def test_can_afford_check(self):
        """Test budget affordability checks."""
        # Should be able to afford small costs initially
        assert self.budget_manager.can_afford(1.0) is True
        assert self.budget_manager.can_afford(10.0) is True

        # Simulate spending most of budget
        self.budget_manager.current_spend = 45.0  # 90% of $50 budget

        # Should not afford large costs near budget limit (95% safety margin)
        assert self.budget_manager.can_afford(5.0) is False  # Would exceed 95% limit
        assert self.budget_manager.can_afford(2.0) is True  # Still within 95% limit

    def test_cost_recording(self):
        """Test recording actual API call costs."""
        initial_spend = self.budget_manager.current_spend
        initial_questions = self.budget_manager.questions_processed

        # Record a successful forecast cost
        cost = self.budget_manager.record_cost(
            question_id="test-123",
            model="gpt-4o-mini",
            input_tokens=800,
            output_tokens=200,
            task_type="forecast",
            success=True,
        )

        # Verify cost calculation and recording
        expected_cost = (800 * 0.00015 / 1000) + (200 * 0.0006 / 1000)
        assert abs(cost - expected_cost) < 0.0001
        assert self.budget_manager.current_spend == initial_spend + cost
        assert self.budget_manager.questions_processed == initial_questions + 1
        assert len(self.budget_manager.cost_records) == 1

        # Verify record details
        record = self.budget_manager.cost_records[0]
        assert record.question_id == "test-123"
        assert record.model_used == "gpt-4o-mini"
        assert record.task_type == "forecast"
        assert record.success is True

    def test_research_task_recording(self):
        """Test that research tasks don't increment question count."""
        initial_questions = self.budget_manager.questions_processed

        # Record a successful research cost
        self.budget_manager.record_cost(
            question_id="research-123",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            task_type="research",
            success=True,
        )

        # Questions processed should not increment for research tasks
        assert self.budget_manager.questions_processed == initial_questions
        assert len(self.budget_manager.cost_records) == 1

    def test_failed_call_recording(self):
        """Test recording failed API calls."""
        initial_spend = self.budget_manager.current_spend
        initial_questions = self.budget_manager.questions_processed

        # Record a failed forecast
        cost = self.budget_manager.record_cost(
            question_id="failed-123",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=0,  # No output for failed call
            task_type="forecast",
            success=False,
        )

        # Cost should still be calculated and recorded
        assert cost > 0
        assert self.budget_manager.current_spend == initial_spend + cost
        assert (
            self.budget_manager.questions_processed == initial_questions
        )  # No increment for failed forecast
        assert len(self.budget_manager.cost_records) == 1

        # Verify record details
        record = self.budget_manager.cost_records[0]
        assert record.success is False

    def test_budget_status_generation(self):
        """Test budget status report generation."""
        # Add some spending
        self.budget_manager.record_cost("q1", "gpt-4o", 1000, 500, "forecast", True)
        self.budget_manager.record_cost("q2", "gpt-4o-mini", 800, 200, "forecast", True)

        status = self.budget_manager.get_budget_status()

        # Verify status structure and values
        assert isinstance(status, BudgetStatus)
        assert status.total_budget == 50.0
        assert status.spent > 0
        assert status.remaining == 50.0 - status.spent
        assert status.utilization_percentage == (status.spent / 50.0) * 100
        assert status.questions_processed == 2
        assert status.average_cost_per_question > 0
        assert status.estimated_questions_remaining > 0
        assert status.status_level in ["normal", "conservative", "emergency"]

    def test_status_level_determination(self):
        """Test budget status level determination."""
        # Test normal status (under 80%)
        self.budget_manager.current_spend = 30.0  # 60% of $50
        status = self.budget_manager.get_budget_status()
        assert status.status_level == "normal"

        # Test conservative status (80-95%)
        self.budget_manager.current_spend = 42.5  # 85% of $50
        status = self.budget_manager.get_budget_status()
        assert status.status_level == "conservative"

        # Test emergency status (95%+)
        self.budget_manager.current_spend = 48.0  # 96% of $50
        status = self.budget_manager.get_budget_status()
        assert status.status_level == "emergency"

    def test_budget_alert_checks(self):
        """Test budget usage alert conditions."""
        # Should not alert initially
        assert self.budget_manager.should_alert_budget_usage() is False
        assert self.budget_manager.get_budget_alert_level() == "NORMAL"

        # Should alert at 80% usage
        self.budget_manager.current_spend = 40.0  # 80% of $50
        assert self.budget_manager.should_alert_budget_usage() is True
        assert self.budget_manager.get_budget_alert_level() == "WARNING"

        # Should escalate alert levels
        self.budget_manager.current_spend = 45.0  # 90% of $50
        assert self.budget_manager.get_budget_alert_level() == "HIGH"

        self.budget_manager.current_spend = 47.5  # 95% of $50
        assert self.budget_manager.get_budget_alert_level() == "CRITICAL"

    def test_cost_breakdown_analysis(self):
        """Test detailed cost breakdown generation."""
        # Add varied spending data
        self.budget_manager.record_cost("q1", "gpt-4o", 1000, 500, "forecast", True)
        self.budget_manager.record_cost("q2", "gpt-4o-mini", 800, 200, "research", True)
        self.budget_manager.record_cost("q3", "gpt-4o", 1200, 600, "forecast", True)

        breakdown = self.budget_manager.get_cost_breakdown()

        # Verify breakdown structure
        assert "by_model" in breakdown
        assert "by_task_type" in breakdown
        assert "by_day" in breakdown
        assert "total_tokens" in breakdown

        # Verify model breakdown
        assert "gpt-4o" in breakdown["by_model"]
        assert "gpt-4o-mini" in breakdown["by_model"]
        assert breakdown["by_model"]["gpt-4o"]["calls"] == 2
        assert breakdown["by_model"]["gpt-4o-mini"]["calls"] == 1

        # Verify task type breakdown
        assert "forecast" in breakdown["by_task_type"]
        assert "research" in breakdown["by_task_type"]
        assert breakdown["by_task_type"]["forecast"]["calls"] == 2
        assert breakdown["by_task_type"]["research"]["calls"] == 1

        # Verify token totals
        assert breakdown["total_tokens"]["input"] == 3000  # 1000+800+1200
        assert breakdown["total_tokens"]["output"] == 1300  # 500+200+600

    def test_data_persistence(self):
        """Test saving and loading budget data."""
        # Add some data
        self.budget_manager.record_cost("q1", "gpt-4o", 1000, 500, "forecast", True)
        self.budget_manager.record_cost("q2", "gpt-4o-mini", 800, 200, "research", True)

        # Force save
        self.budget_manager._save_data()

        # Verify file was created
        assert self.budget_manager.data_file.exists()

        # Create new budget manager and load data
        new_manager = BudgetManager(budget_limit=50.0)
        new_manager.data_file = self.test_data_file
        new_manager._load_existing_data()

        # Verify data was loaded correctly
        assert len(new_manager.cost_records) == 2
        assert new_manager.current_spend > 0
        assert new_manager.questions_processed == 1  # Only forecast tasks count

    def test_budget_reset(self):
        """Test budget reset functionality."""
        # Add some data
        self.budget_manager.record_cost("q1", "gpt-4o", 1000, 500, "forecast", True)
        assert self.budget_manager.current_spend > 0
        assert len(self.budget_manager.cost_records) == 1

        # Reset budget
        self.budget_manager.reset_budget(new_limit=75.0)

        # Verify reset
        assert self.budget_manager.budget_limit == 75.0
        assert self.budget_manager.current_spend == 0.0
        assert self.budget_manager.questions_processed == 0
        assert len(self.budget_manager.cost_records) == 0

    def test_logging_methods(self):
        """Test logging methods don't crash."""
        # Add some data
        self.budget_manager.record_cost("q1", "gpt-4o", 1000, 500, "forecast", True)

        # Test logging methods (should not raise exceptions)
        with patch("src.infrastructure.config.budget_manager.logger") as mock_logger:
            self.budget_manager.log_budget_status()
            assert mock_logger.info.called

    def teardown_method(self):
        """Clean up test environment."""
        # Clean up temporary files
        if self.test_data_file.exists():
            self.test_data_file.unlink()
