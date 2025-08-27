"""
Tests for enhanced TokenTracker with real-time cost calculation and monitoring.
"""
import pytest
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.infrastructure.config.token_tracker import TokenTracker, TokenUsageRecord


class TestEnhancedTokenTracker:
    """Test enhanced TokenTracker functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_file = Path(self.temp_dir) / "test_token_usage.json"

        # Create tracker with test data file
        self.tracker = TokenTracker()
        self.tracker.data_file = self.test_data_file

    def test_real_time_cost_calculation(self):
        """Test real-time cost calculation for different models."""
        # Test GPT-4o cost calculation
        cost_4o = self.tracker.calculate_real_time_cost("gpt-4o", 1000, 500)
        expected_4o = (1000 * 0.0025 / 1000) + (500 * 0.01 / 1000)  # $0.0025 + $0.005 = $0.0075
        assert abs(cost_4o - expected_4o) < 0.0001

        # Test GPT-4o-mini cost calculation
        cost_mini = self.tracker.calculate_real_time_cost("gpt-4o-mini", 1000, 500)
        expected_mini = (1000 * 0.00015 / 1000) + (500 * 0.0006 / 1000)  # $0.00015 + $0.0003 = $0.00045
        assert abs(cost_mini - expected_mini) < 0.0001

        # Test unknown model defaults to GPT-4o pricing
        cost_unknown = self.tracker.calculate_real_time_cost("unknown-model", 1000, 500)
        assert abs(cost_unknown - expected_4o) < 0.0001

    def test_api_call_tracking(self):
        """Test comprehensive API call tracking."""
        # Track a successful API call
        record = self.tracker.track_api_call(
            question_id="test-123",
            model="gpt-4o-mini",
            task_type="research",
            input_tokens=800,
            output_tokens=200,
            success=True
        )

        # Verify record creation
        assert isinstance(record, TokenUsageRecord)
        assert record.question_id == "test-123"
        assert record.model_used == "gpt-4o-mini"
        assert record.task_type == "research"
        assert record.input_tokens == 800
        assert record.output_tokens == 200
        assert record.total_tokens == 1000
        assert record.success is True
        assert record.estimated_cost > 0

        # Verify tracking updates
        assert len(self.tracker.usage_records) == 1
        assert self.tracker.total_tokens_used["input"] == 800
        assert self.tracker.total_tokens_used["output"] == 200
        assert self.tracker.total_tokens_used["total"] == 1000
        assert self.tracker.total_estimated_cost == record.estimated_cost

    def test_failed_api_call_tracking(self):
        """Test tracking of failed API calls."""
        initial_totals = self.tracker.total_tokens_used.copy()
        initial_cost = self.tracker.total_estimated_cost

        # Track a failed API call
        record = self.tracker.track_api_call(
            question_id="test-failed",
            model="gpt-4o",
            task_type="forecast",
            input_tokens=1000,
            output_tokens=500,
            success=False
        )

        # Verify record is created but totals are not updated for failed calls
        assert len(self.tracker.usage_records) == 1
        assert record.success is False
        assert self.tracker.total_tokens_used == initial_totals  # Should not change
        assert self.tracker.total_estimated_cost == initial_cost  # Should not change

    def test_usage_summary_generation(self):
        """Test comprehensive usage summary generation."""
        # Add multiple records
        self.tracker.track_api_call("q1", "gpt-4o", "research", 500, 300, True)
        self.tracker.track_api_call("q2", "gpt-4o-mini", "forecast", 800, 200, True)
        self.tracker.track_api_call("q3", "gpt-4o", "research", 600, 400, False)  # Failed

        summary = self.tracker.get_usage_summary()

        # Verify overall summary
        assert summary["total_calls"] == 3
        assert summary["success_rate"] == 2/3  # 2 successful out of 3

        # Verify by model breakdown
        assert "gpt-4o" in summary["by_model"]
        assert "gpt-4o-mini" in summary["by_model"]

        gpt4o_stats = summary["by_model"]["gpt-4o"]
        assert gpt4o_stats["calls"] == 2  # 2 calls to gpt-4o
        assert gpt4o_stats["tokens"]["total"] == 800  # Only successful call counted (500+300)

        mini_stats = summary["by_model"]["gpt-4o-mini"]
        assert mini_stats["calls"] == 1
        assert mini_stats["tokens"]["total"] == 1000  # 800+200

        # Verify by task type breakdown
        assert "research" in summary["by_task_type"]
        assert "forecast" in summary["by_task_type"]

    def test_cost_efficiency_metrics(self):
        """Test cost efficiency metrics calculation."""
        # Add records with different costs
        self.tracker.track_api_call("q1", "gpt-4o", "research", 1000, 500, True)
        self.tracker.track_api_call("q2", "gpt-4o-mini", "research", 1000, 500, True)

        metrics = self.tracker.get_cost_efficiency_metrics()

        # Verify metrics structure
        assert "average_cost_per_call" in metrics
        assert "average_tokens_per_call" in metrics
        assert "cost_per_token" in metrics
        assert "model_efficiency" in metrics
        assert "task_efficiency" in metrics

        # Verify model efficiency comparison
        gpt4o_efficiency = metrics["model_efficiency"]["gpt-4o"]["cost_per_token"]
        mini_efficiency = metrics["model_efficiency"]["gpt-4o-mini"]["cost_per_token"]

        # GPT-4o should be more expensive per token than mini
        assert gpt4o_efficiency > mini_efficiency

    def test_data_persistence(self):
        """Test saving and loading of usage data."""
        # Add some records
        self.tracker.track_api_call("q1", "gpt-4o", "research", 500, 300, True)
        self.tracker.track_api_call("q2", "gpt-4o-mini", "forecast", 800, 200, True)

        # Force save
        self.tracker._save_data()

        # Verify file was created
        assert self.tracker.data_file.exists()

        # Create new tracker and load data
        new_tracker = TokenTracker()
        new_tracker.data_file = self.test_data_file
        new_tracker._load_existing_data()

        # Verify data was loaded correctly
        assert len(new_tracker.usage_records) == 2
        assert new_tracker.total_tokens_used["total"] == 1800  # 800 + 1000
        assert new_tracker.total_estimated_cost > 0

    def test_actual_usage_tracking_with_strings(self):
        """Test tracking actual usage from prompt and response strings."""
        prompt = "What is the probability that AI will achieve AGI by 2030?"
        response = "Based on current trends and expert opinions, I estimate a 25% probability."

        result = self.tracker.track_actual_usage(
            prompt=prompt,
            response=response,
            model="gpt-4o-mini",
            question_id="agi-2030",
            task_type="forecast"
        )

        # Verify result structure
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert "total_tokens" in result
        assert "estimated_cost" in result
        assert "record" in result

        # Verify tokens were counted
        assert result["input_tokens"] > 0
        assert result["output_tokens"] > 0
        assert result["total_tokens"] == result["input_tokens"] + result["output_tokens"]

        # Verify record was created and tracked
        assert len(self.tracker.usage_records) == 1
        record = self.tracker.usage_records[0]
        assert record.question_id == "agi-2030"
        assert record.task_type == "forecast"

    def test_model_name_normalization(self):
        """Test model name normalization for cost calculation."""
        # Test provider prefix removal
        assert self.tracker._normalize_model_name("openai/gpt-4o") == "gpt-4o"
        assert self.tracker._normalize_model_name("anthropic/claude-3-5-sonnet") == "claude-3-5-sonnet"

        # Test direct model names
        assert self.tracker._normalize_model_name("gpt-4o-mini") == "gpt-4o-mini"
        assert self.tracker._normalize_model_name("claude-3-haiku") == "claude-3-haiku"

    def test_reset_tracking(self):
        """Test resetting tracking data."""
        # Add some data
        self.tracker.track_api_call("q1", "gpt-4o", "research", 500, 300, True)
        assert len(self.tracker.usage_records) == 1
        assert self.tracker.total_estimated_cost > 0

        # Reset tracking
        self.tracker.reset_tracking()

        # Verify everything is reset
        assert len(self.tracker.usage_records) == 0
        assert self.tracker.total_tokens_used == {"input": 0, "output": 0, "total": 0}
        assert self.tracker.total_estimated_cost == 0.0

    def test_logging_methods(self):
        """Test logging methods don't crash."""
        # Add some data
        self.tracker.track_api_call("q1", "gpt-4o", "research", 500, 300, True)
        self.tracker.track_api_call("q2", "gpt-4o-mini", "forecast", 800, 200, True)

        # Test logging methods (should not raise exceptions)
        with patch('src.infrastructure.config.token_tracker.logger') as mock_logger:
            self.tracker.log_usage_summary()
            assert mock_logger.info.called

    def test_empty_data_handling(self):
        """Test handling of empty data scenarios."""
        # Test empty usage summary
        summary = self.tracker.get_usage_summary()
        assert summary["total_calls"] == 0
        assert summary["total_cost"] == 0.0
        assert summary["success_rate"] == 0.0

        # Test empty efficiency metrics
        metrics = self.tracker.get_cost_efficiency_metrics()
        assert "error" in metrics

    def teardown_method(self):
        """Clean up test environment."""
        # Clean up temporary files
        if self.test_data_file.exists():
            self.test_data_file.unlink()
