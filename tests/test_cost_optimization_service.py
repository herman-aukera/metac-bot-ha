"""
Tests for Cost Optimization Service.
"""
import pytest
from unittest.mock import Mock, patch
from src.domain.services.cost_optimization_service import (
    CostOptimizationService, TaskPriority, TaskComplexity,
    TaskPrioritizationResult, ModelSelectionResult, ResearchDepthConfig
)
from src.infrastructure.config.operation_modes import OperationMode


class TestCostOptimizationService:
    """Test cases for cost optimization service."""

    def setup_method(self):
        """Setup test fixtures."""
        self.service = CostOptimizationService()

    @patch('src.domain.services.cost_optimization_service.budget_manager')
    def test_optimize_model_selection_normal_mode(self, mock_budget_manager):
        """Test model selection optimization in normal mode."""
        # Mock budget status
        mock_budget_status = Mock()
        mock_budget_status.utilization_percentage = 50.0
        mock_budget_manager.get_budget_status.return_value = mock_budget_status

        result = self.service.optimize_model_selection(
            task_type="forecast",
            original_model="claude-3-5-sonnet",
            operation_mode=OperationMode.NORMAL,
            task_complexity=TaskComplexity.MEDIUM
        )

        assert isinstance(result, ModelSelectionResult)
        assert result.original_model == "claude-3-5-sonnet"
        assert result.selected_model in ["openai/gpt-4o", "claude-3-5-sonnet", "openai/gpt-4o-mini"]

    @patch('src.domain.services.cost_optimization_service.budget_manager')
    def test_optimize_model_selection_emergency_mode(self, mock_budget_manager):
        """Test model selection optimization in emergency mode."""
        # Mock budget status
        mock_budget_status = Mock()
        mock_budget_status.utilization_percentage = 95.0
        mock_budget_manager.get_budget_status.return_value = mock_budget_status

        result = self.service.optimize_model_selection(
            task_type="research",
            original_model="openai/gpt-4o",
            operation_mode=OperationMode.EMERGENCY,
            task_complexity=TaskComplexity.HIGH
        )

        assert isinstance(result, ModelSelectionResult)
        assert result.selected_model in ["openai/gpt-4o-mini", "claude-3-haiku"]
        assert result.cost_reduction >= 0.0

    @patch('src.domain.services.cost_optimization_service.budget_manager')
    def test_prioritize_task_normal_mode(self, mock_budget_manager):
        """Test task prioritization in normal mode."""
        # Mock budget status
        mock_budget_status = Mock()
        mock_budget_status.remaining = 50.0
        mock_budget_manager.get_budget_status.return_value = mock_budget_status

        result = self.service.prioritize_task(
            task_description="Test forecasting task",
            task_priority=TaskPriority.HIGH,
            task_complexity=TaskComplexity.MEDIUM,
            operation_mode=OperationMode.NORMAL,
            estimated_tokens=1000
        )

        assert isinstance(result, TaskPrioritizationResult)
        assert result.should_process is True
        assert result.priority_score > 0.0
        assert "approved" in result.reason.lower()

    @patch('src.domain.services.cost_optimization_service.budget_manager')
    def test_prioritize_task_emergency_mode_low_priority(self, mock_budget_manager):
        """Test task prioritization rejects low priority in emergency mode."""
        # Mock budget status
        mock_budget_status = Mock()
        mock_budget_status.remaining = 5.0
        mock_budget_manager.get_budget_status.return_value = mock_budget_status

        result = self.service.prioritize_task(
            task_description="Low priority task",
            task_priority=TaskPriority.LOW,
            task_complexity=TaskComplexity.MINIMAL,
            operation_mode=OperationMode.EMERGENCY,
            estimated_tokens=500
        )

        assert isinstance(result, TaskPrioritizationResult)
        assert result.should_process is False
        assert "emergency mode" in result.reason.lower()

    def test_adapt_research_depth_normal_mode(self):
        """Test research depth adaptation in normal mode."""
        base_config = {
            "max_sources": 10,
            "max_depth": 3,
            "max_iterations": 5
        }

        result = self.service.adapt_research_depth(
            base_config=base_config,
            operation_mode=OperationMode.NORMAL,
            task_complexity=TaskComplexity.MEDIUM,
            budget_remaining=0.8
        )

        assert isinstance(result, ResearchDepthConfig)
        assert result.max_sources == 10  # No reduction in normal mode
        assert result.max_depth == 3
        assert result.max_iterations == 5
        assert result.enable_deep_analysis is True

    def test_adapt_research_depth_emergency_mode(self):
        """Test research depth adaptation in emergency mode."""
        base_config = {
            "max_sources": 10,
            "max_depth": 3,
            "max_iterations": 5
        }

        result = self.service.adapt_research_depth(
            base_config=base_config,
            operation_mode=OperationMode.EMERGENCY,
            task_complexity=TaskComplexity.HIGH,
            budget_remaining=0.05
        )

        assert isinstance(result, ResearchDepthConfig)
        assert result.max_sources <= 2  # Heavily reduced
        assert result.max_depth <= 1
        assert result.max_iterations <= 1
        assert result.enable_deep_analysis is False

    def test_get_graceful_degradation_strategy_normal(self):
        """Test graceful degradation strategy in normal mode."""
        result = self.service.get_graceful_degradation_strategy(
            operation_mode=OperationMode.NORMAL,
            budget_remaining=0.8
        )

        assert result["complexity_analysis"] is True
        assert result["multi_stage_validation"] is True
        assert result["detailed_logging"] is True
        assert result["retry_attempts"] == 3

    def test_get_graceful_degradation_strategy_emergency(self):
        """Test graceful degradation strategy in emergency mode."""
        result = self.service.get_graceful_degradation_strategy(
            operation_mode=OperationMode.EMERGENCY,
            budget_remaining=0.03
        )

        assert result["complexity_analysis"] is False
        assert result["multi_stage_validation"] is False
        assert result["detailed_logging"] is False
        assert result["retry_attempts"] == 1
        assert result["batch_size"] == 1
        assert result["caching_enabled"] is False  # Disabled for very low budget
