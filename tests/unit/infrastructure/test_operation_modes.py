"""
Unit tests for operation modes functionality.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.infrastructure.config.operation_modes import (
    ModeTransition,
    OperationMode,
    OperationModeConfig,
    OperationModeManager,
)


class TestOperationModeManager:
    """Test operation mode manager functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        with patch(
            "src.infrastructure.config.operation_modes.budget_manager"
        ) as mock_budget:
            mock_budget.get_budget_status.return_value = Mock(
                utilization_percentage=50.0, remaining=50.0
            )
            self.manager = OperationModeManager()

    def test_initialization(self):
        """Test operation mode manager initialization."""
        assert self.manager.current_mode == OperationMode.NORMAL
        assert len(self.manager.mode_configs) == 3
        assert OperationMode.NORMAL in self.manager.mode_configs
        assert OperationMode.CONSERVATIVE in self.manager.mode_configs
        assert OperationMode.EMERGENCY in self.manager.mode_configs

    def test_mode_configurations(self):
        """Test mode configurations are properly set."""
        normal_config = self.manager.mode_configs[OperationMode.NORMAL]
        assert normal_config.max_questions_per_batch == 10
        assert normal_config.enable_complexity_analysis is True
        assert normal_config.skip_low_priority_questions is False

        conservative_config = self.manager.mode_configs[OperationMode.CONSERVATIVE]
        assert conservative_config.budget_threshold == 0.80
        assert conservative_config.max_questions_per_batch == 5
        assert conservative_config.skip_low_priority_questions is True

        emergency_config = self.manager.mode_configs[OperationMode.EMERGENCY]
        assert emergency_config.budget_threshold == 0.95
        assert emergency_config.max_questions_per_batch == 2
        assert emergency_config.enable_complexity_analysis is False

    def test_determine_mode_from_utilization(self):
        """Test mode determination based on budget utilization."""
        assert (
            self.manager._determine_mode_from_utilization(0.5) == OperationMode.NORMAL
        )
        assert (
            self.manager._determine_mode_from_utilization(0.85)
            == OperationMode.CONSERVATIVE
        )
        assert (
            self.manager._determine_mode_from_utilization(0.97)
            == OperationMode.EMERGENCY
        )

    @patch("src.infrastructure.config.operation_modes.budget_manager")
    def test_check_and_update_mode_no_change(self, mock_budget):
        """Test mode check when no change is needed."""
        mock_budget.get_budget_status.return_value = Mock(utilization_percentage=50.0)

        changed, transition = self.manager.check_and_update_mode()

        assert changed is False
        assert transition is None
        assert self.manager.current_mode == OperationMode.NORMAL

    def test_check_and_update_mode_with_change(self):
        """Test mode change when budget threshold is exceeded."""
        with patch.object(
            self.manager.budget_manager, "get_budget_status"
        ) as mock_status:
            mock_status.return_value = Mock(utilization_percentage=85.0)

            changed, transition = self.manager.check_and_update_mode()

            assert changed is True
            assert transition is not None
            assert transition.from_mode == OperationMode.NORMAL
            assert transition.to_mode == OperationMode.CONSERVATIVE
            assert self.manager.current_mode == OperationMode.CONSERVATIVE

    def test_force_mode_transition(self):
        """Test manual mode transition."""
        transition = self.manager.force_mode_transition(
            OperationMode.EMERGENCY, "testing"
        )

        assert self.manager.current_mode == OperationMode.EMERGENCY
        assert transition.from_mode == OperationMode.NORMAL
        assert transition.to_mode == OperationMode.EMERGENCY
        assert transition.trigger_reason == "testing"

    def test_can_process_question_normal_mode(self):
        """Test question processing check in normal mode."""
        with patch.object(
            self.manager.budget_manager, "get_budget_status"
        ) as mock_status:
            mock_status.return_value = Mock(remaining=10.0)

            can_process, reason = self.manager.can_process_question("low")
            assert can_process is True
            assert "can be processed" in reason

    def test_can_process_question_emergency_mode(self):
        """Test question processing check in emergency mode."""
        self.manager.current_mode = OperationMode.EMERGENCY

        with patch.object(
            self.manager.budget_manager, "get_budget_status"
        ) as mock_status:
            mock_status.return_value = Mock(remaining=1.0)

            # Low priority should be rejected
            can_process, reason = self.manager.can_process_question("low")
            assert can_process is False
            assert "Emergency mode" in reason

            # High priority should be accepted
            can_process, reason = self.manager.can_process_question("high")
            assert can_process is True

    def test_can_process_question_no_budget(self):
        """Test question processing when no budget remains."""
        with patch.object(
            self.manager.budget_manager, "get_budget_status"
        ) as mock_status:
            mock_status.return_value = Mock(remaining=0.0)

            can_process, reason = self.manager.can_process_question()
            assert can_process is False
            assert "No budget remaining" in reason

    def test_get_model_for_task_with_complexity(self):
        """Test model selection with complexity analysis."""
        with patch.object(
            self.manager.complexity_analyzer, "get_model_for_task"
        ) as mock_get_model:
            mock_assessment = Mock()
            mock_assessment.level.value = "medium"
            mock_get_model.return_value = "openai/gpt-4o"

            model = self.manager.get_model_for_task("forecast", mock_assessment)

            mock_get_model.assert_called_once_with(
                "forecast", mock_assessment, "normal"
            )
            assert model == "openai/gpt-4o"

    def test_get_model_for_task_emergency_override(self):
        """Test model selection override in emergency mode."""
        self.manager.current_mode = OperationMode.EMERGENCY

        model = self.manager.get_model_for_task("forecast")

        assert model == "openai/gpt-4o-mini"

    def test_get_processing_limits(self):
        """Test getting processing limits for current mode."""
        limits = self.manager.get_processing_limits()

        expected_keys = [
            "max_questions_per_batch",
            "max_retries",
            "timeout_seconds",
            "enable_complexity_analysis",
            "skip_low_priority_questions",
        ]

        for key in expected_keys:
            assert key in limits

        assert limits["max_questions_per_batch"] == 10  # Normal mode default

    def test_get_graceful_degradation_strategy(self):
        """Test graceful degradation strategy generation."""
        with patch.object(
            self.manager.budget_manager, "get_budget_status"
        ) as mock_status:
            # Test normal utilization
            mock_status.return_value = Mock(utilization_percentage=50.0)
            strategy = self.manager.get_graceful_degradation_strategy()

            assert strategy["current_mode"] == "normal"
            assert "Normal operation" in strategy["actions"][0]

            # Test high utilization
            mock_status.return_value = Mock(utilization_percentage=97.0)
            strategy = self.manager.get_graceful_degradation_strategy()

            assert "Process only critical priority questions" in strategy["actions"]
            assert "Use minimal model" in strategy["actions"][1]

    def test_mode_history_tracking(self):
        """Test mode transition history tracking."""
        initial_count = len(self.manager.mode_transitions)

        self.manager.force_mode_transition(OperationMode.CONSERVATIVE, "test1")
        self.manager.force_mode_transition(OperationMode.EMERGENCY, "test2")

        assert len(self.manager.mode_transitions) == initial_count + 2

        history = self.manager.get_mode_history()
        assert len(history) == initial_count + 2
        assert history[-1].to_mode == OperationMode.EMERGENCY
        assert history[-1].trigger_reason == "test2"

    def test_reset_mode_history(self):
        """Test resetting mode history."""
        self.manager.force_mode_transition(OperationMode.CONSERVATIVE, "test")
        assert len(self.manager.mode_transitions) > 0

        self.manager.reset_mode_history()
        assert len(self.manager.mode_transitions) == 0
