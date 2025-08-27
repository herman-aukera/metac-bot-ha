"""
Integration tests for operation modes with enhanced LLM configuration.
"""
import pytest
from unittest.mock import Mock, patch

from src.infrastructure.config.operation_modes import OperationMode
from src.infrastructure.config.enhanced_llm_config import EnhancedLLMConfig


class TestOperationModesIntegration:
    """Test operation modes integration with enhanced LLM config."""

    @patch('src.infrastructure.config.enhanced_llm_config.api_key_manager')
    @patch('src.infrastructure.config.enhanced_llm_config.budget_manager')
    @patch('src.infrastructure.config.enhanced_llm_config.operation_mode_manager')
    def test_llm_config_uses_operation_modes(self, mock_operation_manager, mock_budget, mock_api_keys):
        """Test that enhanced LLM config integrates with operation modes."""
        # Setup mocks
        mock_api_keys.get_api_key.return_value = "test-key"
        mock_budget.get_budget_status.return_value = Mock(
            utilization_percentage=50.0,
            status_level="normal"
        )
        mock_operation_manager.check_and_update_mode.return_value = (False, None)
        mock_operation_manager.get_model_for_task.return_value = "openai/gpt-4o-mini"
        mock_operation_manager.get_processing_limits.return_value = {
            "max_retries": 2,
            "timeout_seconds": 60,
            "enable_complexity_analysis": True
        }
        mock_operation_manager.current_mode = OperationMode.NORMAL

        # Create enhanced LLM config
        config = EnhancedLLMConfig()

        # Test getting LLM for task
        llm = config.get_llm_for_task("research")

        # Verify operation mode manager was called
        mock_operation_manager.check_and_update_mode.assert_called_once()
        mock_operation_manager.get_model_for_task.assert_called_once_with("research", None)
        mock_operation_manager.get_processing_limits.assert_called_once()

    @patch('src.infrastructure.config.enhanced_llm_config.api_key_manager')
    @patch('src.infrastructure.config.enhanced_llm_config.budget_manager')
    @patch('src.infrastructure.config.enhanced_llm_config.operation_mode_manager')
    def test_mode_change_during_task_request(self, mock_operation_manager, mock_budget, mock_api_keys):
        """Test automatic mode change during task request."""
        # Setup mocks
        mock_api_keys.get_api_key.return_value = "test-key"
        mock_budget.get_budget_status.return_value = Mock(
            utilization_percentage=85.0,
            status_level="conservative"
        )

        # Mock mode change
        mock_transition = Mock()
        mock_transition.from_mode.value = "normal"
        mock_transition.to_mode.value = "conservative"
        mock_operation_manager.check_and_update_mode.return_value = (True, mock_transition)
        mock_operation_manager.get_model_for_task.return_value = "openai/gpt-4o-mini"
        mock_operation_manager.get_processing_limits.return_value = {
            "max_retries": 2,
            "timeout_seconds": 60,
            "enable_complexity_analysis": True
        }
        mock_operation_manager.current_mode = OperationMode.CONSERVATIVE

        # Create enhanced LLM config and request LLM
        config = EnhancedLLMConfig()
        llm = config.get_llm_for_task("forecast")

        # Verify mode change was detected and handled
        mock_operation_manager.check_and_update_mode.assert_called_once()

    @patch('src.infrastructure.config.enhanced_llm_config.api_key_manager')
    @patch('src.infrastructure.config.enhanced_llm_config.operation_mode_manager')
    def test_question_processing_check(self, mock_operation_manager, mock_api_keys):
        """Test question processing check integration."""
        # Setup mocks
        mock_api_keys.get_api_key.return_value = "test-key"
        mock_operation_manager.can_process_question.return_value = (True, "Can process")

        # Create enhanced LLM config
        config = EnhancedLLMConfig()

        # Test question processing check
        can_process, reason = config.can_process_question("high")

        assert can_process is True
        assert reason == "Can process"
        mock_operation_manager.can_process_question.assert_called_once_with("high")

    @patch('src.infrastructure.config.enhanced_llm_config.api_key_manager')
    @patch('src.infrastructure.config.enhanced_llm_config.operation_mode_manager')
    def test_configuration_status_logging(self, mock_operation_manager, mock_api_keys):
        """Test configuration status logging includes operation modes."""
        # Setup mocks
        mock_api_keys.get_api_key.return_value = "test-key"
        mock_operation_manager.log_mode_status = Mock()
        mock_operation_manager.get_model_for_task.return_value = "openai/gpt-4o-mini"

        # Create enhanced LLM config
        config = EnhancedLLMConfig()

        # Test logging
        config.log_configuration_status()

        # Verify operation mode status was logged
        mock_operation_manager.log_mode_status.assert_called_once()
